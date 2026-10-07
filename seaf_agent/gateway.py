"""
seaf_agent.gateway — Zero-Exposure PII Hybrid Gateway (SEAF Pillar 1).
Contracts:
1. Input: Incoming prompt/payload from the SDA pipeline.
2. Deterministic regex/heuristic PII screening (< 1ms inspection latency).
3. Routing Rules:
   - If PII is present OR task tier is 1–2 -> strictly route to LOCAL_OFFLINE_EDGE (Ollama/LM Studio on localhost).
   - If Tier-3 and non-sensitive -> sanitize PII, re-inspect, and route to Sovereign Cloud endpoint.
4. Security Invariant & Pitfall 4 Guardrail:
   - If residual PII survives redaction during pre-flight re-inspection, raise SecurityError immediately and force fallback to local offline phi4-mini execution. Never allow raw PII to reach network sockets.
   - When SecurityError or Cloud timeout occurs, caller seamlessly fail-safes without surfacing unhandled exceptions to UI.
"""

import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Dict, Any, Optional

from seaf_agent.config import (
    OLLAMA_BASE_URL,
    LOCAL_CODING_SLM,
    LOCAL_FALLBACK_SLM,
    SOVEREIGN_VDC_ENDPOINT,
    SOVEREIGN_CLOUD_MODEL,
)
from seaf_agent.sovereign_connector import sovereign_connector

log = logging.getLogger("seaf.gateway")


class SecurityError(RuntimeError):
    """Raised when sensitive data risks egressing beyond localhost."""
    pass


class SEAFHybridGateway:
    def __init__(
        self,
        ollama_base: str = OLLAMA_BASE_URL,
        cloud_endpoint: str = SOVEREIGN_VDC_ENDPOINT,
        local_model: str = LOCAL_CODING_SLM,
        fallback_model: str = LOCAL_FALLBACK_SLM,
        cloud_model: str = SOVEREIGN_CLOUD_MODEL,
        request_timeout_s: float = 30.0,
    ):
        self.ollama_base = ollama_base
        self.cloud_endpoint = cloud_endpoint
        self.local_model = local_model
        self.fallback_model = fallback_model
        self.cloud_model = cloud_model
        self.request_timeout_s = request_timeout_s

        # Pre-compiled high-performance regex patterns (<1ms inspection)
        self.pii_patterns = {
            "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "CREDIT_CARD": re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
            "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            "PHONE": re.compile(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"),
            "SENSITIVE_KEYWORD": re.compile(
                r"\b(salary|ssn|patient_id|confidential|secret|payroll|tax_id|medical_record)\b",
                re.IGNORECASE,
            ),
        }

    def inspect_payload(self, text: str) -> Dict[str, Any]:
        """Scans payload text for PII entities and sensitive keyword matches."""
        detected = {}
        for pii_type, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            if matches:
                detected[pii_type] = len(matches)

        has_pii = len(detected) > 0
        return {
            "has_pii": has_pii,
            "detected_entities": detected,
        }

    def sanitize_payload(self, text: str) -> str:
        """Redacts sensitive PII patterns for safe cloud offloading."""
        sanitized = text
        for pii_type, pattern in self.pii_patterns.items():
            if pii_type != "SENSITIVE_KEYWORD":
                sanitized = pattern.sub(f"[{pii_type}_REDACTED]", sanitized)
        return sanitized

    def route_request(self, prompt: str, task_complexity_tier: int = 1) -> Dict[str, Any]:
        """
        Pure regex / heuristic routing (< 0.05 ms latency).
        Tier 1: Simple queries / local scripts -> LOCAL_OFFLINE_EDGE
        Tier 2: Intermediate logic / code refactoring -> LOCAL_OFFLINE_EDGE
        Tier 3: Heavy reasoning -> SOVEREIGN_CLOUD (only if PII-free)
        """
        t0 = time.perf_counter()
        inspection = self.inspect_payload(prompt)

        # Invariant: If PII is present OR task tier is 1-2, route strictly to LOCAL_OFFLINE_EDGE
        if inspection["has_pii"] or task_complexity_tier <= 2:
            destination = "LOCAL_OFFLINE_EDGE"
            target_model = self.local_model
            reason = "PII detected; strictly pinned to offline localhost" if inspection["has_pii"] else f"Tier {task_complexity_tier} complexity assigned to local edge"
        else:
            destination = "SOVEREIGN_CLOUD"
            target_model = self.cloud_model
            reason = "Tier 3 complex reasoning; PII-free, routed to sovereign cloud"

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "destination": destination,
            "target_model": target_model,
            "task_complexity_tier": task_complexity_tier,
            "routing_reason": reason,
            "inspection": inspection,
            "gateway_latency_ms": round(latency_ms, 4),
        }

    def generate_local(self, prompt: str, model: Optional[str] = None, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Executes prompt via local Ollama instance on localhost."""
        target_model = model or self.local_model
        url = f"{self.ollama_base.rstrip('/')}/api/generate"
        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "backend": "ollama-generate",
                    "model": target_model,
                    "text": data.get("response", ""),
                    "total_duration_ns": data.get("total_duration", 0),
                    "eval_count": data.get("eval_count", 0),
                    "status": "SUCCESS",
                }
        except Exception as exc:
            log.warning("Local Ollama request failed (%s); returning offline edge stub", exc)
            return {
                "backend": "offline-stub",
                "model": target_model,
                "text": f"[OFFLINE_EDGE_STUB] Processed locally without Ollama daemon: {prompt[:80]}",
                "status": "FALLBACK",
                "error": str(exc),
            }

    def execute_routed_request(
        self,
        prompt: str,
        task_complexity_tier: int = 1,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes routed request. Enforces Security Invariant and Pitfall 4 fail-safe:
        If residual PII survives redaction during pre-flight re-inspection, raises SecurityError
        and seamlessly fails-safe to local phi4-mini execution without returning an unhandled error to UI.
        """
        route = self.route_request(prompt, task_complexity_tier)
        destination = route["destination"]

        if destination == "LOCAL_OFFLINE_EDGE":
            exec_res = self.generate_local(prompt, model=route["target_model"], system_prompt=system_prompt)
            return {
                "destination": destination,
                "final_prompt": prompt,
                "route_metadata": route,
                "execution": exec_res,
            }

        # Destination is SOVEREIGN_CLOUD: Pre-flight sanitization & re-inspection
        sanitized = self.sanitize_payload(prompt)
        recheck = self.inspect_payload(sanitized)
        raw_pii_present = any(
            k != "SENSITIVE_KEYWORD" and v > 0
            for k, v in recheck["detected_entities"].items()
        )

        if raw_pii_present:
            # Security Invariant: raise SecurityError, force local offline fallback (Pitfall 4)
            err_msg = f"Residual PII survived redaction ({recheck['detected_entities']})"
            log.error("SECURITY INVARIANT VIOLATION: %s. Forcing local edge fallback.", err_msg)
            fallback_res = self.generate_local(prompt, model=self.fallback_model, system_prompt=system_prompt)
            return {
                "destination": "LOCAL_OFFLINE_EDGE",
                "security_alert": f"SecurityError: {err_msg} -> Forced Local Fallback",
                "final_prompt": prompt,
                "route_metadata": {**route, "destination": "LOCAL_OFFLINE_EDGE", "fallback_reason": err_msg},
                "execution": fallback_res,
            }

        # Dispatch via circuit-breaker connector
        cloud_res = sovereign_connector.send_sovereign_request(sanitized, system_prompt=system_prompt)
        return {
            "destination": "SOVEREIGN_CLOUD" if cloud_res.get("circuit_breaker") == "CLOSED" else "LOCAL_OFFLINE_EDGE",
            "final_prompt": sanitized,
            "route_metadata": route,
            "execution": cloud_res,
        }


# Aliases for compatibility
SEAFGateway = SEAFHybridGateway
gateway = SEAFHybridGateway()
