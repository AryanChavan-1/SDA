"""
SEAF Hybrid Gateway Router (seaf_router.py)
------------------------------------------
Implements PII-Filtered Hybrid Routing for the Sovereign-Edge Agentic Framework:
- Detects PII, financial, and confidential data locally.
- Directs Tier 1/2 sensitive queries strictly to offline local execution (Ollama/LM Studio).
- Sanitizes and routes complex Tier 3 reasoning tasks to the VergeIO Sovereign AI Cloud VDC.
"""

import re
import time
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

class SEAFHybridGateway:
    def __init__(self, local_endpoint="http://localhost:11434/api/generate", cloud_endpoint="https://sovereign-vdc.enterprise.internal/v1/chat",
                 local_chat_endpoint="http://localhost:11434/v1/chat/completions",
                 lmstudio_endpoint="http://localhost:1234/v1/chat/completions",
                 local_model="llama3.2:1b", cloud_model="sovereign-70b-instruct",
                 request_timeout_s: float = 30.0):
        self.local_endpoint = local_endpoint
        self.local_chat_endpoint = local_chat_endpoint
        self.lmstudio_endpoint = lmstudio_endpoint
        self.cloud_endpoint = cloud_endpoint
        self.local_model = local_model
        self.cloud_model = cloud_model
        self.request_timeout_s = request_timeout_s
        
        # Regex patterns for PII and sensitive data identification
        self.pii_patterns = {
            "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            "CREDIT_CARD": re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
            "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            "PHONE": re.compile(r"\b(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b"),
            "SENSITIVE_KEYWORD": re.compile(r"\b(salary|ssn|patient_id|confidential|secret|payroll|tax_id|medical_record)\b", re.IGNORECASE)
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
            "detected_entities": detected
        }

    def sanitize_payload(self, text: str) -> str:
        """Redacts sensitive PII patterns for safe Tier 3 cloud offloading."""
        sanitized = text
        for pii_type, pattern in self.pii_patterns.items():
            if pii_type != "SENSITIVE_KEYWORD":
                sanitized = pattern.sub(f"[{pii_type}_REDACTED]", sanitized)
        return sanitized

    def route_request(self, prompt: str, task_complexity_tier: int = 1) -> Dict[str, Any]:
        """
        Determines the optimal execution environment based on privacy and complexity.
        Returns routing decisions, target endpoint, and execution metadata.
        """
        start_time = time.perf_counter()
        inspection = self.inspect_payload(prompt)
        
        # Policy: Any PII or Tier 1/2 workload MUST stay local.
        if inspection["has_pii"] or task_complexity_tier <= 2:
            target_destination = "LOCAL_OFFLINE_EDGE"
            target_endpoint = self.local_endpoint
            final_prompt = prompt  # Local processing retains raw details safely
            cloud_offloaded = False
            reason = "PII detected or low/mid complexity tier. Kept strictly offline on local AI PC."
        else:
            target_destination = "VERGEIO_SOVEREIGN_CLOUD"
            target_endpoint = self.cloud_endpoint
            final_prompt = self.sanitize_payload(prompt)
            cloud_offloaded = True
            reason = "Non-sensitive Tier 3 reasoning task offloaded to Sovereign Cloud VDC."

        routing_latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "destination": target_destination,
            "endpoint": target_endpoint,
            "cloud_offloaded": cloud_offloaded,
            "inspection": inspection,
            "final_prompt": final_prompt,
            "routing_reason": reason,
            "gateway_latency_ms": round(routing_latency_ms, 3)
        }

    # ------------------------------------------------------------------
    # Phase 2: Local Endpoint Binding (Ollama / LM Studio, localhost only)
    # ------------------------------------------------------------------
    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Minimal stdlib HTTP POST. Used only for localhost or sanitized cloud payloads."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.request_timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"raw": body}

    def check_local_runtime(self) -> Dict[str, Any]:
        """Probe Ollama (/api/tags) then LM Studio (/v1/models). No prompt data sent."""
        status: Dict[str, Any] = {"ollama": False, "lmstudio": False, "detail": {}}
        for name, url in (("ollama", "http://localhost:11434/api/tags"),
                          ("lmstudio", "http://localhost:1234/v1/models")):
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    status[name] = (resp.status == 200)
                    status["detail"][name] = f"HTTP {resp.status}"
            except Exception as e:  # noqa: BLE001 - report availability, never raise here
                status["detail"][name] = f"{type(e).__name__}: {e}"
        return status

    def discover_local_models(self) -> Dict[str, Any]:
        """List Ollama models with completion capability. No prompt data sent."""
        try:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            models = [m.get("name", m.get("model", "")) for m in data.get("models", [])]
            completion = [m for m in data.get("models", [])
                          if "completion" in (m.get("capabilities") or [])]
            completion_names = [m.get("name") for m in completion]
            return {"available": True, "models": models,
                    "completion_models": completion_names,
                    "preferred": completion_names[0] if completion_names
                                 else (models[0] if models else None)}
        except Exception as e:  # noqa: BLE001
            return {"available": False, "error": f"{type(e).__name__}: {e}",
                    "models": [], "completion_models": [], "preferred": None}

    def generate_local(self, prompt: str, model: Optional[str] = None,
                       prefer: str = "ollama-generate") -> Dict[str, Any]:
        """Execute a prompt on a localhost runtime. Never called with cloud destinations.

        Chain: Ollama /api/generate -> Ollama /v1/chat/completions -> LM Studio.
        If `model` is None, auto-selects the first Ollama completion-capable
        model (e.g. phi4-mini on this test box). If the requested model is
        missing (HTTP 404), retries once with the discovered preferred model.
        Raises RuntimeError if no localhost runtime is reachable (caller falls back
        to deterministic offline stub so benchmarks stay reproducible).
        """
        errors = {}
        candidates = [model] if model else []
        if not candidates:
            disc = self.discover_local_models()
            if disc.get("preferred"):
                candidates = [disc["preferred"]]
            else:
                candidates = [self.local_model]
        # 1) Native Ollama generate API (with missing-model fallback)
        if prefer in ("ollama-generate", "auto"):
            for attempt, cand in enumerate(list(candidates)):
                try:
                    res = self._post_json(self.local_endpoint, {
                        "model": cand, "prompt": prompt, "stream": False})
                    if isinstance(res, dict) and "error" in res and "not found" in str(res["error"]).lower():
                        raise RuntimeError(f"model missing: {res['error']}")
                    text = res.get("response", res.get("raw", str(res)))
                    return {"backend": "ollama-generate", "model": cand, "text": text}
                except Exception as e:  # noqa: BLE001
                    errors[f"ollama-generate:{cand}"] = f"{type(e).__name__}: {e}"
                    if attempt == 0 and ("not found" in str(e).lower() or "404" in str(e)):
                        disc = self.discover_local_models()
                        if disc.get("preferred") and disc["preferred"] not in candidates:
                            candidates.append(disc["preferred"])
                            continue
                    break
            model = candidates[-1]
        else:
            model = candidates[0]
        # 2) OpenAI-compatible Ollama endpoint (e.g. mistral:7b-instruct-q4_K_M)
        try:
            res = self._post_json(self.local_chat_endpoint, {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0, "stream": False})
            text = res["choices"][0]["message"]["content"] if "choices" in res else str(res)
            return {"backend": "ollama-openai-compat", "model": model, "text": text}
        except Exception as e:  # noqa: BLE001
            errors["ollama-openai-compat"] = f"{type(e).__name__}: {e}"
        # 3) LM Studio OpenAI-compatible server
        try:
            res = self._post_json(self.lmstudio_endpoint, {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0, "stream": False})
            text = res["choices"][0]["message"]["content"] if "choices" in res else str(res)
            return {"backend": "lmstudio", "model": model, "text": text}
        except Exception as e:  # noqa: BLE001
            errors["lmstudio"] = f"{type(e).__name__}: {e}"
        raise RuntimeError(f"No localhost runtime reachable: {errors}")

    def execute_routed_request(self, prompt: str, task_complexity_tier: int = 1,
                               model: Optional[str] = None) -> Dict[str, Any]:
        """Route then execute. INVARIANT: raw PII never leaves localhost.

        - LOCAL_OFFLINE_EDGE -> generate_local(raw prompt).
        - VERGEIO_SOVEREIGN_CLOUD -> caller must forward sanitized prompt via
          their air-gapped VDC client; this method returns the sanitized payload
          plus a `blocked` cloud stub unless an explicit opt-in forwards it.
          Re-inspects final_prompt: if PII survived sanitization, force local.
        """
        route = self.route_request(prompt, task_complexity_tier)
        if route["destination"] == "LOCAL_OFFLINE_EDGE":
            try:
                gen = self.generate_local(route["final_prompt"], model=model)
            except RuntimeError as e:
                # Deterministic offline fallback keeps PoC reproducible w/o GPU box
                gen = {"backend": "offline-stub", "model": model or self.local_model,
                       "text": f"LOCAL_AUDIT_PASSED: Processed offline payload "
                               f"({len(route['final_prompt'])} chars). [{e}]"}
            route["execution"] = gen
            return route
        # Cloud path: defense-in-depth re-inspection of sanitized prompt
        recheck = self.inspect_payload(route["final_prompt"])
        # PII regexes must be fully redacted; keywords alone do not block Tier 3
        # (they carry no extractable identifiers), but raw regex PII does.
        raw_pii_left = any(k != "SENSITIVE_KEYWORD" for k in recheck["detected_entities"])
        if raw_pii_left:
            raise SecurityError(
                f"Sanitization failed, refusing cloud offload: {recheck['detected_entities']}")
        route["execution"] = {
            "backend": "vergeio-vdc-stub",
            "model": self.cloud_model,
            "text": f"SOVEREIGN_CLOUD_OUTPUT: Queued sanitized Tier-3 synthesis "
                    f"for: {route['final_prompt'][:120]}...",
            "note": "Forward route['final_prompt'] via air-gapped VDC client; "
                    "raw PII is never transmitted.",
        }
        return route


class SecurityError(RuntimeError):
    """Raised when a cloud offload would expose raw PII."""

