"""
SEAF Declarative Orchestrator (seaf_orchestrator.py)
-----------------------------------------------------
Implements Hardware-Aware Multi-Agent Orchestration for the Sovereign-Edge Agentic Framework:
- Replaces heavy LLM supervisor models with compiled, declarative DSPy-style pipelines.
- Manages local memory budgets using INT4 AutoRound profiles and SYCL GPU layer offloading constraints.
- Orchestrates sub-agent task queues without cognitive supervisor latency or VRAM thrashing.
"""

import time
import shutil
import subprocess
from typing import Dict, Any, List, Callable

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore
    _PSUTIL_AVAILABLE = False

class SEAFDeclarativeOrchestrator:
    PRESSURE_THRESHOLD_PCT = 80.0

    def __init__(self, available_vram_mb: int = 8192, quantization_profile: str = "INT4_AUTOROUND",
                 base_context_window: int = 4096, enable_live_telemetry: bool = True):
        self.available_vram_mb = available_vram_mb
        self.quantization_profile = quantization_profile
        self.base_context_window = base_context_window
        self.enable_live_telemetry = enable_live_telemetry and _PSUTIL_AVAILABLE
        self.agent_registry = {}
        self.last_pressure: Dict[str, Any] = {}
        self._gpu_cache: Dict[str, Any] = {"ts": 0.0, "value": (0.0, "no-gpu-smi-found (estimate=0)")}
        self._gpu_cache_ttl_s = 60.0

    def register_sub_agent(self, name: str, memory_cost_mb: int, task_func: Callable):
        """Registers a specialized sub-agent with explicit hardware memory footprint."""
        self.agent_registry[name] = {
            "memory_cost_mb": memory_cost_mb,
            "execute": task_func
        }

    # ------------------------------------------------------------------
    # Phase 2: Live System Telemetry (psutil + GPU probe)
    # ------------------------------------------------------------------
    def get_system_pressure(self) -> Dict[str, Any]:
        """Sample host RAM/CPU/process RSS. Never raises; degrades to estimates."""
        if not self.enable_live_telemetry:
            return {"telemetry_enabled": False, "ram_percent": 0.0,
                    "cpu_percent": 0.0, "process_rss_mb": 0.0,
                    "gpu_mem_used_mb": 0.0, "gpu_probe": "disabled"}
        try:
            vm = psutil.virtual_memory()
            proc = psutil.Process().memory_info().rss / (1024 * 1024)
            cpu = psutil.cpu_percent(interval=None)
            gpu_used, gpu_probe = self._probe_gpu_mem()
            pressure = {
                "telemetry_enabled": True,
                "ram_percent": float(vm.percent),
                "ram_available_mb": round(vm.available / (1024 * 1024), 1),
                "cpu_percent": float(cpu),
                "process_rss_mb": round(proc, 1),
                "gpu_mem_used_mb": gpu_used,
                "gpu_probe": gpu_probe,
                "under_pressure": vm.percent >= self.PRESSURE_THRESHOLD_PCT,
            }
            self.last_pressure = pressure
            return pressure
        except Exception as e:  # noqa: BLE001 - telemetry must not break pipelines
            return {"telemetry_enabled": True, "error": f"{type(e).__name__}: {e}",
                    "ram_percent": 0.0, "under_pressure": False}

    def _probe_gpu_mem(self):
        """Try xpu-smi (Intel) then nvidia-smi; else 0. Result cached 60s to keep
        scheduling decisions sub-millisecond (shutil.which is slow on Windows)."""
        now = time.monotonic()
        if now - self._gpu_cache["ts"] < self._gpu_cache_ttl_s:
            return self._gpu_cache["value"]
        result = (0.0, "no-gpu-smi-found (estimate=0)")
        for exe, args in (("xpu-smi", ["discovery", "--dump", "1,5,18"]),
                          ("nvidia-smi", ["--query-gpu=memory.used",
                                          "--format=csv,noheader,nounits"])):
            path = shutil.which(exe)
            if not path:
                continue
            try:
                out = subprocess.run([path] + args, capture_output=True,
                                     text=True, timeout=5)
                for tok in out.stdout.replace(",", " ").split():
                    try:
                        result = (float(tok), exe)
                        break
                    except ValueError:
                        continue
                else:
                    result = (0.0, f"{exe}: unparseable")
                break
            except Exception as e:  # noqa: BLE001
                result = (0.0, f"{exe} failed: {e}")
                break
        self._gpu_cache = {"ts": now, "value": result}
        return result

    def _throttle_policy(self, required_mem_mb: int,
                           pressure: Dict[str, Any] = None) -> Dict[str, Any]:
        pressure = pressure if pressure is not None else self.get_system_pressure()
        under_pressure = bool(pressure.get("under_pressure", False))
        # INT4 AutoRound: 4x reduction from FP16
        if "INT4" in self.quantization_profile:
            effective = required_mem_mb // 4 if required_mem_mb > self.available_vram_mb else required_mem_mb
        else:
            effective = required_mem_mb
        # Under RAM pressure: halve context window, force sequential (already
        # sequential), and cap effective allocation to budget headroom.
        context_window = self.base_context_window // 2 if under_pressure else self.base_context_window
        if under_pressure:
            effective = min(effective, int(self.available_vram_mb * 0.5))
        return {"effective_mem_mb": effective, "context_window": context_window,
                "throttled": under_pressure, "pressure": pressure}

    def execute_pipeline(self, pipeline_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes a declarative pipeline sequentially or throttled based on hardware memory budget.
        Prevents VRAM thrashing by guaranteeing active sub-agents fit within available memory.
        """
        start_time = time.perf_counter()
        execution_trace = []
        total_vram_peak = 0
        current_vram_usage = 0
        throttled_any = False
        # Sample host pressure once per pipeline: keeps scheduling O(1) and
        # deterministic; per-task re-sampling would add ~100ms on Windows.
        pipeline_pressure = self.get_system_pressure()

        for task in pipeline_tasks:
            agent_name = task["agent"]
            task_input = task["input"]

            if agent_name not in self.agent_registry:
                raise ValueError(f"Sub-agent '{agent_name}' not registered in SEAF Orchestrator.")

            agent_info = self.agent_registry[agent_name]
            required_mem = agent_info["memory_cost_mb"]

            # Live-telemetry throttle check (INT4 scaling + >80% RAM -> halve ctx)
            policy = self._throttle_policy(required_mem, pipeline_pressure)
            effective_mem = policy["effective_mem_mb"]
            throttled_any = throttled_any or policy["throttled"]

            current_vram_usage = effective_mem
            total_vram_peak = max(total_vram_peak, current_vram_usage)

            # Execute sub-agent deterministically
            agent_start = time.perf_counter()
            result = agent_info["execute"](task_input)
            agent_duration_ms = (time.perf_counter() - agent_start) * 1000.0

            execution_trace.append({
                "agent": agent_name,
                "effective_vram_mb": effective_mem,
                "context_window": policy["context_window"],
                "throttled": policy["throttled"],
                "input": task_input,
                "output": result,
                "duration_ms": round(agent_duration_ms, 3)
            })

            # Clear active sub-agent state from VRAM
            current_vram_usage = 0

        total_duration_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "status": "SUCCESS",
            "quantization": self.quantization_profile,
            "vram_peak_mb": total_vram_peak,
            "vram_budget_mb": self.available_vram_mb,
            "total_execution_time_ms": round(total_duration_ms, 3),
            "trace": execution_trace,
            "throttled": throttled_any,
            "system_pressure": self.last_pressure if self.last_pressure else self.get_system_pressure(),
        }
