"""
seaf_agent.telemetry — Live Hardware Telemetry for AI PCs / Edge Laptops.
Reads:
- Host RAM (total, used, available, percent)
- Process RSS (memory occupied by current agent process)
- CPU utilization
- GPU VRAM (used, total) via cached nvidia-smi / xpu-smi probes
"""

import os
import shutil
import subprocess
import time
from typing import Dict, Any, Tuple
import psutil

from seaf_agent.config import RAM_PRESSURE_THRESHOLD_PCT, GPU_CACHE_TTL_SEC


class SystemTelemetry:
    def __init__(self, pressure_threshold_pct: float = RAM_PRESSURE_THRESHOLD_PCT,
                 gpu_cache_ttl_s: float = GPU_CACHE_TTL_SEC):
        self.pressure_threshold_pct = pressure_threshold_pct
        self.gpu_cache_ttl_s = gpu_cache_ttl_s
        self._gpu_cache: Dict[str, Any] = {
            "ts": 0.0,
            "value": (0.0, 0.0, "uninitialized")
        }

    def _probe_gpu_mem(self) -> Tuple[float, float, str]:
        """
        Probe GPU memory (used_mb, total_mb, source).
        Cached with TTL to prevent subprocess latency from impacting scheduling.
        """
        now = time.monotonic()
        if (now - self._gpu_cache["ts"]) < self.gpu_cache_ttl_s:
            return self._gpu_cache["value"]

        # 1. NVIDIA probe
        if shutil.which("nvidia-smi"):
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
                    capture_output=True, text=True, timeout=2.0
                )
                if res.returncode == 0 and res.stdout.strip():
                    line = res.stdout.strip().splitlines()[0]
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 2:
                        used = float(parts[0])
                        total = float(parts[1])
                        val = (used, total, "nvidia-smi")
                        self._gpu_cache = {"ts": now, "value": val}
                        return val
            except Exception:
                pass

        # 2. Intel XPU probe
        if shutil.which("xpu-smi"):
            try:
                res = subprocess.run(
                    ["xpu-smi", "discovery"],
                    capture_output=True, text=True, timeout=2.0
                )
                if res.returncode == 0:
                    val = (0.0, 0.0, "xpu-smi-present")
                    self._gpu_cache = {"ts": now, "value": val}
                    return val
            except Exception:
                pass

        val = (0.0, 0.0, "no-gpu-detected")
        self._gpu_cache = {"ts": now, "value": val}
        return val

    def sample(self) -> Dict[str, Any]:
        """Sample host RAM, CPU, RSS, and GPU metrics."""
        vm = psutil.virtual_memory()
        proc_rss = psutil.Process().memory_info().rss / (1024 * 1024)
        cpu_pct = psutil.cpu_percent(interval=None)
        gpu_used_mb, gpu_total_mb, gpu_source = self._probe_gpu_mem()

        ram_pct = float(vm.percent)
        under_pressure = ram_pct >= self.pressure_threshold_pct

        gpu_pct = 0.0
        if gpu_total_mb > 0:
            gpu_pct = round((gpu_used_mb / gpu_total_mb) * 100.0, 1)

        return {
            "ram_total_mb": round(vm.total / (1024 * 1024), 1),
            "ram_available_mb": round(vm.available / (1024 * 1024), 1),
            "ram_used_mb": round(vm.used / (1024 * 1024), 1),
            "ram_percent": ram_pct,
            "cpu_percent": float(cpu_pct),
            "process_rss_mb": round(proc_rss, 2),
            "gpu_used_mb": round(gpu_used_mb, 1),
            "gpu_total_mb": round(gpu_total_mb, 1),
            "gpu_percent": gpu_pct,
            "gpu_source": gpu_source,
            "under_pressure": under_pressure,
            "timestamp": time.time(),
        }


# Global singleton instance for easy import
telemetry = SystemTelemetry()
