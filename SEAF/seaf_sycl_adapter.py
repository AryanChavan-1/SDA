"""
SEAF llama.cpp SYCL Hardware Adapter (seaf_sycl_adapter.py)
------------------------------------------------------------
Phase 2: explicit GPU layer offloading and SYCL backend bindings for
Intel integrated/discrete GPUs (AI PCs with unified memory).

Design notes:
- Pure stdlib: no hard dependency on llama.cpp binary or Intel XPU libs.
- Probes for `llama-server` / `llama-cli` via PATH; otherwise returns the
  exact command the operator should run (air-gapped friendly).
- AutoRound INT4 memory model: FP16 weights / 4 + KV-cache + overhead must
  fit inside the unified-memory budget (default 8192 MB).
"""

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


QUANT_PROFILES = {
    "INT4_AUTOROUND": {"scale": 4.0, "gguf": "Q4_K_M", "bpp": 0.5},
    "Q4_K_M": {"scale": 4.0, "gguf": "Q4_K_M", "bpp": 0.5},
    "Q5_K_M": {"scale": 3.2, "gguf": "Q5_K_M", "bpp": 0.625},
    "Q8_0": {"scale": 2.0, "gguf": "Q8_0", "bpp": 1.0},
    "F16": {"scale": 1.0, "gguf": "F16", "bpp": 2.0},
}

# Approximate FP16 parameter footprints (GB) for common edge models.
MODEL_FP16_GB = {
    "llama3.2:1b": 2.0,
    "llama3.2:3b": 6.0,
    "mistral:7b-instruct-q4_K_M": 14.0,
    "mistral-7b": 14.0,
    "qwen2.5:7b": 14.0,
    "phi-3-mini": 7.6,
}


@dataclass
class SYCLAdapterConfig:
    model_id: str = "llama3.2:1b"
    gguf_path: str = "./models/llama3.2-1b-Q4_K_M.gguf"
    vram_budget_mb: int = 8192
    quantization_profile: str = "INT4_AUTOROUND"
    ctx_size: int = 4096
    n_gpu_layers: Optional[int] = None  # None = auto (all layers)
    backend: str = "sycl"               # sycl | cuda | cpu
    host: str = "127.0.0.1"
    port: int = 8080
    extra_args: List[str] = field(default_factory=list)


class SYCLHardwareAdapter:
    def __init__(self, config: Optional[SYCLAdapterConfig] = None, **kwargs):
        self.config = config or SYCLAdapterConfig(**kwargs)

    # -- detection -----------------------------------------------------
    def detect_accelerator(self) -> Dict[str, Any]:
        """Best-effort Intel GPU detection. Never raises; returns evidence dict."""
        info: Dict[str, Any] = {
            "platform": platform.system(),
            "intel_gpu_present": False,
            "evidence": [],
            "llama_bin": None,
        }
        for binary in ("llama-server", "llama-cli", "main", "llama.cpp"):
            found = shutil.which(binary)
            if found:
                info["llama_bin"] = found
                info["evidence"].append(f"llama binary: {found}")
                break
        # Windows: wmic / powershell CIM query for Intel graphics
        try:
            if platform.system() == "Windows":
                out = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "name"],
                    capture_output=True, text=True, timeout=5)
                if "Intel" in out.stdout:
                    info["intel_gpu_present"] = True
                    info["evidence"].append("wmic: Intel VideoController found")
            else:
                for probe in (["lspci"], ["xpu-smi", "discovery"]):
                    exe = shutil.which(probe[0])
                    if exe:
                        out = subprocess.run([exe] + probe[1:], capture_output=True,
                                             text=True, timeout=5)
                        if "Intel" in out.stdout:
                            info["intel_gpu_present"] = True
                            info["evidence"].append(f"{probe[0]}: Intel device found")
        except Exception as e:  # noqa: BLE001 - detection must never crash
            info["evidence"].append(f"detection probe failed: {e}")
        if os.environ.get("SYCL_DEVICE_FILTER") or os.environ.get("ONEAPI_DEVICE_SELECTOR"):
            info["evidence"].append("SYCL device filter env set")
        return info

    # -- memory model --------------------------------------------------
    def estimate_footprint_mb(self, n_layers_offloaded: Optional[int] = None,
                              total_layers: int = 32) -> Dict[str, float]:
        """INT4 weight footprint + KV-cache + runtime overhead vs budget."""
        cfg = self.config
        fp16_gb = MODEL_FP16_GB.get(cfg.model_id, 14.0)
        prof = QUANT_PROFILES.get(cfg.quantization_profile, QUANT_PROFILES["INT4_AUTOROUND"])
        frac = 1.0 if n_layers_offloaded is None else max(0.0, min(1.0, n_layers_offloaded / total_layers))
        weights_mb = fp16_gb * 1024 / prof["scale"] * frac
        # KV-cache ~ 2 * layers * ctx * d_model * 2 bytes; ~0.15 MB per ctx-token at 7B
        kv_mb = 0.15 * cfg.ctx_size / 16 * frac
        overhead_mb = 512.0  # runtime + framework reserve
        total = weights_mb + kv_mb + overhead_mb
        return {
            "weights_mb": round(weights_mb, 1),
            "kv_cache_mb": round(kv_mb, 1),
            "overhead_mb": overhead_mb,
            "total_mb": round(total, 1),
            "budget_mb": float(cfg.vram_budget_mb),
            "fits": total <= cfg.vram_budget_mb,
        }

    def recommend_offload(self, total_layers: int = 32) -> Dict[str, Any]:
        """Binary-search max offloadable layers that fit the budget."""
        cfg = self.config
        if cfg.n_gpu_layers is not None:
            est = self.estimate_footprint_mb(cfg.n_gpu_layers, total_layers)
            return {"n_gpu_layers": cfg.n_gpu_layers, "estimate": est, "mode": "manual"}
        lo, hi, best = 0, total_layers, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.estimate_footprint_mb(mid, total_layers)["fits"]:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        return {"n_gpu_layers": best if best else 0,
                "estimate": self.estimate_footprint_mb(best, total_layers),
                "mode": "auto-fit"}

    # -- command builder -----------------------------------------------
    def build_llama_command(self, total_layers: int = 32) -> List[str]:
        """Exact llama.cpp server command with explicit --n-gpu-layers."""
        cfg = self.config
        rec = self.recommend_offload(total_layers)
        n_layers = rec["n_gpu_layers"]
        binary = shutil.which("llama-server") or "llama-server"
        cmd = [binary, "-m", cfg.gguf_path,
               "--n-gpu-layers", str(n_layers),
               "-c", str(cfg.ctx_size),
               "--host", cfg.host, "--port", str(cfg.port)]
        if cfg.backend.lower() == "sycl":
            # SYCL build selects Intel GPU; keep env override documented
            cmd += ["--jinja"] if False else []
        cmd += cfg.extra_args
        self._last_recommendation = rec
        return cmd

    def describe(self, total_layers: int = 32) -> Dict[str, Any]:
        det = self.detect_accelerator()
        rec = self.recommend_offload(total_layers)
        return {
            "config": self.config.__dict__,
            "accelerator": det,
            "offload": rec,
            "llama_command": " ".join(self.build_llama_command(total_layers)),
            "sycl_env_hint": "set ONEAPI_DEVICE_SELECTOR=level_zero:gpu "
                             "(Windows) or SYCL_DEVICE_FILTER=level_zero:gpu",
        }
