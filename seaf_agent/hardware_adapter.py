"""
seaf_agent.hardware_adapter — Hardware layer offload calculator and runtime adapter.
Supports:
- INT4 AutoRound profiles and GGUF quantization scales
- VRAM fit estimation (weights + KV-cache + runtime overhead)
- Intel SYCL and NVIDIA CUDA layer offload calculation
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import platform
import shutil

from seaf_agent.config import VRAM_BUDGET_MB, QUANTIZATION_PROFILE, BASE_CONTEXT_WINDOW

QUANT_PROFILES = {
    "INT4_AUTOROUND": {"scale": 4.0, "gguf": "Q4_K_M", "bpp": 0.5},
    "Q4_K_M":         {"scale": 4.0, "gguf": "Q4_K_M", "bpp": 0.5},
    "Q5_K_M":         {"scale": 3.2, "gguf": "Q5_K_M", "bpp": 0.625},
    "Q8_0":           {"scale": 2.0, "gguf": "Q8_0", "bpp": 1.0},
    "F16":            {"scale": 1.0, "gguf": "F16", "bpp": 2.0},
}

MODEL_FP16_GB = {
    "llama3.2:1b": 2.0,
    "llama3.2:3b": 6.0,
    "phi4-mini:latest": 7.6,
    "phi4-mini": 7.6,
    "phi-3-mini": 7.6,
    "mistral:7b": 14.0,
    "mistral:7b-instruct-q4_K_M": 14.0,
    "qwen2.5-coder:7b": 14.0,
    "qwen2.5:7b": 14.0,
}

MODEL_TOTAL_LAYERS = {
    "llama3.2:1b": 16,
    "llama3.2:3b": 28,
    "phi4-mini": 32,
    "phi4-mini:latest": 32,
    "phi-3-mini": 32,
    "mistral:7b": 32,
    "mistral:7b-instruct-q4_K_M": 32,
    "qwen2.5-coder:7b": 28,
    "qwen2.5:7b": 28,
}


@dataclass
class HardwareAdapterConfig:
    model_id: str = "qwen2.5-coder:7b"
    vram_budget_mb: int = VRAM_BUDGET_MB
    quantization_profile: str = QUANTIZATION_PROFILE
    ctx_size: int = BASE_CONTEXT_WINDOW
    n_gpu_layers: Optional[int] = None
    backend: str = "auto"  # auto | cuda | sycl | cpu


class EdgeHardwareAdapter:
    def __init__(self, config: Optional[HardwareAdapterConfig] = None):
        self.config = config or HardwareAdapterConfig()

    def estimate_vram(self, model_id: Optional[str] = None, ctx_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculates memory needed:
        weights_mb = (fp16_gb * 1024) / quant_scale
        kv_cache_mb ~= (ctx_size / 2048) * (model_fp16_gb * 18)
        runtime_overhead_mb ~= 512 MB
        """
        mid = model_id or self.config.model_id
        ctx = ctx_size or self.config.ctx_size
        profile = QUANT_PROFILES.get(self.config.quantization_profile, QUANT_PROFILES["INT4_AUTOROUND"])
        scale = profile["scale"]

        fp16_gb = MODEL_FP16_GB.get(mid, 7.0)
        weights_mb = (fp16_gb * 1024.0) / scale
        kv_mb = (ctx / 2048.0) * (fp16_gb * 18.0)
        overhead_mb = 512.0

        total_mb = weights_mb + kv_mb + overhead_mb
        fits = total_mb <= self.config.vram_budget_mb
        headroom_mb = self.config.vram_budget_mb - total_mb

        return {
            "model_id": mid,
            "weights_mb": round(weights_mb, 1),
            "kv_cache_mb": round(kv_mb, 1),
            "overhead_mb": round(overhead_mb, 1),
            "total_estimated_mb": round(total_mb, 1),
            "vram_budget_mb": self.config.vram_budget_mb,
            "fits": fits,
            "headroom_mb": round(headroom_mb, 1),
            "utilization_pct": round((total_mb / self.config.vram_budget_mb) * 100.0, 1),
        }

    def compute_max_gpu_layers(self, model_id: Optional[str] = None) -> int:
        """Determines how many layers can safely be placed in VRAM."""
        mid = model_id or self.config.model_id
        total_layers = MODEL_TOTAL_LAYERS.get(mid, 32)
        est = self.estimate_vram(mid)

        if est["fits"]:
            return total_layers

        # Binary search / linear proportion if does not fully fit
        weights_budget = max(256.0, self.config.vram_budget_mb - est["kv_cache_mb"] - est["overhead_mb"])
        fraction = max(0.0, min(1.0, weights_budget / est["weights_mb"]))
        return max(0, int(total_layers * fraction))
