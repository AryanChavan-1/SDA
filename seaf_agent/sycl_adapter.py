"""
seaf_agent.sycl_adapter — Hardware SYCL/CUDA Offloading Adapter (SEAF Pillar 4).
Dynamically calculates optimal --n-gpu-layers offloading for llama-server or edge runtimes
on Intel/NVIDIA unified or discrete edge hardware.
"""

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

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
    "phi4-mini:latest": 32,
    "phi4-mini": 32,
    "phi-3-mini": 32,
    "mistral:7b": 32,
    "mistral:7b-instruct-q4_K_M": 32,
    "qwen2.5-coder:7b": 28,
    "qwen2.5:7b": 28,
}


@dataclass
class SYCLAdapterConfig:
    model_id: str = "qwen2.5-coder:7b"
    gguf_path: str = "./models/qwen2.5-coder-7b-Q4_K_M.gguf"
    vram_budget_mb: int = VRAM_BUDGET_MB
    quantization_profile: str = QUANTIZATION_PROFILE
    ctx_size: int = BASE_CONTEXT_WINDOW
    n_gpu_layers: Optional[int] = None  # None = calculate dynamically
    backend: str = "sycl"               # sycl | cuda | cpu
    host: str = "127.0.0.1"
    port: int = 8080
    extra_args: List[str] = field(default_factory=list)


class SYCLHardwareAdapter:
    def __init__(self, config: Optional[SYCLAdapterConfig] = None):
        self.config = config or SYCLAdapterConfig()

    def estimate_memory_mb(self) -> Dict[str, Any]:
        """Calculates memory requirements: weights + KV-cache + overhead."""
        profile = QUANT_PROFILES.get(self.config.quantization_profile, QUANT_PROFILES["INT4_AUTOROUND"])
        scale = profile["scale"]

        fp16_gb = MODEL_FP16_GB.get(self.config.model_id, 14.0)
        weights_mb = (fp16_gb * 1024.0) / scale
        kv_cache_mb = (self.config.ctx_size / 2048.0) * (fp16_gb * 18.0)
        overhead_mb = 512.0

        total_mb = weights_mb + kv_cache_mb + overhead_mb
        fits = total_mb <= self.config.vram_budget_mb
        headroom_mb = self.config.vram_budget_mb - total_mb

        return {
            "model_id": self.config.model_id,
            "quantization": profile["gguf"],
            "scale": scale,
            "weights_mb": round(weights_mb, 1),
            "kv_cache_mb": round(kv_cache_mb, 1),
            "overhead_mb": round(overhead_mb, 1),
            "total_mb": round(total_mb, 1),
            "vram_budget_mb": self.config.vram_budget_mb,
            "fits": fits,
            "headroom_mb": round(headroom_mb, 1),
        }

    def compute_n_gpu_layers(self) -> int:
        """Determines the optimal number of layers to offload to GPU."""
        total_layers = MODEL_TOTAL_LAYERS.get(self.config.model_id, 32)
        if self.config.n_gpu_layers is not None:
            return min(self.config.n_gpu_layers, total_layers)

        est = self.estimate_memory_mb()
        if est["fits"]:
            return total_layers

        # Proportional layer budgeting
        weights_budget = max(256.0, self.config.vram_budget_mb - est["kv_cache_mb"] - est["overhead_mb"])
        frac = max(0.0, min(1.0, weights_budget / est["weights_mb"]))
        return max(0, int(total_layers * frac))

    def build_command(self) -> List[str]:
        """Builds llama-server invocation with SYCL/CUDA layer offloading."""
        n_layers = self.compute_n_gpu_layers()
        binary = "llama-server"
        cmd = [
            binary,
            "-m", self.config.gguf_path,
            "-c", str(self.config.ctx_size),
            "--n-gpu-layers", str(n_layers),
            "--host", self.config.host,
            "--port", str(self.config.port),
        ]
        cmd.extend(self.config.extra_args)
        return cmd
