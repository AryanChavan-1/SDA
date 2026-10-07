"""Test telemetry and hardware adapter."""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from seaf_agent.telemetry import telemetry
from seaf_agent.hardware_adapter import EdgeHardwareAdapter, HardwareAdapterConfig


def test_telemetry_sample():
    sample = telemetry.sample()
    assert "ram_total_mb" in sample
    assert "ram_percent" in sample
    assert "cpu_percent" in sample
    assert "under_pressure" in sample
    assert sample["ram_total_mb"] > 1000
    print("[PASS] test_telemetry_sample:", sample)


def test_hardware_adapter_estimate():
    adapter = EdgeHardwareAdapter(HardwareAdapterConfig(model_id="qwen2.5-coder:7b", vram_budget_mb=6144))
    est = adapter.estimate_vram()
    assert est["weights_mb"] > 0
    assert "fits" in est
    layers = adapter.compute_max_gpu_layers()
    assert layers > 0
    print("[PASS] test_hardware_adapter_estimate:", est, "layers:", layers)


if __name__ == "__main__":
    test_telemetry_sample()
    test_hardware_adapter_estimate()
    print("All telemetry tests passed successfully!")
