"""Test SEAFDeclarativeOrchestrator pipeline execution and pressure throttling."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from seaf_agent.orchestrator import SEAFDeclarativeOrchestrator


def test_orchestrator_pipeline():
    orch = SEAFDeclarativeOrchestrator(vram_budget_mb=6144, quant_profile="INT4_AUTOROUND")

    orch.register_sub_agent("A1", memory_cost_mb=1024, task_func=lambda x: f"A1:{x}")
    orch.register_sub_agent("A2", memory_cost_mb=2048, task_func=lambda x: f"A2:{x.upper()}")

    pipeline = [
        {"agent": "A1", "input": "test_input"},
        {"agent": "A2", "input": "another_input"},
    ]

    res = orch.execute_pipeline(pipeline)
    assert res["status"] == "SUCCESS"
    assert len(res["step_results"]) == 2
    assert res["step_results"][0]["output"] == "A1:test_input"
    assert res["step_results"][1]["output"] == "A2:ANOTHER_INPUT"
    assert "policy" in res
    assert "system_pressure" in res
    print("[PASS] test_orchestrator_pipeline:", res["policy"])


def test_pressure_throttling_simulation():
    orch = SEAFDeclarativeOrchestrator(vram_budget_mb=6144)
    fake_pressure = {"ram_percent": 88.5, "under_pressure": True}
    policy = orch._compute_throttle_policy(total_declared_mb=1024, pressure=fake_pressure)
    assert policy["throttled"] is True
    assert policy["context_window"] == 2048
    assert policy["effective_budget_mb"] == 3072.0
    print("[PASS] test_pressure_throttling_simulation: Throttled mode verified")


if __name__ == "__main__":
    test_orchestrator_pipeline()
    test_pressure_throttling_simulation()
    print("All orchestrator tests passed successfully!")
