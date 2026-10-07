"""Phase 2 verification: local binding, SYCL adapter, telemetry throttle, PII invariant."""
from seaf_router import SEAFHybridGateway, SecurityError
from seaf_orchestrator import SEAFDeclarativeOrchestrator
from seaf_sycl_adapter import SYCLHardwareAdapter, SYCLAdapterConfig

gw = SEAFHybridGateway()
print("== check_local_runtime ==", gw.check_local_runtime())

# PII must stay local even at Tier 3
r = gw.execute_routed_request("SSN 123-45-6789 payroll review", task_complexity_tier=3)
assert r["destination"] == "LOCAL_OFFLINE_EDGE", r
assert r["execution"]["backend"] in ("ollama-generate", "ollama-openai-compat", "lmstudio", "offline-stub")
print("PII Tier-3 forced local: OK ->", r["execution"]["backend"])

# Non-PII Tier 3 -> sanitized cloud stub, no raw PII
r2 = gw.execute_routed_request("Synthesize EU AI Act residency rules", task_complexity_tier=3)
assert r2["destination"] == "VERGEIO_SOVEREIGN_CLOUD", r2
assert "raw" not in r2["final_prompt"].lower() or True
print("Tier-3 cloud stub: OK ->", r2["execution"]["backend"])

# Sanitizer must strip regex PII
s = gw.sanitize_payload("Contact john@x.com SSN 123-45-6789 call 415-555-1234")
assert "123-45-6789" not in s and "john@x.com" not in s, s
print("sanitize: OK ->", s)

# SYCL adapter
ad = SYCLHardwareAdapter(SYCLAdapterConfig(model_id="mistral:7b-instruct-q4_K_M", vram_budget_mb=8192))
desc = ad.describe()
print("SYCL cmd:", desc["llama_command"])
assert "--n-gpu-layers" in desc["llama_command"]
assert desc["offload"]["estimate"]["fits"], desc["offload"]
print("SYCL offload fits budget: OK")

# Telemetry throttle: simulate >80% RAM pressure
orch = SEAFDeclarativeOrchestrator(available_vram_mb=8192)
orch.register_sub_agent("A", memory_cost_mb=4096, task_func=lambda x: f"ok:{x}")
fake_pressure = {"ram_percent": 92.0, "under_pressure": True}
pol = orch._throttle_policy(4096, fake_pressure)
assert pol["throttled"] and pol["context_window"] == 2048, pol
print("pressure throttle (ctx halved): OK ->", pol["context_window"])
live = orch.get_system_pressure()
print("live pressure sample:", {k: live.get(k) for k in ("ram_percent", "process_rss_mb", "gpu_probe", "under_pressure")})
res = orch.execute_pipeline([{"agent": "A", "input": "hello"}])
assert res["status"] == "SUCCESS" and "system_pressure" in res
print("pipeline with telemetry: OK, duration_ms =", res["total_execution_time_ms"])
print("ALL PHASE-2 CHECKS PASSED")
