# SEAF Benchmark Results — Formal Tables (IEEE/ACM)

Source: `run_seaf_benchmark.py` output (`scratch_seaf_poc/benchmark_results.json`),
reproduced 2026-09-25 on Windows / RTX 3050 6GB Laptop / 16 GB RAM / Ollama
(`phi4-mini:latest` 3.8B Q4_K_M, `qwen2.5-coder:7b` Q4_K_M). Host RAM was
80–86% utilized during the run, so the orchestrator's >80% pressure throttle
(context-window halving, sequential execution) was active — reported values
are therefore conservative (under-pressure) figures. Re-runs on the same box
stay in the same band (gateway avg 0.023–0.025 ms, P99 0.05–0.13 ms; query
2.9–5.4 ms; pipeline 2.9–3.4 ms); exact JSON snapshots vary with host load.

## RQ1 (Efficiency): VRAM footprint and throughput under declarative orchestration

| Metric | Value | Method |
|---|---|---|
| Peak VRAM allocation (INT4 AutoRound) | 4,096 MB / 8,192 MB budget (50.0%) | 3-agent pipeline (1024+2048+4096 MB declared), sequential release, max observed |
| Avg pipeline duration (3 agents) | 3.3842 ms | Mean over 100 runs, incl. live `psutil` pressure sample (cached GPU probe) |
| Process RSS footprint | 46.29 MB | `psutil.Process.rss` after benchmark |
| Cold-start scheduling overhead | ~100 ms (first pipeline only) | Uncached `shutil.which` GPU probe; amortized via 60 s cache |
| Local Tier-1 inference feasibility | `phi4-mini` replies `EDGE_OK` in ~18.8 s cold | Live Ollama `/api/generate` on RTX 3050 (evidence local edge path is functional, Tier-3 offload motivated) |

## RQ2 (Privacy): Gateway routing latency and PII containment

| Metric | Value | Method |
|---|---|---|
| Gateway avg inspection latency | 0.0232 ms | Mean over 1,000 routed requests (5-prompt mix, Tiers 1–3) |
| Gateway P99 inspection latency | 0.0520 ms | 99th percentile, same 1,000 runs |
| Total wall time (1,000 requests) | 25.47 ms | `perf_counter` around loop |
| PII containment & routing accuracy | 100.00% (1,000/1,000) | Expected-PII labels vs `has_pii`; PII or Tier ≤2 always `LOCAL_OFFLINE_EDGE`; Tier-3 sanitized payload re-inspected before cloud stub (`SecurityError` on residual regex PII) |
| Core-operation invariant | Routing decision < 5 ms (no LLM in path) | `route_request` uses regex only; LLM calls isolated in `generate_local` |

## RQ3 (Governance): Scope-chain isolation integrity and MLOps Git sync

| Metric | Value | Method |
|---|---|---|
| Avg memory write latency (SQLite + Markdown sync) | 19.0011 ms/frame | Mean over 500 `store_frame` calls across 5 scopes (each triggers full Markdown export) |
| Avg scope-chain query latency | 2.9065 ms | Mean over 100 `query_memory("finance.payroll")` (inherits `finance` + `global`) |
| Cross-scope isolation | PASSED (0% leakage) | `finance.payroll` frames invisible from `hr.recruiting` / `marketing.campaign`; child inherits parents only |
| MLOps Markdown source-of-truth | VERIFIED | `procedural_memory_master.md` regenerated on every write; `EXISTS & VALID` in demo suite |
| Audit export format | Per-scope Markdown tables (Key/Value/Sensitivity/Agent/Timestamp) | Git-versionable flat files |

## Test environment (for reproducibility footnote)

| Component | Detail |
|---|---|
| OS / Python | Windows, Python 3.13.3, `psutil` 7.1.0 |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop (535 MiB used at idle probe) |
| RAM | 16 GB total, ~80–86% utilized (pressure throttle engaged) |
| Local runtime | Ollama `phi4-mini:latest` (3.8B, Q4_K_M), `qwen2.5-coder:7b` (Q4_K_M) |
| Config | `available_vram_mb=8192`, `INT4_AUTOROUND`, `base_context_window=4096` (halved to 2048 under pressure) |
