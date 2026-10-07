# SEAF: Sovereign-Edge Agentic Framework — Paper Draft (Phase 3)

> Target venues: IEEE EDGE, IEEE ICCA, ACM EuroSys. Sections below: Introduction,
> System Architecture, Evaluation. All quantitative claims cite
> `benchmark_tables.md` / `scratch_seaf_poc/benchmark_results.json`.

## 1. Introduction

Resource-constrained edge devices (consumer AI PCs with unified memory) cannot
sustain contemporary multi-agent stacks: a cognitive LLM supervisor co-resident
with worker agents thrashes VRAM, evicts KV-cache, and crashes the host, while
naive cloud offloading exposes personally identifiable information (PII) and
dynamic vector-DB memory defeats enterprise MLOps auditability. We present the
**Sovereign-Edge Agentic Framework (SEAF)**, a privacy-first multi-agent
framework organized around three principles: (i) **hardware-aware declarative
orchestration** replaces the LLM supervisor with compiled DSPy-style pipelines
whose per-agent memory costs are statically declared and throttled through
AutoRound INT4 profiles and explicit `llama.cpp` SYCL layer offloading;
(ii) **file-first scope-chained memory governance** splits memory into a
Git-versionable Markdown source-of-truth (`procedural_memory_master.md`) and a
SQLite working store (`frames.db`) with strict departmental isolation (child
scopes inherit parents; siblings are invisible); (iii) a **zero-exposure hybrid
gateway** inspects every payload locally with sub-millisecond regex screening
and pins PII-bearing or Tier-1/2 workloads to offline localhost runtimes
(Ollama/LM Studio), forwarding only sanitized Tier-3 prompts to an air-gapped
sovereign cloud (VergeIO VDC).

Our contributions are: (1) the SEAF design and its three invariants (PII never
leaves localhost; routing/memory scheduling stay under 5 ms with no LLM in the
decision path; scope-chain strictness with synchronous Markdown audit export);
(2) Phase-2 systems artifacts — localhost endpoint binding with missing-model
fallback, a `llama.cpp` SYCL adapter computing `--n-gpu-layers` from an INT4
memory model, and `psutil`-driven adaptive throttling (context halving and
sequential forcing above 80% RAM); (3) a reproducible benchmark answering three
research questions on efficiency (RQ1), privacy (RQ2), and governance (RQ3).

## 2. System Architecture

**2.1 PII-filtered hybrid edge gateway (`seaf_router.py`).** `SEAFHybridGateway`
screens prompts against compiled SSN, credit-card, email, phone, and sensitive
keyword patterns. `route_request` is pure regex (mean 0.0232 ms, P99 0.052 ms
over 1,000 requests) and returns a `LOCAL_OFFLINE_EDGE` vs
`VERGEIO_SOVEREIGN_CLOUD` decision. Phase 2 adds `discover_local_models`,
`check_local_runtime`, `generate_local` (Ollama `/api/generate` →
Ollama OpenAI-compat → LM Studio, stdlib `urllib` only), and
`execute_routed_request`, which executes PII payloads strictly on localhost
(auto-selecting e.g. `phi4-mini`) and re-inspects sanitized Tier-3 prompts,
raising `SecurityError` if regex-grade PII survives redaction. Verified live:
a PII Tier-3 prompt is forced local and served by `ollama-generate`, while
`phi4-mini` answers a canary (`EDGE_OK`) in ~18.8 s cold on an RTX 3050 —
confirming the local path is real and motivating Tier-3 offload for heavy
reasoning.

**2.2 Hardware-aware declarative orchestrator (`seaf_orchestrator.py`,
`seaf_sycl_adapter.py`).** Sub-agents register with declared `memory_cost_mb`;
`execute_pipeline` applies the INT4 4× reduction, samples host pressure once
per pipeline via `psutil` (RAM/CPU/RSS plus cached `xpu-smi`/`nvidia-smi`
probe, 60 s TTL to keep scheduling O(1)), and, above 80% RAM, halves the
context window (4096→2048), caps allocations at 50% of budget, and enforces
sequential execution with VRAM release between agents. The SYCL adapter
(`SYCLHardwareAdapter`) models INT4 weights + KV-cache + 512 MB overhead
against the unified-memory budget, binary-searches the maximum fittable
`--n-gpu-layers`, and emits the exact `llama-server` command with SYCL device
hints (`ONEAPI_DEVICE_SELECTOR=level_zero:gpu`), degrading gracefully to a
runnable command string when no accelerator is present.

**2.3 File-first scope-chained memory (`seaf_memory.py`).**
`SEAFMemoryEngine` stores frames `(timestamp, scope, key, value, sensitivity,
author)` in SQLite indexed by scope. `query_memory` resolves the inheritance
chain (`finance.payroll` → `finance` → `global`) while sibling subtrees
(e.g. `marketing.campaign`, `hr.recruiting`) are excluded by construction.
Every `store_frame` synchronously regenerates the per-scope Markdown audit
trail, making procedural memory `git diff`-able and human-reviewable — a
deliberate contrast to black-box vector stores.

## 3. Evaluation

Testbed: Windows, Python 3.13, RTX 3050 6 GB Laptop GPU, 16 GB RAM at
80–86% occupancy (throttle engaged, hence conservative numbers), Ollama
Q4_K_M models, `available_vram_mb=8192`, INT4 AutoRound.

**RQ1 — Efficiency.** The 3-agent INT4 pipeline peaks at 4,096 MB (50% of
budget) with mean duration 3.3842 ms over 100 runs and 46.29 MB process RSS
(Table RQ1). First-pipeline cold cost (~100 ms, uncached GPU discovery) is
amortized by caching; steady-state scheduling adds <1 ms. Live local inference
(`phi4-mini`, `EDGE_OK` in ~18.8 s) bounds the edge-capable workload and
justifies sovereign offload for Tier-3 tasks.

**RQ2 — Privacy.** Over 1,000 mixed Tier-1–3 requests, routing accuracy and
PII containment are both 100%: every PII-bearing prompt remained on
`LOCAL_OFFLINE_EDGE` and every Tier-3 cloud payload passed sanitization plus
re-inspection (Table RQ2). Mean routing latency 0.0232 ms (P99 0.052 ms,
25.47 ms total) satisfies the <5 ms core-operation invariant by keeping LLMs
out of the decision path.

**RQ3 — Governance.** Across 500 writes, mean write latency is 19.0 ms/frame
(dominated by synchronous full Markdown export — the auditable-sync cost);
scope-chain reads average 2.91 ms over 100 queries (Table RQ3). The isolation
audit passes with zero cross-scope leakage, and the Markdown master verifies
`EXISTS & VALID` on every run, confirming Git-syncable MLOps compliance.

**Threats and limits.** Write latency scales with full-file Markdown
regeneration (incremental export is future work); the 8,192 MB budget exceeds
this testbed's 6 GB discrete VRAM, so reported headroom assumes unified-memory
AI PCs; cloud-side Tier-3 latency is out of scope (air-gapped VDC stub).
Nevertheless, SEAF demonstrates that declarative orchestration, scope-chained
file-first memory, and a zero-exposure gateway jointly deliver edge-feasible,
auditable multi-agent execution under real memory pressure.
