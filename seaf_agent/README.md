# SEAF-SDA: Sovereign-Edge Agentic Framework ⚡

**Unified Autonomous Edge Agent for AI PCs & Laptops**  
Target Hardware: Intel Core i5 / AMD Ryzen CPU, NVIDIA GeForce RTX 3050 (6GB) / RTX 2050 (4GB) VRAM, 16GB System RAM.

SEAF-SDA combines the theoretical principles of the **Sovereign-Edge Agentic Framework (SEAF)** with the autonomous **Software Development Agent (SDA)** Reasoner-Critic execution loop into a single production-ready package.

---

## 🏛️ The 4 Core Architectural Pillars

1. **Zero-Exposure PII Edge Gateway (`gateway.py`)**:
   - Sub-millisecond (< 0.05 ms) regex & heuristic screening for PII (SSN, credit cards, emails, phones, and sensitive keywords).
   - **Invariant:** Payloads with PII or Tier 1–2 complexity are **strictly pinned to local offline execution** on `localhost` (Ollama).
   - If pre-flight re-inspection detects residual PII after redaction, a `SecurityError` is raised and the request immediately falls back to local edge execution.

2. **Hardware-Aware Declarative Orchestrator (`orchestrator.py` & `telemetry.py`)**:
   - Replaces heavy LLM supervisor loops with compiled DSPy-style declarative execution pipelines.
   - Statically applies an **INT4 AutoRound 4× memory scaling factor** to model per-agent footprints against the unified VRAM budget.
   - **Adaptive Throttling:** Live host memory is sampled via `psutil` **strictly once per pipeline invocation** (with a 60s cached GPU probe). If host RAM exceeds **80%**, the context window is automatically halved from `4096` to `2048` tokens, allocation is capped at 50%, and sequential sub-agent execution with explicit `gc.collect()` is enforced.

3. **File-First Scoped Memory Governance (`memory.py`)**:
   - Relational memory store backed by SQLite (`frames.db`) configured with Write-Ahead Logging (`PRAGMA journal_mode=WAL;`) to guarantee zero transaction locks during concurrent reads and writes.
   - **Scope-Chain Inheritance:** Querying `sda.finance.payroll` inherits rules from `sda.finance` and `sda.global`. Sibling subtrees (e.g., `sda.marketing`) are 100% invisible (0% data leakage).
   - Synchronously exports human-readable, Git-versionable Markdown audit logs to `memory_md/procedural_memory_master.md` on every write.

4. **Hardware & Sovereign Connectors (`sycl_adapter.py` & `sovereign_connector.py`)**:
   - Dynamic `--n-gpu-layers` calculation for `llama-server` and edge hardware.
   - Circuit-breaker REST connector for air-gapped Sovereign Virtual Data Center (VDC) clusters with seamless fail-safe fallback to local `phi4-mini:latest`.

---

## 🛡️ Mitigation of Critical Pitfalls

| Pitfall | Problem | SEAF-SDA Solution |
|---|---|---|
| **1. Over-sampling `psutil`** | High-frequency polling in token loops adds CPU jitter | Sampled **strictly once per pipeline run**, utilizing a 60-second TTL cache for GPU probes |
| **2. SQLite Lock Contention** | Concurrent multi-agent reads fail during Markdown writes | Initialized with `PRAGMA journal_mode=WAL;` and explicit connection cleanup (`try...finally`) |
| **3. Scope Hierarchy Leakage** | Malformed scope query defaulting to root exposes data | Strict dot-notation validator defaults malformed scopes to narrowest isolated scope, never global |
| **4. Cloud Failure Handling** | Network timeouts or `SecurityError` crashes UI | Seamless circuit-breaker fail-safe to local `phi4-mini` without bubbling unhandled errors |

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment
Ensure you have Python 3.11–3.13 and Ollama running locally with the required models:
```powershell
ollama pull qwen2.5-coder:7b
ollama pull phi4-mini:latest
ollama pull nomic-embed-text:latest
```

### 2. Run the Desktop GUI Application
Launch the dark-mode CustomTkinter interface with live telemetry gauges, PII screening badges, and scoped memory viewer:
```powershell
# From project directory
& ".\Agentic AI\micro_sda\.venv\Scripts\python.exe" ".\seaf_agent\app.py"
```

### 3. Run the End-to-End Verification Suite (CLI)
Test all 3 research scenarios (HR payroll PII containment, Declarative Orchestration under pressure, and Scoped Memory isolation):
```powershell
& ".\Agentic AI\micro_sda\.venv\Scripts\python.exe" ".\seaf_agent\run_seaf_demo.py"
```

### 4. Run the Autonomous Coding Agent Loop (CLI)
Execute the complete Reasoner-Critic generation, sandbox execution, and auto-patching loop:
```powershell
& ".\Agentic AI\micro_sda\.venv\Scripts\python.exe" ".\seaf_agent\test_coding_run.py"
```

---

## 📂 Package Layout

```
seaf_agent/
├── __init__.py               # Package initialization
├── config.py                 # Central hardware & endpoint configuration
├── telemetry.py              # Single-sample psutil & cached GPU telemetry
├── gateway.py                # Zero-exposure PII screening & hybrid router
├── orchestrator.py           # Declarative hardware-aware orchestrator
├── memory.py                 # Scoped SQLite memory (WAL) & Markdown sync
├── rag.py                    # ChromaDB semantic retrieval & experience store
├── sycl_adapter.py           # SYCL/CUDA layer offload calculator
├── sovereign_connector.py    # Circuit-breaker sovereign cloud connector
├── agent.py                  # LangGraph cognitive router & AST filters
├── critic.py                 # Autonomous Reasoner-Critic state machine
├── app.py                    # CustomTkinter dark-mode desktop GUI
├── run_seaf_demo.py          # End-to-end SEAF research verification suite
├── test_coding_run.py        # CLI coding loop integration test
├── tools/
│   ├── filters.py            # Caveman text compressor & Ponytail AST review
│   ├── traceback_parser.py   # Traceback classifier & auto-pip guidance
│   └── sandbox.py            # Subprocess execution sandbox with timeout
└── data/
    ├── frames.db             # Relational memory frames (WAL mode)
    ├── memory_md/            # Git-trackable procedural memory markdown files
    ├── chroma_db/            # Vector embeddings store
    └── workspace/            # Sandboxed code execution staging
```
