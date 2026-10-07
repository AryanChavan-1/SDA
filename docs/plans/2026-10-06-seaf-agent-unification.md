# SEAF-Agent Unified Implementation Plan

> **For Agent:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Create a unified edge AI framework package (`seaf_agent`) in `D:/New folder (2)/seaf_agent` that combines the theoretical SEAF framework (zero-exposure PII gateway, hardware-aware declarative orchestration with live psutil/GPU telemetry, and file-first scope-chained memory) with the autonomous Micro-SDA Reasoner-Critic coding loop and an upgraded CustomTkinter desktop GUI.

**Architecture:** A modular Python package containing:
1. `gateway.py` (PII regex inspection + hybrid local/cloud routing)
2. `telemetry.py` (Host RAM/CPU and NVIDIA GPU VRAM live monitor)
3. `orchestrator.py` (Declarative pipelines, INT4 profiles, dynamic RAM throttling)
4. `memory.py` (Dual-tier SQLite scope-chained frames + Git-versionable Markdown sync + ChromaDB RAG integration)
5. `tools/` (MCP and local filters: Caveman compressor, Ponytail AST reviewer, Traceback parser, Docker/subprocess executor)
6. `agent.py` & `critic.py` (LangGraph Reasoner-Critic state machine with auto-pip install and experience logging)
7. `app.py` (Upgraded CustomTkinter desktop GUI with live hardware telemetry, PII badges, and memory scope visualizer)
8. `run_demo.py` & `test_mnist_run.py` (CLI demo and end-to-end coding agent verification)

**Tech Stack:** Python 3.13, CustomTkinter, LangGraph, LangChain, Ollama (`qwen2.5-coder:7b`, `phi4-mini:latest`, `nomic-embed-text`), SQLite3, ChromaDB, psutil, FastMCP / stdlib subprocess.

---

### Task 1: Initialize Package Structure and Dependencies
**Files:**
- Create: `D:/New folder (2)/seaf_agent/__init__.py`
- Create: `D:/New folder (2)/seaf_agent/config.py`
- Create: `D:/New folder (2)/seaf_agent/requirements.txt`
- Create: `D:/New folder (2)/seaf_agent/.env.example`

**Step 1:** Define package configurations in `config.py` targeting RTX 3050 (6GB VRAM), 16GB RAM, Ollama endpoints (`http://localhost:11434`), default models (`qwen2.5-coder:7b`, `phi4-mini:latest`, `nomic-embed-text`), and memory storage paths.
**Step 2:** Provide clean `requirements.txt` based on existing working environment.
**Step 3:** Verify imports and setup.

---

### Task 2: Implement Telemetry & Hardware Adaptation
**Files:**
- Create: `D:/New folder (2)/seaf_agent/telemetry.py`
- Create: `D:/New folder (2)/seaf_agent/hardware_adapter.py`
- Test: `D:/New folder (2)/seaf_agent/tests/test_telemetry.py`

**Step 1:** Create `telemetry.py` using `psutil` and cached `nvidia-smi` queries (with 30s TTL) to reliably report RAM %, process RSS, and GPU VRAM usage.
**Step 2:** Port and enhance `SYCLHardwareAdapter` to `hardware_adapter.py` to support INT4 AutoRound profiles and VRAM allocation checks for both NVIDIA CUDA and Intel SYCL.
**Step 3:** Run `test_telemetry.py` to verify real-time metrics reading from the RTX 3050.

---

### Task 3: Implement Zero-Exposure PII Hybrid Gateway
**Files:**
- Create: `D:/New folder (2)/seaf_agent/gateway.py`
- Test: `D:/New folder (2)/seaf_agent/tests/test_gateway.py`

**Step 1:** Port `SEAFHybridGateway` with regex screening (SSN, credit card, email, phone, sensitive financial/health keywords) and <0.05 ms inspection latency.
**Step 2:** Implement routing logic: all PII-bearing or Tier 1/2 prompts are strictly pinned to localhost (`qwen2.5-coder:7b` or `phi4-mini`); sanitized Tier 3 tasks can route to external endpoints with invariant checks raising `SecurityError` if any PII pattern survives.
**Step 3:** Add local Ollama execution fallback directly using stdlib/httpx.
**Step 4:** Run `test_gateway.py` to assert 100% PII containment.

---

### Task 4: Implement File-First Scope-Chained Memory Engine
**Files:**
- Create: `D:/New folder (2)/seaf_agent/memory.py`
- Create: `D:/New folder (2)/seaf_agent/rag.py`
- Test: `D:/New folder (2)/seaf_agent/tests/test_memory.py`

**Step 1:** Implement `SEAFMemoryEngine` with SQLite table `memory_frames` supporting scope inheritance (`scope.sub` inherits `scope` and `global`, sibling isolation).
**Step 2:** Implement synchronous Markdown export to `memory_md/procedural_memory_master.md` with structured Git-versionable audit tables.
**Step 3:** Integrate ChromaDB vector store (`rag.py`) for semantic chunking and past-experience lookup (`store_experience` / `query_experience`).
**Step 4:** Run `test_memory.py` asserting cross-scope isolation and markdown regeneration.

---

### Task 5: Implement Declarative Hardware-Aware Orchestrator
**Files:**
- Create: `D:/New folder (2)/seaf_agent/orchestrator.py`
- Test: `D:/New folder (2)/seaf_agent/tests/test_orchestrator.py`

**Step 1:** Port `SEAFDeclarativeOrchestrator` incorporating live telemetry from `telemetry.py`.
**Step 2:** Implement adaptive pressure throttling: when host RAM > 80% (which is common on 16GB laptops), dynamically halve context budget (4096 -> 2048), cap allocations at 50% budget, and force sequential execution with memory cleanup.
**Step 3:** Support sub-agent registration with explicit memory declarations.
**Step 4:** Verify pipeline execution under live laptop memory pressure.

---

### Task 6: Implement Tooling & Filters (Caveman, Ponytail, Traceback, Sandbox)
**Files:**
- Create: `D:/New folder (2)/seaf_agent/tools/filters.py` (Caveman text compression + Ponytail AST review)
- Create: `D:/New folder (2)/seaf_agent/tools/traceback_parser.py` (Traceback parsing + fix recommendations)
- Create: `D:/New folder (2)/seaf_agent/tools/sandbox.py` (Subprocess execution with timeout, output capping, Docker fallback)
- Test: `D:/New folder (2)/seaf_agent/tests/test_tools.py`

**Step 1:** Implement `filters.py` for stripping filler words and checking AST anti-patterns.
**Step 2:** Implement `traceback_parser.py` mapping Python exceptions to error categories.
**Step 3:** Implement `sandbox.py` executing code safely with standard timeout and stdout/stderr capture.
**Step 4:** Run tests to verify AST catching, error parsing, and code execution.

---

### Task 7: Implement LangGraph Autonomous Reasoner-Critic Agent
**Files:**
- Create: `D:/New folder (2)/seaf_agent/agent.py`
- Create: `D:/New folder (2)/seaf_agent/critic.py`
- Test: `D:/New folder (2)/seaf_agent/tests/test_critic_loop.py`

**Step 1:** Build the integrated LangGraph state graph combining Gateway PII check $\rightarrow$ RAG retrieval $\rightarrow$ Caveman compression $\rightarrow$ Ponytail check $\rightarrow$ Reasoner generation $\rightarrow$ HitL check $\rightarrow$ Sandbox execution $\rightarrow$ Traceback parsing & auto-pip install $\rightarrow$ Critic evaluation $\rightarrow$ Patcher loop $\rightarrow$ Memory frame & experience logging.
**Step 2:** Connect `critic.py` with `memory.py` so successful resolutions are logged to both SQLite/Markdown procedural memory and ChromaDB experience store.
**Step 3:** Run `test_critic_loop.py` with a synthetic error to verify auto-patching and recovery.

---

### Task 8: Upgrade CustomTkinter Desktop GUI
**Files:**
- Create: `D:/New folder (2)/seaf_agent/app.py`

**Step 1:** Build modern dark-mode GUI with CustomTkinter.
**Step 2:** Add Top/Sidebar Hardware Telemetry dashboard (Live RAM % gauge with color change when >80% throttled, Live GPU VRAM % gauge from RTX 3050).
**Step 3:** Add Live Gateway PII status pill/badge (Green "Safe / Local Edge", Red/Orange "PII Detected - Edge Locked").
**Step 4:** Implement multi-step progress bar (Ingest $\rightarrow$ Compress $\rightarrow$ Generate $\rightarrow$ Review $\rightarrow$ Execute $\rightarrow$ Critic $\rightarrow$ Memory Sync).
**Step 5:** Add HitL confirmation modal, code editor view, execution logs, and Memory Scopes browser tab.

---

### Task 9: Implement CLI Demos & End-to-End Verification
**Files:**
- Create: `D:/New folder (2)/seaf_agent/run_demo.py`
- Create: `D:/New folder (2)/seaf_agent/test_coding_run.py`
- Create: `D:/New folder (2)/seaf_agent/README.md`

**Step 1:** Create `run_demo.py` replicating the 3 SEAF scenarios (HR Payroll PII audit, Hardware Orchestration under RAM pressure, Scope-Chained Memory isolation) with real agent execution.
**Step 2:** Create `test_coding_run.py` to test an end-to-end coding problem using local `qwen2.5-coder:7b`.
**Step 3:** Write comprehensive `README.md` detailing architecture, hardware specs, GUI operation, and CLI usage.
