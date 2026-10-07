"""
seaf_agent.critic — Autonomous Reasoner-Critic Agentic Loop.
Flow:
PII/Telemetry Check -> RAG Retrieval -> Caveman Compress -> Reasoner -> Ponytail Review -> HitL Pause -> Sandbox Execute -> Critic Evaluate -> Patcher -> Retry/Pass.
On success: Synchronously syncs memory to SQLite (frames.db), Git Markdown, and ChromaDB.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from seaf_agent.agent import (
    AgentState,
    _extract_code_block,
    node_caveman_compress,
    node_pii_and_telemetry,
    node_ponytail_review,
    node_rag_retrieval,
    node_reasoner_generate,
)
from seaf_agent.config import (
    MAX_RETRIES,
    LOCAL_CODING_SLM,
    OLLAMA_BASE_URL,
    EXEC_TIMEOUT_SEC,
)
from seaf_agent.memory import memory_engine
from seaf_agent.rag import rag_engine
from seaf_agent.tools.sandbox import execute_python_code
from seaf_agent.tools.traceback_parser import parse_traceback

log = logging.getLogger("seaf.critic")


class CriticState(AgentState):
    exec_result: Dict[str, Any]
    exec_attempts: int
    critic_verdict: str  # "PASS" | "PATCH" | "RETRY_SAME" | "ESCALATE"
    parsed_error: Dict[str, Any]
    patch_prompt: str
    cycle_log: List[Dict[str, Any]]
    hitl_enabled: bool
    hitl_approved: bool
    hitl_event: Any
    hitl_callback: Any


async def node_hitl_pause(state: CriticState) -> dict:
    """Human-in-the-Loop checkpoint before code execution."""
    if state.get("hitl_enabled"):
        callback = state.get("hitl_callback")
        if callback:
            callback()

        event = state.get("hitl_event")
        if event:
            await event.wait()

        if not state.get("hitl_approved", True):
            log.info("HitL rejected execution. Escalating.")
            return {
                "critic_verdict": "ESCALATE",
                "exec_result": {"stdout": "", "stderr": "Execution cancelled by user.", "exit_code": -1},
            }

    return {}


async def node_sandbox_execute(state: CriticState) -> dict:
    """Executes generated code in safe subprocess sandbox."""
    code = state.get("generated_code", "")
    attempt = state.get("exec_attempts", 0)

    if not code:
        return {
            "exec_result": {"stdout": "", "stderr": "No code generated to execute.", "exit_code": -1, "duration_sec": 0.0},
            "exec_attempts": attempt + 1,
        }

    log.info("Executing generated code (attempt %d/%d)...", attempt + 1, MAX_RETRIES)
    exec_res = execute_python_code(code, timeout_sec=EXEC_TIMEOUT_SEC)
    log.info("Execution complete: exit_code=%d, duration=%.2fs", exec_res["exit_code"], exec_res["duration_sec"])

    return {
        "exec_result": exec_res,
        "exec_attempts": attempt + 1,
    }


async def node_critic_evaluate(state: CriticState) -> dict:
    """Evaluates execution outcome, auto-installs missing dependencies, or determines patch need."""
    exec_res = state.get("exec_result", {})
    attempts = state.get("exec_attempts", 1)
    cycle_log = list(state.get("cycle_log", []))

    cycle_log.append({
        "attempt": attempts,
        "exit_code": exec_res.get("exit_code"),
        "duration_sec": exec_res.get("duration_sec"),
        "stdout": exec_res.get("stdout", "")[:500],
        "stderr": exec_res.get("stderr", "")[:500],
    })

    # Case 1: Execution Succeeded
    if exec_res.get("exit_code") == 0 and not exec_res.get("timed_out"):
        log.info("Critic Verdict: PASS (Exit code 0)")
        return {
            "critic_verdict": "PASS",
            "cycle_log": cycle_log,
        }

    # Case 2: Execution Failed — Parse Traceback
    stderr = exec_res.get("stderr", "")
    parsed = parse_traceback(stderr)
    log.info("Critic Parsed Error: Category=%s | Msg=%s", parsed.get("category"), parsed.get("exc_message"))

    # Auto-install missing module if ModuleNotFoundError
    if parsed.get("category") == "import_error" and parsed.get("missing_module"):
        module = parsed["missing_module"]
        log.info("Auto-installing missing module: %s", module)

        pip_pkg = module
        if module == "sklearn": pip_pkg = "scikit-learn"
        if module == "cv2": pip_pkg = "opencv-python"
        if module == "PIL": pip_pkg = "pillow"

        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "pip", "install", pip_pkg,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        # Retry same code without penalizing attempt counter
        return {
            "critic_verdict": "RETRY_SAME",
            "exec_attempts": max(0, attempts - 1),
            "parsed_error": parsed,
            "cycle_log": cycle_log,
        }

    # Case 3: Retries Exhausted -> Escalate
    if attempts >= MAX_RETRIES:
        log.warning("Critic Verdict: ESCALATE (Reached max retries: %d)", MAX_RETRIES)
        return {
            "critic_verdict": "ESCALATE",
            "parsed_error": parsed,
            "cycle_log": cycle_log,
        }

    # Case 4: Needs Patch -> Build Patch Prompt
    code = state.get("generated_code", "")
    patch_prompt = (
        f"The previous Python script failed with {parsed.get('exc_type', 'Error')}:\n"
        f"Line {parsed.get('line_no', 'unknown')}: {parsed.get('exc_message', '')}\n"
        f"Fix Suggestion: {parsed.get('suggestion', '')}\n\n"
        f"Failing Script:\n```python\n{code}\n```\n\n"
        f"Rewrite the script to fix this error. Return ONLY executable Python code in ```python ... ```."
    )

    return {
        "critic_verdict": "PATCH",
        "parsed_error": parsed,
        "patch_prompt": patch_prompt,
        "cycle_log": cycle_log,
    }


async def node_patcher(state: CriticState) -> dict:
    """Generates patched code using local SLM."""
    patch_prompt = state.get("patch_prompt", "")
    budget = state.get("effective_token_budget", 4096)

    llm = ChatOllama(
        model=LOCAL_CODING_SLM,
        base_url=OLLAMA_BASE_URL,
        num_ctx=budget,
        temperature=0.2,
        num_predict=1536,
    )

    messages = [
        SystemMessage(content="You are an expert Python bug fixer. Return ONLY the complete fixed Python code in ```python ... ```."),
        HumanMessage(content=patch_prompt),
    ]

    response = await llm.ainvoke(messages)
    fixed_code = _extract_code_block(response.content)

    return {
        "generated_code": fixed_code,
        "messages": [AIMessage(content=f"```python\n{fixed_code}\n```")],
    }


async def node_log_success(state: CriticState) -> dict:
    """Synchronously syncs memory to SQLite (frames.db), Git Markdown, and ChromaDB."""
    code = state.get("generated_code", "")
    scope = state.get("target_scope", "sda.coding")
    prompt = state.get("raw_prompt", "Coding Task")

    # 1. Dual-tier scoped memory write
    memory_engine.store_frame(
        scope=scope,
        key=f"solution_{int(asyncio.get_event_loop().time() * 1000)}",
        value=f"Task: {prompt} | Lines of code: {len(code.splitlines())}",
        sensitivity="INTERNAL",
        author="SEAF_CRITIC_AGENT",
    )

    # 2. ChromaDB experience store write
    rag_engine.store_experience(problem=prompt, solution=code)

    log.info("Success logged to Scoped Memory (%s) and ChromaDB experience store", scope)
    return {}


def edge_verdict_router(state: CriticState) -> str:
    verdict = state.get("critic_verdict", "ESCALATE").lower()
    if verdict == "pass":
        return "log_success"
    elif verdict == "patch":
        return "patcher"
    elif verdict == "retry_same":
        return "sandbox_execute"
    return "end"


def build_full_graph() -> Any:
    """Assembles the full SEAF reasoner-critic agentic state machine."""
    workflow = StateGraph(CriticState)

    # Add nodes
    workflow.add_node("pii_and_telemetry", node_pii_and_telemetry)
    workflow.add_node("rag_retrieval", node_rag_retrieval)
    workflow.add_node("caveman_compress", node_caveman_compress)
    workflow.add_node("reasoner_generate", node_reasoner_generate)
    workflow.add_node("ponytail_review", node_ponytail_review)
    workflow.add_node("hitl_pause", node_hitl_pause)
    workflow.add_node("sandbox_execute", node_sandbox_execute)
    workflow.add_node("critic_evaluate", node_critic_evaluate)
    workflow.add_node("patcher", node_patcher)
    workflow.add_node("log_success", node_log_success)

    # Set entry point
    workflow.set_entry_point("pii_and_telemetry")

    # Edges
    workflow.add_edge("pii_and_telemetry", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "caveman_compress")
    workflow.add_edge("caveman_compress", "reasoner_generate")
    workflow.add_edge("reasoner_generate", "ponytail_review")
    workflow.add_edge("ponytail_review", "hitl_pause")
    workflow.add_edge("hitl_pause", "sandbox_execute")
    workflow.add_edge("sandbox_execute", "critic_evaluate")

    # Conditional branching on critic verdict
    workflow.add_conditional_edges(
        "critic_evaluate",
        edge_verdict_router,
        {
            "log_success": "log_success",
            "patcher": "patcher",
            "sandbox_execute": "sandbox_execute",
            "end": END,
        },
    )

    workflow.add_edge("patcher", "sandbox_execute")
    workflow.add_edge("log_success", END)

    return workflow.compile()
