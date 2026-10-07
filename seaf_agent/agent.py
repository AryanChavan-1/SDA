"""
seaf_agent.agent — SEAF-Integrated Autonomous Cognitive Router.
Coordinates:
- Zero-exposure PII screening before prompt processing (SEAF Pillar 1)
- Live hardware telemetry & adaptive token budgeting (SEAF Pillar 2)
- Scoped memory & experience retrieval (SEAF Pillar 3)
- Caveman context compression & Ponytail AST review (VRAM guardrails)
- Local SLM generation (qwen2.5-coder:7b / phi4-mini via Ollama)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, add_messages

from seaf_agent.config import (
    OLLAMA_BASE_URL,
    LOCAL_CODING_SLM,
    LOCAL_FALLBACK_SLM,
    BASE_CONTEXT_WINDOW,
    THROTTLED_CONTEXT_WINDOW,
    PONYTAIL_STRICT,
)
from seaf_agent.gateway import gateway
from seaf_agent.memory import memory_engine
from seaf_agent.rag import rag_engine
from seaf_agent.telemetry import telemetry
from seaf_agent.tools.filters import apply_caveman_compression, run_ponytail_review

log = logging.getLogger("seaf.agent")


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    raw_prompt: str
    target_scope: str
    task_complexity: str
    pii_inspection: Dict[str, Any]
    telemetry_pressure: Dict[str, Any]
    effective_token_budget: int
    raw_context: str
    compressed_context: str
    generated_code: str
    ponytail_result: Dict[str, Any]
    token_count: int


def _extract_code_block(text: str) -> str:
    """Extracts Python code from model response (closed blocks, unclosed blocks, or plain code)."""
    if not text:
        return ""

    # 1. Standard markdown code fences ```python ... ```
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 2. Unclosed code block (when output token limit was reached)
    m = re.search(r"```(?:python)?\s*\n(.+)", text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        code = re.sub(r"`{1,3}$", "", code).strip()
        if len(code) > 20:
            return code

    # 3. Plain Python code without markdown
    lines = text.strip().splitlines()
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("import ", "from ", "def ", "class ", "#!", "if __name__")):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines and len("\n".join(code_lines)) > 20:
        return "\n".join(code_lines).strip()

    return text.strip()


async def node_pii_and_telemetry(state: AgentState) -> dict:
    """Step 1: Check PII and evaluate host memory pressure once per invocation."""
    prompt = state.get("raw_prompt", "")
    if not prompt and state.get("messages"):
        prompt = state["messages"][-1].content

    # Pillar 1: PII inspection (<1ms)
    pii_res = gateway.inspect_payload(prompt)

    # Pillar 2: Single sample of hardware telemetry
    pressure = telemetry.sample()
    under_pressure = pressure["under_pressure"]

    # Adaptive token budget
    budget = THROTTLED_CONTEXT_WINDOW if under_pressure else BASE_CONTEXT_WINDOW

    log.info(
        "PII Gateway: %s | RAM: %.1f%% (Throttled: %s) | Budget: %d tokens",
        "PII_DETECTED" if pii_res["has_pii"] else "CLEAN",
        pressure["ram_percent"],
        under_pressure,
        budget,
    )

    return {
        "raw_prompt": prompt,
        "pii_inspection": pii_res,
        "telemetry_pressure": pressure,
        "effective_token_budget": budget,
    }


async def node_rag_retrieval(state: AgentState) -> dict:
    """Step 2: Semantic search in ChromaDB and past experiences."""
    prompt = state.get("raw_prompt", "")
    scope = state.get("target_scope", "sda.coding")

    # Query local RAG and scoped memory
    kb_context = rag_engine.query(prompt, top_k=3, max_chars=1500)
    exp_context = rag_engine.query_experience(prompt, top_k=2)

    scoped_frames = memory_engine.query_memory(scope)
    scoped_context_lines = [f"{f['key']}: {f['value']}" for f in scoped_frames[-5:]]
    scoped_text = "\n".join(scoped_context_lines)

    combined = ""
    if kb_context:
        combined += f"[Knowledge Base]\n{kb_context}\n\n"
    if exp_context:
        combined += f"[Past Solutions]\n{exp_context}\n\n"
    if scoped_text:
        combined += f"[Scoped Governance Memory ({scope})]\n{scoped_text}\n"

    return {"raw_context": combined}


async def node_caveman_compress(state: AgentState) -> dict:
    """Step 3: Strip filler words to fit hardware token budget."""
    raw_ctx = state.get("raw_context", "")
    budget = state.get("effective_token_budget", BASE_CONTEXT_WINDOW)

    comp = apply_caveman_compression(raw_ctx, token_budget=budget)
    return {"compressed_context": comp["compressed_text"]}


async def node_reasoner_generate(state: AgentState) -> dict:
    """Step 4: Generate Python solution using local Ollama model."""
    prompt = state.get("raw_prompt", "")
    ctx = state.get("compressed_context", "")
    budget = state.get("effective_token_budget", BASE_CONTEXT_WINDOW)

    llm = ChatOllama(
        model=LOCAL_CODING_SLM,
        base_url=OLLAMA_BASE_URL,
        num_ctx=budget,
        temperature=0.2,
        num_predict=1536,
    )

    system_prompt = (
        "You are SEAF-SDA, an autonomous coding agent operating on an edge laptop. "
        "Return ONLY executable Python code enclosed inside ```python ... ``` blocks. "
        "No conversational filler, no explanations."
    )

    messages = [SystemMessage(content=system_prompt)]
    if ctx:
        messages.append(HumanMessage(content=f"Reference knowledge:\n{ctx}"))
    messages.append(HumanMessage(content=prompt))

    try:
        response = await llm.ainvoke(messages)
        code = _extract_code_block(response.content)
    except Exception as exc:
        log.warning("Local coding SLM failed (%s); falling back to edge SLM", exc)
        llm_fallback = ChatOllama(
            model=LOCAL_FALLBACK_SLM,
            base_url=OLLAMA_BASE_URL,
            num_ctx=2048,
        )
        response = await llm_fallback.ainvoke(messages)
        code = _extract_code_block(response.content)

    return {
        "generated_code": code,
        "messages": [AIMessage(content=f"```python\n{code}\n```")],
    }


async def node_ponytail_review(state: AgentState) -> dict:
    """Step 5: AST code bloat and safety review."""
    code = state.get("generated_code", "")
    res = run_ponytail_review(code)
    return {"ponytail_result": res}
