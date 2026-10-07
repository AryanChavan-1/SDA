"""
seaf_agent.test_coding_run — CLI Integration Test for SEAF Autonomous Coding Loop.
Runs the complete LangGraph agentic loop:
PII check -> Telemetry budget -> RAG -> Reasoner -> Ponytail AST -> Sandbox Exec -> Critic -> Memory Sync.
"""

import asyncio
import json
import logging
# Ensure UTF-8 console encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from langchain_core.messages import HumanMessage
from seaf_agent.config import BASE_CONTEXT_WINDOW
from seaf_agent.critic import build_full_graph, CriticState

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

TEST_PROMPT = (
    "Write a self-contained Python script to calculate the prime factors of 123456789. "
    "Print the result as a sorted list."
)


async def main():
    print("=" * 70)
    print("   SEAF-SDA Coding Loop Test — Local Edge Inference")
    print("=" * 70)

    graph = build_full_graph()

    initial: CriticState = {
        "messages": [HumanMessage(content=TEST_PROMPT)],
        "raw_prompt": TEST_PROMPT,
        "target_scope": "sda.coding.math",
        "task_complexity": "tier_1",
        "pii_inspection": {},
        "telemetry_pressure": {},
        "effective_token_budget": BASE_CONTEXT_WINDOW,
        "raw_context": "",
        "compressed_context": "",
        "generated_code": "",
        "ponytail_result": {},
        "token_count": 0,
        "exec_result": {},
        "exec_attempts": 0,
        "critic_verdict": "",
        "parsed_error": {},
        "patch_prompt": "",
        "cycle_log": [],
        "hitl_enabled": False,
        "hitl_approved": True,
        "hitl_event": None,
        "hitl_callback": None,
    }

    result = await graph.ainvoke(initial)

    print("\n" + "=" * 70)
    print("  GENERATED SOLUTION")
    print("=" * 70)
    print(result.get("generated_code", "No code generated"))

    print("\n" + "=" * 70)
    print("  CRITIC EVALUATION & OUTCOME")
    print("=" * 70)
    print(f"  Verdict  : {result.get('critic_verdict')}")
    print(f"  Attempts : {result.get('exec_attempts')}")
    print(f"  Exec Out : {result.get('exec_result', {}).get('stdout', '').strip()}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
