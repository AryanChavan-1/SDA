"""
seaf_agent.tools.filters — VRAM Guardrails and Code Quality Filters.
1. CAVEMAN Compression: Strips verbose filler and compresses text to preserve context budget.
2. PONYTAIL AST Review: Static AST inspection detecting bloat, unnecessary factory patterns, and unhandled risky operations.
"""

import ast
import re
from typing import Dict, Any, List, Tuple

# Precompiled regex patterns for filler words
_CAVEMAN_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)\b(certainly|absolutely|of course|sure thing|great question|feel free to)[,!.]?\s*", re.MULTILINE), ""),
    (re.compile(r"(?i)(I hope this (helps|answers your question)[.!]?\s*)", re.MULTILINE), ""),
    (re.compile(r"(?i)(please (let me know|don't hesitate)[^.]*\.\s*)", re.MULTILINE), ""),
    (re.compile(r"(?i)\b(in (other|simple) words|to (summarize|recap|reiterate),?)\s*", re.MULTILINE), ""),
    (re.compile(r"(?i)\b(as (mentioned|stated|noted) (above|earlier|previously),?)\s*", re.MULTILINE), ""),
    (re.compile(r"(?i)^(so,?|well,?|now,?|basically,?|essentially,?)\s+", re.MULTILINE), ""),
    (re.compile(r"\n{3,}"), "\n\n"),
]

_BLOAT_SIGNATURES: List[Tuple[str, str]] = [
    ("AbstractFactory", "Over-engineered factory pattern detected"),
    ("BaseAbstractMixin", "Triple-inheritance bloat detected"),
    ("register_plugin", "Plugin registry added without explicit requirement"),
    ("self.__dict__.update", "Dynamic dict-unpacking antipattern"),
    ("**kwargs", "Unconstrained kwargs without signature validation"),
]


def apply_caveman_compression(raw_text: str, token_budget: int = 2048) -> Dict[str, Any]:
    """Compresses text by stripping conversational boilerplate and capping length."""
    if not raw_text:
        return {"compressed_text": "", "original_chars": 0, "compressed_chars": 0, "truncated": False}

    orig_len = len(raw_text)
    text = raw_text
    for pat, rep in _CAVEMAN_PATTERNS:
        text = pat.sub(rep, text)

    text = text.strip()

    # Approximate token cap (1 token ~= 4 chars)
    char_cap = token_budget * 4
    truncated = len(text) > char_cap
    if truncated:
        cut = text[:char_cap]
        last_period = cut.rfind(". ")
        text = cut[: last_period + 1] if last_period != -1 else cut

    return {
        "compressed_text": text,
        "original_chars": orig_len,
        "compressed_chars": len(text),
        "truncated": truncated,
    }


def run_ponytail_review(code_str: str) -> Dict[str, Any]:
    """Scans code for syntactic validity, structural bloat, and basic error handling."""
    if not code_str or not code_str.strip():
        return {"passed": False, "reason": "No code provided for review"}

    # 1. Parse AST
    try:
        tree = ast.parse(code_str)
    except SyntaxError as exc:
        return {
            "passed": False,
            "syntax_valid": False,
            "reason": f"SyntaxError: {exc.msg} at line {exc.lineno}",
            "findings": [{"type": "syntax_error", "line": exc.lineno, "message": exc.msg}],
        }

    findings = []

    # 2. Check for bloat signatures
    for sig, reason in _BLOAT_SIGNATURES:
        if sig in code_str:
            findings.append({"type": "bloat_signature", "signature": sig, "message": reason})

    # 3. Check for unhandled exceptions in try blocks
    has_try = any(isinstance(node, ast.Try) for node in ast.walk(tree))
    has_complex_calls = any(
        isinstance(node, ast.Call) and getattr(node.func, "id", "") in {"open", "socket", "connect"}
        for node in ast.walk(tree)
    )

    if has_complex_calls and not has_try:
        findings.append({
            "type": "missing_guard",
            "message": "I/O or network call found without enclosing try-except block"
        })

    passed = len([f for f in findings if f["type"] == "syntax_error"]) == 0
    return {
        "passed": passed,
        "syntax_valid": True,
        "findings": findings,
        "findings_count": len(findings),
    }
