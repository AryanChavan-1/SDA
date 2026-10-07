"""
seaf_agent.tools.traceback_parser — Structured Traceback Analysis & Patch Guidance.
Parses stderr from code execution, classifies the failure category, and extracts
actionable context for the Critic and Patcher nodes.
"""

import re
from typing import Dict, Any, Optional

_ERROR_CATEGORIES = [
    (re.compile(r"SyntaxError"), "syntax", "Fix the syntax error at the indicated line."),
    (re.compile(r"IndentationError"), "indentation", "Correct indentation using 4-space tabs consistently."),
    (re.compile(r"NameError"), "undefined_name", "Ensure all variables and functions are defined before use."),
    (re.compile(r"AttributeError"), "attr_error", "Verify object type and attribute name; check for None values."),
    (re.compile(r"TypeError"), "type_error", "Check argument types and function signatures."),
    (re.compile(r"ValueError"), "value_error", "Validate input ranges and data formats before passing to function."),
    (re.compile(r"KeyError"), "key_error", "Use dict.get() with a default value or check key existence first."),
    (re.compile(r"IndexError"), "index_error", "Validate list/array index bounds before accessing element."),
    (re.compile(r"ImportError|ModuleNotFoundError"), "import_error", "Install the missing package or correct module import path."),
    (re.compile(r"FileNotFoundError"), "file_not_found", "Ensure file path exists using pathlib.Path.exists() before opening."),
    (re.compile(r"ZeroDivisionError"), "zero_division", "Guard denominator with `if denom != 0` before division."),
    (re.compile(r"AssertionError"), "assertion", "Fix logic condition that caused assertion failure."),
    (re.compile(r"RuntimeError"), "runtime", "Inspect runtime state; check device/tensor dimensions."),
]


def parse_traceback(stderr: str) -> Dict[str, Any]:
    """Extracts structured exception info from raw stderr."""
    if not stderr or not stderr.strip():
        return {"has_error": False, "category": "none", "exc_type": "", "exc_message": ""}

    lines = [ln.strip() for ln in stderr.strip().splitlines() if ln.strip()]

    # Extract exception line (usually last line of traceback)
    last_line = lines[-1] if lines else ""
    exc_type = ""
    exc_message = ""
    if ":" in last_line:
        parts = last_line.split(":", 1)
        exc_type = parts[0].strip()
        exc_message = parts[1].strip()
    else:
        exc_type = last_line
        exc_message = ""

    # Classify category
    category = "generic_error"
    suggestion = "Review the execution traceback and apply targeted fix."
    for pat, cat, hint in _ERROR_CATEGORIES:
        if pat.search(stderr):
            category = cat
            suggestion = hint
            break

    # Extract line number
    line_matches = re.findall(r'line (\d+)', stderr)
    line_no = int(line_matches[-1]) if line_matches else None

    # Check for missing package name if import error
    missing_module = None
    if category == "import_error":
        m = re.search(r"No module named '([^']+)'", stderr)
        if m:
            missing_module = m.group(1)

    return {
        "has_error": True,
        "category": category,
        "exc_type": exc_type,
        "exc_message": exc_message,
        "line_no": line_no,
        "missing_module": missing_module,
        "suggestion": suggestion,
        "raw_stderr": stderr[:2000],
    }
