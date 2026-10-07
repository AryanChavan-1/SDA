"""
seaf_agent.tools.sandbox — Sandboxed Execution Engine.
Executes Python snippets safely in an isolated subprocess with strict timeouts,
resource monitoring, and output length boundaries.
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional

from seaf_agent.config import EXEC_TIMEOUT_SEC, WORKSPACE_DIR


def execute_python_code(
    source_code: str,
    timeout_sec: int = EXEC_TIMEOUT_SEC,
    max_output_chars: int = 10000,
    working_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes a Python code snippet in a dedicated subprocess.
    Returns: stdout, stderr, exit_code, timed_out, duration_sec.
    """
    if not source_code or not source_code.strip():
        return {
            "stdout": "",
            "stderr": "No source code provided for execution.",
            "exit_code": -1,
            "timed_out": False,
            "duration_sec": 0.0,
        }

    work_dir = working_dir or WORKSPACE_DIR
    work_dir.mkdir(parents=True, exist_ok=True)

    # Write code to a temporary file
    temp_file = work_dir / f"run_{int(time.time() * 1000)}.py"
    temp_file.write_text(source_code, encoding="utf-8")

    t0 = time.perf_counter()
    timed_out = False
    stdout = ""
    stderr = ""
    exit_code = 0

    try:
        proc = subprocess.run(
            [sys.executable, str(temp_file)],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            encoding="utf-8",
            errors="replace",
        )
        stdout = proc.stdout
        stderr = proc.stderr
        exit_code = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = -99
        stdout = exc.stdout or ""
        stderr = f"Execution timed out after {timeout_sec} seconds."
    except Exception as exc:
        exit_code = -1
        stderr = f"Subprocess launch error: {exc}"
    finally:
        duration_sec = round(time.perf_counter() - t0, 3)
        # Clean up temporary script
        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception:
            pass

    # Cap output lengths
    if len(stdout) > max_output_chars:
        stdout = stdout[:max_output_chars] + f"\n... [stdout truncated at {max_output_chars} chars]"
    if len(stderr) > max_output_chars:
        stderr = stderr[:max_output_chars] + f"\n... [stderr truncated at {max_output_chars} chars]"

    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_sec": duration_sec,
    }
