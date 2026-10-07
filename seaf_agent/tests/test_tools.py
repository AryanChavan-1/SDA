"""Test tools: Caveman compressor, Ponytail review, Traceback parser, Sandbox execution."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from seaf_agent.tools.filters import apply_caveman_compression, run_ponytail_review
from seaf_agent.tools.traceback_parser import parse_traceback
from seaf_agent.tools.sandbox import execute_python_code


def test_caveman_compression():
    verbose = "Certainly! Of course, I hope this helps you! Here is the python code:\nprint('hello world')"
    comp = apply_caveman_compression(verbose)
    assert "Certainly" not in comp["compressed_text"]
    assert "hello world" in comp["compressed_text"]
    print("[PASS] test_caveman_compression")


def test_ponytail_review():
    bloated_code = """
class MyAbstractFactory(AbstractFactory):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
"""
    rev = run_ponytail_review(bloated_code)
    assert len(rev["findings"]) > 0
    sigs = [f.get("signature") for f in rev["findings"] if "signature" in f]
    assert "AbstractFactory" in sigs
    print("[PASS] test_ponytail_review")


def test_traceback_parser():
    sample_trace = """Traceback (most recent call last):
  File "test.py", line 42, in <module>
    x = 10 / 0
ZeroDivisionError: division by zero
"""
    parsed = parse_traceback(sample_trace)
    assert parsed["has_error"]
    assert parsed["category"] == "zero_division"
    assert parsed["line_no"] == 42
    assert "denom != 0" in parsed["suggestion"]
    print("[PASS] test_traceback_parser")


def test_sandbox_execution():
    code = "import math\nprint(f'PI={round(math.pi, 4)}')"
    res = execute_python_code(code, timeout_sec=5)
    assert res["exit_code"] == 0
    assert "PI=3.1416" in res["stdout"]
    print("[PASS] test_sandbox_execution")


if __name__ == "__main__":
    test_caveman_compression()
    test_ponytail_review()
    test_traceback_parser()
    test_sandbox_execution()
    print("All tools tests passed successfully!")
