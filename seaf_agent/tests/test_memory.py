"""Test SEAFMemoryEngine scope inheritance, isolation, and Markdown export."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from seaf_agent.memory import SEAFMemoryEngine


def test_memory_scopes_and_markdown(tmp_path):
    db_file = tmp_path / "test_frames.db"
    md_folder = tmp_path / "test_md"
    mem = SEAFMemoryEngine(db_path=db_file, md_dir=md_folder)

    # 1. Store frames in parent, child, and sibling scopes
    mem.store_frame("global", "sys_version", "1.0.0", "INTERNAL", "SystemAgent")
    mem.store_frame("finance", "quarterly_budget", "$500k", "CONFIDENTIAL", "FinanceLead")
    mem.store_frame("finance.payroll", "jane_salary", "$120k", "HIGHLY_CONFIDENTIAL", "PayrollAgent")
    mem.store_frame("engineering.backend", "db_pool_size", "20", "INTERNAL", "BackendDev")

    # 2. Query child scope 'finance.payroll' -> should see finance.payroll, finance, and global
    res_child = mem.query_memory("finance.payroll")
    keys_child = [r["key"] for r in res_child]
    assert "jane_salary" in keys_child
    assert "quarterly_budget" in keys_child
    assert "sys_version" in keys_child
    assert "db_pool_size" not in keys_child  # Sibling scope isolated!

    # 3. Query sibling scope 'engineering.backend' -> should NOT see any finance frames
    res_eng = mem.query_memory("engineering.backend")
    keys_eng = [r["key"] for r in res_eng]
    assert "db_pool_size" in keys_eng
    assert "sys_version" in keys_child
    assert "quarterly_budget" not in keys_eng
    assert "jane_salary" not in keys_eng

    # 4. Verify Markdown export
    master_file = md_folder / "procedural_memory_master.md"
    assert master_file.exists()
    content = master_file.read_text(encoding="utf-8")
    assert "SEAF Procedural Memory Master Audit Trail" in content
    assert "jane_salary" in content
    assert "quarterly_budget" in content
    print("[PASS] test_memory_scopes_and_markdown: All scope isolation and markdown export checks passed!")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_memory_scopes_and_markdown(Path(td))
    print("All memory tests passed successfully!")
