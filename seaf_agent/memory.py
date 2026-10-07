"""
seaf_agent.memory — Dual-Tier Scoped Memory Engine (SEAF Pillar 3).
Contracts:
1. SQLite relational store (frames.db) with WAL mode enabled (PRAGMA journal_mode=WAL).
2. Dot-notation scope-chain inheritance:
   - Querying 'sda.finance.payroll' inherits 'sda.finance.payroll', 'sda.finance', 'sda.global'.
   - Sibling subtrees (e.g. 'sda.marketing') are 100% invisible (0% data leakage).
   - Strict scope validation: malformed scopes default to the narrowest sub-scope (never global).
3. Synchronous Git-versionable Markdown sync (procedural_memory_master.md) on every write.
"""

import os
import sqlite3
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from seaf_agent.config import DB_PATH, MEMORY_MD_DIR, MASTER_MD_PATH

_VALID_SCOPE_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)*$")


class SEAFMemoryEngine:
    def __init__(self, db_path: Optional[Path] = None, md_dir: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.md_dir = Path(md_dir) if md_dir else MEMORY_MD_DIR
        self.master_md_path = self.md_dir / "procedural_memory_master.md"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.md_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Pitfall 2: WAL mode prevents database locks during concurrent reads/writes
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    sensitivity TEXT DEFAULT 'INTERNAL',
                    author TEXT DEFAULT 'SEAF_SYSTEM'
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_scope ON memory_frames(scope);")
            conn.commit()
        finally:
            conn.close()

    def validate_and_normalize_scope(self, scope: str) -> str:
        """
        Validates dot-notation hierarchical scope.
        Pitfall 3: If malformed, default to the narrowest sub-scope rather than global scope.
        """
        if not scope or not isinstance(scope, str):
            return "sda.isolated_default"

        s = scope.strip().lower()
        # Clean invalid characters
        s = re.sub(r"[^a-z0-9_.-]", "_", s)
        s = re.sub(r"\.{2,}", ".", s).strip(".")

        if not s or not _VALID_SCOPE_PATTERN.match(s):
            # Fallback to narrowest isolated scope, NOT global
            return "sda.isolated_subscope"

        return s

    def store_frame(
        self,
        scope: str,
        key: str,
        value: str,
        sensitivity: str = "INTERNAL",
        author: str = "SEAF_AGENT",
    ) -> int:
        """Stores a memory frame inside SQLite under a designated scope and syncs to Markdown."""
        ts = datetime.now(timezone.utc).isoformat()
        clean_scope = self.validate_and_normalize_scope(scope)

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memory_frames (timestamp, scope, key, value, sensitivity, author)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (ts, clean_scope, key, value, sensitivity.upper(), author))
            frame_id = cursor.lastrowid
            conn.commit()
        finally:
            conn.close()

        # Synchronously export Markdown audit log
        self.export_markdown_procedural()
        return frame_id

    def _build_scope_hierarchy(self, target_scope: str) -> List[str]:
        """
        Builds inheritance chain from target_scope up to root + global.
        Example: 'sda.finance.payroll' -> ['sda.finance.payroll', 'sda.finance', 'sda', 'global', 'sda.global']
        """
        parts = target_scope.split(".")
        chain = []
        for i in range(len(parts), 0, -1):
            chain.append(".".join(parts[:i]))

        # Include parent globals if not already present
        if "sda.global" not in chain and parts[0] == "sda":
            chain.append("sda.global")
        if "global" not in chain:
            chain.append("global")
        return chain

    def query_memory(self, target_scope: str, include_parent_scopes: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieves memory frames enforcing scope-chain inheritance.
        Child scopes inherit parents; sibling subtrees are strictly isolated (0% leakage).
        """
        clean_scope = self.validate_and_normalize_scope(target_scope)

        if include_parent_scopes:
            search_scopes = self._build_scope_hierarchy(clean_scope)
        else:
            search_scopes = [clean_scope]

        placeholders = ",".join(["?"] * len(search_scopes))
        query = f"""
            SELECT id, timestamp, scope, key, value, sensitivity, author
            FROM memory_frames
            WHERE scope IN ({placeholders})
            ORDER BY id ASC
        """

        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, search_scopes)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def export_markdown_procedural(self) -> Path:
        """
        Renders all memory frames into structured Markdown tables grouped by scope.
        Enables git diff and human readability for MLOps compliance.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT scope, timestamp, key, value, sensitivity, author
                FROM memory_frames
                ORDER BY scope ASC, id ASC
            """)
            rows = cursor.fetchall()
        finally:
            conn.close()

        grouped: Dict[str, List[sqlite3.Row]] = {}
        for r in rows:
            sc = r["scope"]
            grouped.setdefault(sc, []).append(r)

        lines = [
            "# SEAF Procedural Memory Master Audit Trail",
            f"> Last Synchronized (UTC): {datetime.now(timezone.utc).isoformat()}",
            f"> Storage Backend: SQLite ({self.db_path.name}) with WAL Mode & Scope Governance",
            "",
            "---",
            "",
        ]

        if not grouped:
            lines.append("*No memory frames stored yet.*")
        else:
            for sc in sorted(grouped.keys()):
                lines.append(f"## Scope: `{sc}`")
                lines.append("")
                lines.append("| Timestamp | Key | Value | Sensitivity | Author |")
                lines.append("|---|---|---|---|---|")
                for item in grouped[sc]:
                    val = str(item["value"]).replace("|", "\\|").replace("\n", " ")
                    if len(val) > 120:
                        val = val[:117] + "..."
                    lines.append(
                        f"| {item['timestamp'][:19]} | `{item['key']}` | {val} | `{item['sensitivity']}` | {item['author']} |"
                    )
                lines.append("")

        content = "\n".join(lines)
        self.master_md_path.write_text(content, encoding="utf-8")
        return self.master_md_path


# Global instance
memory_engine = SEAFMemoryEngine()
