"""
SEAF Memory Engine (seaf_memory.py)
----------------------------------
Implements a dual-tier canonical memory architecture for the Sovereign-Edge Agentic Framework:
1. SQLite Scope-Chained Knowledge Graph (frames.db) for strict project/department isolation.
2. Markdown-as-Source-of-Truth Sync for flat-file Git version control and MLOps auditing.
"""

import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime, timezone

class SEAFMemoryEngine:
    def __init__(self, db_path=None, md_dir=None):
        base = Path(__file__).resolve().parent / "scratch_seaf_poc"
        self.db_path = Path(db_path) if db_path is not None else base / "frames.db"
        self.md_dir = Path(md_dir) if md_dir is not None else base / "memory_md"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.md_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    sensitivity TEXT DEFAULT 'INTERNAL',
                    author_agent TEXT DEFAULT 'SEAF_SYSTEM'
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_scope ON memory_frames(scope)")
            conn.commit()

    def store_frame(self, scope: str, key: str, value: str, sensitivity: str = "INTERNAL", agent: str = "SEAF_AGENT"):
        """Stores a memory frame inside SQLite under a designated scope."""
        ts = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO memory_frames (timestamp, scope, key, value, sensitivity, author_agent)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (ts, scope.lower(), key, value, sensitivity.upper(), agent))
            conn.commit()
        
        # Trigger automatic Markdown export for procedural memory sync
        self.export_markdown_procedural()

    def query_memory(self, target_scope: str, include_parent_scopes: bool = True):
        """
        Retrieves memory frames enforcing scope-chain inheritance.
        Child scope (e.g., 'finance.audit_2026') inherits parent scopes ('finance', 'global').
        Cross-department scopes are strictly isolated.
        """
        target_scope = target_scope.lower()
        allowed_scopes = [target_scope]
        
        if include_parent_scopes:
            parts = target_scope.split(".")
            for i in range(1, len(parts)):
                allowed_scopes.append(".".join(parts[:i]))
            if "global" not in allowed_scopes:
                allowed_scopes.append("global")

        placeholders = ",".join(["?"] * len(allowed_scopes))
        query = f"SELECT timestamp, scope, key, value, sensitivity, author_agent FROM memory_frames WHERE scope IN ({placeholders}) ORDER BY id ASC"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, allowed_scopes)
            rows = cursor.fetchall()

        results = []
        for r in rows:
            results.append({
                "timestamp": r[0],
                "scope": r[1],
                "key": r[2],
                "value": r[3],
                "sensitivity": r[4],
                "author_agent": r[5]
            })
        return results

    def export_markdown_procedural(self):
        """Exports procedural rules and learned routines to flat Markdown files for Git versioning and MLOps auditing."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT scope, key, value, timestamp, author_agent, sensitivity FROM memory_frames ORDER BY scope, id ASC")
            rows = cursor.fetchall()

        grouped = {}
        for r in rows:
            scope, key, val, ts, agent, sens = r
            grouped.setdefault(scope, []).append((key, val, ts, agent, sens))

        # Write master procedural Markdown
        master_md = self.md_dir / "procedural_memory_master.md"
        with open(master_md, "w", encoding="utf-8") as f:
            f.write("# SEAF Canonical Memory - Master Audit Trail\n")
            f.write(f"*Last Exported UTC: {datetime.now(timezone.utc).isoformat()}*\n\n")
            f.write("> **Governance Note:** This file serves as the Git-versionable Source-of-Truth for enterprise MLOps compliance.\n\n")

            for scope, frames in grouped.items():
                f.write(f"## Scope: `{scope}`\n")
                f.write("| Key | Value | Sensitivity | Agent | Timestamp |\n")
                f.write("|---|---|---|---|---|\n")
                for item in frames:
                    f.write(f"| `{item[0]}` | {item[1]} | `{item[4]}` | `{item[3]}` | `{item[2]}` |\n")
                f.write("\n")

        return str(master_md)
