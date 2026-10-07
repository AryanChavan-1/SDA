"""
seaf_agent.app — Modern CustomTkinter Desktop GUI for SEAF-SDA.
Features:
- Real-time Hardware Telemetry (Host RAM % gauge with >80% throttling alert, RTX 3050 VRAM meter).
- Zero-Exposure PII Guard Indicator (badge showing PII screening status).
- Scoped Governance Selector (dot-notation scopes e.g. sda.coding, sda.finance).
- Human-in-the-Loop (HitL) execution checkpoint modal.
- Multi-step progress bar (PII Guard -> Ingest -> Reason -> Exec -> Critic -> Sync).
- Tabbed View: Execution Logs, Generated Code, Critic Audit, and Scoped Memory Browser.
"""

import asyncio
import logging
import os
import sys
import threading
import time
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import customtkinter as ctk
from langchain_core.messages import HumanMessage

from seaf_agent.config import (
    BASE_CONTEXT_WINDOW,
    THROTTLED_CONTEXT_WINDOW,
    MASTER_MD_PATH,
    WORKSPACE_DIR,
)
from seaf_agent.critic import build_full_graph, CriticState
from seaf_agent.gateway import gateway
from seaf_agent.memory import memory_engine
from seaf_agent.telemetry import telemetry

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

log = logging.getLogger("seaf.gui")


class TextboxLogHandler(logging.Handler):
    def __init__(self, textbox, after_func):
        super().__init__()
        self.textbox = textbox
        self.after_func = after_func
        self.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))

    def emit(self, record):
        msg = self.format(record)
        self.after_func(0, self._append, msg)
        
    def _append(self, msg):
        self.textbox.configure(state="normal")
        self.textbox.insert("end", msg + "\n")
        self.textbox.see("end")
        self.textbox.configure(state="disabled")


class SEAFDesktopApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("SEAF-SDA — Sovereign-Edge Autonomous Agent")
        self.geometry("1180x760")
        self.minsize(960, 640)

        self.graph = None
        self.is_running = False
        self.hitl_event = None
        self.hitl_approved_var = False

        self._build_ui()
        
        # Attach log handler to root 'seaf' logger to capture all backend logs
        seaf_logger = logging.getLogger("seaf")
        seaf_logger.setLevel(logging.INFO)
        gui_handler = TextboxLogHandler(self.log_box, self.after)
        seaf_logger.addHandler(gui_handler)

        self._start_telemetry_loop()

    def _build_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # =====================================================================
        # SIDEBAR FRAME (Navigation, Telemetry & Controls)
        # =====================================================================
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_rowconfigure(14, weight=1)

        # Title
        self.logo_label = ctk.CTkLabel(
            self.sidebar, text="⚡ SEAF-SDA", font=ctk.CTkFont(size=22, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 2))

        self.sub_label = ctk.CTkLabel(
            self.sidebar, text="Sovereign-Edge Agentic Framework",
            font=ctk.CTkFont(size=11, slant="italic"), text_color="gray70"
        )
        self.sub_label.grid(row=1, column=0, padx=20, pady=(0, 15))

        # Divider
        ctk.CTkFrame(self.sidebar, height=2, fg_color="gray30").grid(row=2, column=0, sticky="ew", padx=15, pady=5)

        # Telemetry Section
        ctk.CTkLabel(self.sidebar, text="💻 HARDWARE TELEMETRY", font=ctk.CTkFont(size=12, weight="bold"), text_color="#64B5F6").grid(row=3, column=0, sticky="w", padx=20, pady=(5, 2))

        # RAM Gauge
        self.ram_label = ctk.CTkLabel(self.sidebar, text="Host RAM: --%", font=ctk.CTkFont(size=12))
        self.ram_label.grid(row=4, column=0, sticky="w", padx=20, pady=0)
        self.ram_bar = ctk.CTkProgressBar(self.sidebar, height=10)
        self.ram_bar.grid(row=5, column=0, sticky="ew", padx=20, pady=(2, 6))
        self.ram_bar.set(0)

        # GPU VRAM Gauge
        self.gpu_label = ctk.CTkLabel(self.sidebar, text="GPU VRAM: --%", font=ctk.CTkFont(size=12))
        self.gpu_label.grid(row=6, column=0, sticky="w", padx=20, pady=0)
        self.gpu_bar = ctk.CTkProgressBar(self.sidebar, height=10)
        self.gpu_bar.grid(row=7, column=0, sticky="ew", padx=20, pady=(2, 12))
        self.gpu_bar.set(0)

        # Throttle Indicator
        self.throttle_status = ctk.CTkLabel(
            self.sidebar, text="Mode: Normal (4096 ctx)", font=ctk.CTkFont(size=11), text_color="#4CAF50"
        )
        self.throttle_status.grid(row=8, column=0, sticky="w", padx=20, pady=(0, 10))

        # Divider
        ctk.CTkFrame(self.sidebar, height=2, fg_color="gray30").grid(row=9, column=0, sticky="ew", padx=15, pady=5)

        # Scope Selection
        ctk.CTkLabel(self.sidebar, text="🗄️ MEMORY GOVERNANCE SCOPE", font=ctk.CTkFont(size=12, weight="bold"), text_color="#BA68C8").grid(row=10, column=0, sticky="w", padx=20, pady=(5, 2))
        self.scope_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=["sda.coding", "sda.finance.payroll", "sda.engineering.backend", "sda.global"],
        )
        self.scope_menu.grid(row=11, column=0, sticky="ew", padx=20, pady=(2, 12))
        self.scope_menu.set("sda.coding")

        # HitL Toggle
        self.hitl_switch = ctk.CTkSwitch(
            self.sidebar, text="Human-in-the-Loop", font=ctk.CTkFont(size=12, weight="bold")
        )
        self.hitl_switch.grid(row=12, column=0, sticky="w", padx=20, pady=5)
        self.hitl_switch.select()

        # Open Master MD Button
        self.open_md_btn = ctk.CTkButton(
            self.sidebar, text="Open Audit Markdown 📄", fg_color="gray30", hover_color="gray40",
            command=self._open_master_md
        )
        self.open_md_btn.grid(row=13, column=0, sticky="ew", padx=20, pady=(15, 5))

        # Restart GUI Button
        self.restart_btn = ctk.CTkButton(
            self.sidebar, text="Restart GUI 🔄", fg_color="#D84315", hover_color="#BF360C",
            command=self._restart_app
        )
        self.restart_btn.grid(row=14, column=0, sticky="ew", padx=20, pady=(5, 15))

        # Footer
        self.footer = ctk.CTkLabel(self.sidebar, text="RTX 3050 Laptop / INT4 AutoRound", font=ctk.CTkFont(size=10), text_color="gray50")
        self.footer.grid(row=15, column=0, padx=10, pady=15)

        # =====================================================================
        # MAIN CONTENT FRAME
        # =====================================================================
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(2, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Top Bar: Prompt Input & Security Status Badge
        self.top_card = ctk.CTkFrame(self.main_frame)
        self.top_card.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 10))
        self.top_card.grid_columnconfigure(0, weight=1)

        # Security Status Pill
        self.security_badge = ctk.CTkLabel(
            self.top_card, text="🛡️ PII Guard: Standby", font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#81C784"
        )
        self.security_badge.grid(row=0, column=0, sticky="w", padx=15, pady=(10, 2))

        # Prompt Entry Box
        self.prompt_entry = ctk.CTkTextbox(self.top_card, height=85, font=ctk.CTkFont(size=13))
        self.prompt_entry.grid(row=1, column=0, sticky="ew", padx=15, pady=(5, 10))
        self.prompt_entry.insert(
            "0.0",
            "Write a Python script to compute Fibonacci numbers up to 100 with memoization and print execution time."
        )

        # Run Button
        self.run_button = ctk.CTkButton(
            self.top_card, text="Execute SEAF Pipeline 🚀", font=ctk.CTkFont(size=14, weight="bold"),
            height=36, fg_color="#1E88E5", hover_color="#1565C0", command=self._start_agent_thread
        )
        self.run_button.grid(row=1, column=1, padx=15, pady=(5, 10), sticky="e")

        # Step-by-Step Progress Frame
        self.progress_frame = ctk.CTkFrame(self.main_frame)
        self.progress_frame.grid(row=1, column=0, sticky="ew", padx=0, pady=(0, 10))
        self.progress_frame.grid_columnconfigure(0, weight=1)

        self.step_label = ctk.CTkLabel(
            self.progress_frame, text="Pipeline Ready", font=ctk.CTkFont(size=12)
        )
        self.step_label.grid(row=0, column=0, sticky="w", padx=15, pady=(6, 2))

        self.step_progress = ctk.CTkProgressBar(self.progress_frame, height=8)
        self.step_progress.grid(row=1, column=0, sticky="ew", padx=15, pady=(2, 8))
        self.step_progress.set(0)

        # Tabview for Output & Inspections
        self.tabs = ctk.CTkTabview(self.main_frame)
        self.tabs.grid(row=2, column=0, sticky="nsew", padx=0, pady=0)

        self.tab_logs = self.tabs.add("📜 Agent Logs")
        self.tab_code = self.tabs.add("💻 Generated Code")
        self.tab_critic = self.tabs.add("🔍 Critic Audit")
        self.tab_memory = self.tabs.add("🗄️ Scoped Memory")

        # Tab 1: Logs
        self.tab_logs.grid_columnconfigure(0, weight=1)
        self.tab_logs.grid_rowconfigure(0, weight=1)
        self.log_box = ctk.CTkTextbox(self.tab_logs, font=ctk.CTkFont(family="Consolas", size=12))
        self.log_box.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.log_box.configure(state="disabled")

        # Tab 2: Code
        self.tab_code.grid_columnconfigure(0, weight=1)
        self.tab_code.grid_rowconfigure(0, weight=1)
        self.code_box = ctk.CTkTextbox(self.tab_code, font=ctk.CTkFont(family="Consolas", size=13))
        self.code_box.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.code_box.configure(state="disabled")

        # Tab 3: Critic Audit
        self.tab_critic.grid_columnconfigure(0, weight=1)
        self.tab_critic.grid_rowconfigure(0, weight=1)
        self.critic_box = ctk.CTkTextbox(self.tab_critic, font=ctk.CTkFont(family="Consolas", size=12))
        self.critic_box.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.critic_box.configure(state="disabled")

        # Tab 4: Scoped Memory
        self.tab_memory.grid_columnconfigure(0, weight=1)
        self.tab_memory.grid_rowconfigure(0, weight=1)
        self.memory_box = ctk.CTkTextbox(self.tab_memory, font=ctk.CTkFont(family="Consolas", size=12))
        self.memory_box.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.memory_box.configure(state="disabled")

    # =========================================================================
    # TELEMETRY POLLING (Background Thread)
    # =========================================================================
    def _start_telemetry_loop(self):
        def loop():
            while True:
                try:
                    s = telemetry.sample()
                    self.after(0, self._update_telemetry_ui, s)
                except Exception:
                    pass
                time.sleep(2.0)

        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def _update_telemetry_ui(self, s: dict):
        ram_pct = s["ram_percent"]
        self.ram_label.configure(text=f"Host RAM: {ram_pct:.1f}% ({s['ram_used_mb']} MB)")
        self.ram_bar.set(ram_pct / 100.0)

        if ram_pct >= 80.0:
            self.ram_bar.configure(progress_color="#FF5252")  # Red
            self.throttle_status.configure(
                text=f"Mode: Throttled ({THROTTLED_CONTEXT_WINDOW} ctx)", text_color="#FF5252"
            )
        else:
            self.ram_bar.configure(progress_color="#4CAF50")  # Green
            self.throttle_status.configure(
                text=f"Mode: Normal ({BASE_CONTEXT_WINDOW} ctx)", text_color="#4CAF50"
            )

        gpu_pct = s["gpu_percent"]
        self.gpu_label.configure(text=f"GPU VRAM: {gpu_pct:.1f}% ({s['gpu_used_mb']} MB)")
        self.gpu_bar.set(gpu_pct / 100.0)

    # =========================================================================
    # LOGGING & OUTPUT HELPERS
    # =========================================================================
    def _append_log(self, text: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _set_code(self, code: str):
        self.code_box.configure(state="normal")
        self.code_box.delete("0.0", "end")
        self.code_box.insert("0.0", code)
        self.code_box.configure(state="disabled")

    def _open_master_md(self):
        if MASTER_MD_PATH.exists():
            os.startfile(str(MASTER_MD_PATH))
        else:
            self._append_log("Markdown master audit trail has not been generated yet.")

    def _restart_app(self):
        log.info("Restarting GUI...")
        os.execl(sys.executable, sys.executable, *sys.argv)

    # =========================================================================
    # HITL CHECKPOINT MODAL
    # =========================================================================
    def _trigger_hitl_dialog(self):
        self.after(0, self._show_hitl_dialog)

    def _show_hitl_dialog(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("HitL Approval Checkpoint")
        dialog.geometry("540x260")
        dialog.attributes("-topmost", True)

        ctk.CTkLabel(
            dialog, text="⚠️ Code Execution Approval Required", font=ctk.CTkFont(size=16, weight="bold")
        ).pack(padx=20, pady=(20, 10))

        ctk.CTkLabel(
            dialog,
            text="The agent has generated code and is requesting permission\nto execute it in the local sandbox.",
            font=ctk.CTkFont(size=12),
        ).pack(padx=20, pady=5)

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=25)

        def approve():
            self.hitl_approved_var = True
            if self.hitl_event and hasattr(self, 'agent_loop'):
                self.agent_loop.call_soon_threadsafe(self.hitl_event.set)
            dialog.destroy()

        def reject():
            self.hitl_approved_var = False
            if self.hitl_event and hasattr(self, 'agent_loop'):
                self.agent_loop.call_soon_threadsafe(self.hitl_event.set)
            dialog.destroy()

        ctk.CTkButton(btn_frame, text="Reject ❌", fg_color="#D32F2F", hover_color="#B71C1C", command=reject).pack(
            side="left", padx=10
        )
        ctk.CTkButton(btn_frame, text="Approve & Execute ✅", fg_color="#388E3C", hover_color="#2E7D32", command=approve).pack(
            side="left", padx=10
        )

    # =========================================================================
    # PIPELINE EXECUTION
    # =========================================================================
    def _start_agent_thread(self):
        if self.is_running:
            return

        prompt = self.prompt_entry.get("0.0", "end").strip()
        if not prompt:
            return

        self.is_running = True
        self.run_button.configure(state="disabled", text="Running...")
        self.step_progress.set(0.1)
        self.step_label.configure(text="Screening PII & Inspecting Payload...")

        # Pre-flight PII check
        pii_res = gateway.inspect_payload(prompt)
        if pii_res["has_pii"]:
            self.security_badge.configure(
                text=f"🚨 PII Detected ({list(pii_res['detected_entities'].keys())}) — Locked to Local Edge",
                text_color="#FF5252"
            )
        else:
            self.security_badge.configure(
                text="🛡️ PII Guard: 100% Clean — Edge Sovereign",
                text_color="#81C784"
            )

        self._append_log(f"\n{'='*60}\n[SEAF Pipeline Initiated: {time.strftime('%X')}]\nPrompt: {prompt}")

        t = threading.Thread(target=self._run_pipeline_async, args=(prompt,), daemon=True)
        t.start()

    def _run_pipeline_async(self, prompt: str):
        loop = asyncio.new_event_loop()
        self.agent_loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._execute_pipeline(prompt))
        except Exception as exc:
            log.exception("Pipeline execution failed")
            self.after(0, self._append_log, f"\n[PIPELINE ERROR] {exc}")
            self.after(0, lambda: self.run_button.configure(state="normal", text="Execute SEAF Pipeline 🚀"))
            self.after(0, lambda: self.step_label.configure(text=f"Error: {exc}"))
        finally:
            loop.close()

    async def _execute_pipeline(self, prompt: str):
        if not self.graph:
            self.graph = build_full_graph()

        self.hitl_event = asyncio.Event()
        self.hitl_approved_var = False
        target_scope = self.scope_menu.get()

        initial_state: CriticState = {
            "messages": [HumanMessage(content=prompt)],
            "raw_prompt": prompt,
            "target_scope": target_scope,
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
            "hitl_enabled": bool(self.hitl_switch.get()),
            "hitl_approved": True,
            "hitl_event": self.hitl_event,
            "hitl_callback": self._trigger_hitl_dialog,
        }

        node_progress = {
            "pii_and_telemetry": ("1/6 PII Guard & Telemetry check...", 0.2),
            "rag_retrieval": ("2/6 RAG & Scoped Memory retrieval...", 0.35),
            "caveman_compress": ("3/6 Caveman token compression...", 0.45),
            "reasoner_generate": ("4/6 Local SLM code generation...", 0.60),
            "ponytail_review": ("5/6 Ponytail AST review...", 0.70),
            "sandbox_execute": ("6/6 Sandbox execution...", 0.85),
            "critic_evaluate": ("Evaluating outcome & Critic review...", 0.95),
            "patcher": ("Auto-patching detected errors...", 0.75),
            "log_success": ("Synchronizing Scoped Memory & Audit...", 1.0),
        }

        current_state = dict(initial_state)
        async for event in self.graph.astream(initial_state):
            for node, state in event.items():
                if state and isinstance(state, dict):
                    current_state.update(state)
                    
                if node in node_progress:
                    label, prog = node_progress[node]
                    self.after(0, self._update_step, label, prog)
                self.after(0, self._append_log, f"[*] Node completed: {node}")
                
                # Real-time UI Tab Updates
                if "generated_code" in current_state and current_state["generated_code"]:
                    self.after(0, self._set_code, current_state["generated_code"])
                if node in ("critic_evaluate", "patcher", "sandbox_execute"):
                    self.after(0, self._update_critic_box, dict(current_state))
                if node in ("rag_retrieval", "log_success"):
                    self.after(0, self._update_memory_box, dict(current_state))

        self.after(0, self._on_pipeline_done, current_state)

    def _update_step(self, label: str, val: float):
        self.step_label.configure(text=label)
        self.step_progress.set(val)

    def _update_critic_box(self, state: dict):
        self.critic_box.configure(state="normal")
        self.critic_box.delete("0.0", "end")
        verdict = state.get("critic_verdict", "N/A")
        attempts = state.get("exec_attempts", 0)
        cycles = state.get("cycle_log", [])
        self.critic_box.insert("0.0", f"CRITIC VERDICT : {verdict}\nATTEMPTS       : {attempts}\n\nEXECUTION CYCLES:\n")
        for cyc in cycles:
            self.critic_box.insert("end", f"- Attempt {cyc.get('attempt')}: Exit {cyc.get('exit_code')} ({cyc.get('duration_sec')}s)\n")
            if cyc.get("stderr"):
                self.critic_box.insert("end", f"  Stderr: {cyc.get('stderr')[:300]}\n")
        self.critic_box.configure(state="disabled")

    def _update_memory_box(self, state: dict):
        scope = state.get("target_scope", "sda.coding")
        frames = memory_engine.query_memory(scope)
        self.memory_box.configure(state="normal")
        self.memory_box.delete("0.0", "end")
        self.memory_box.insert("0.0", f"SCOPED MEMORY HIERARCHY: {scope}\nTotal Frames: {len(frames)}\n\n")
        for f in frames[-10:]:
            self.memory_box.insert("end", f"[{f['timestamp'][:19]}] [{f['scope']}] {f['key']} = {f['value']}\n")
        self.memory_box.configure(state="disabled")

    def _on_pipeline_done(self, state: CriticState):
        self.is_running = False
        self.run_button.configure(state="normal", text="Execute SEAF Pipeline 🚀")
        self.step_label.configure(text="Pipeline Completed Successfully ✅")
        self.step_progress.set(1.0)

        code = state.get("generated_code", "")
        self._set_code(code)
        self._update_critic_box(state)
        self._update_memory_box(state)

        self._append_log(f"\n[PIPELINE FINISHED] Verdict: {state.get('critic_verdict', 'N/A')} | Code lines: {len(code.splitlines())}")


if __name__ == "__main__":
    app = SEAFDesktopApp()
    app.mainloop()
