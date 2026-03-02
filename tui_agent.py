import json
import os
import shlex
import sys
import uuid
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Input, RichLog, Static

from src.agent import create_agent
from src.logging_utils import get_logger

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path(__file__).with_name("agent_config.json")
DB_PATH = APP_DIR / "agent_sessions.db"
VECTOR_PATH = APP_DIR / "message_history.vector"


class PromptInput(Input):
    def _on_focus(self, event):
        super()._on_focus(event)
        if self.value:
            self.action_select_all()


class AgentTUI(App):
    BINDINGS = [
        ("ctrl+r", "reload_config", "Reload config"),
        ("ctrl+n", "new_session", "New session"),
    ]
    CSS = """
    /* ── Global ──────────────────────────────────────────────────── */
    Screen {
        background: #0d1117;
        color: #cdd9e5;
    }

    #app {
        width: 100%;
        height: 100%;
        padding: 1 2;
        background: #0d1117;
    }

    /* ── Top bar ─────────────────────────────────────────────────── */
    #topbar {
        height: 3;
        padding: 0 2;
        margin-bottom: 1;
        background: #161b22;
        border: tall #30404f;
        align: center middle;
    }

    #brand {
        width: auto;
        color: #f0f6ff;
        text-style: bold;
    }

    #brand_dot {
        width: auto;
        color: #58a6ff;
        text-style: bold;
    }

    #tagline {
        width: 1fr;
        color: #546a80;
        margin-left: 2;
    }

    #runtime_state {
        width: 18;
        content-align: center middle;
    }

    /* ── Workspace ───────────────────────────────────────────────── */
    #workspace {
        height: 1fr;
    }

    /* ── Main column ─────────────────────────────────────────────── */
    #main_column {
        width: 5fr;
        height: 1fr;
    }

    #chat_panel {
        height: 1fr;
        padding: 1 2;
        background: #161b22;
        border: none;
    }

    #chat_header {
        height: auto;
        margin-bottom: 1;
        border-bottom: solid #30404f;
        padding-bottom: 1;
    }

    #chat_title {
        width: auto;
        color: #f0f6ff;
        text-style: bold;
    }

    #chat_meta {
        width: 1fr;
        margin-left: 2;
        color: #546a80;
    }

    #output {
        height: 1fr;
        border: none;
        background: transparent;
        color: #cdd9e5;
    }

    /* ── Composer ────────────────────────────────────────────────── */
    #composer {
        height: 4;
        margin-top: 1;
        background: #161b22;
        border: none;
        padding: 0 1;
    }

    #composer_prompt {
        width: auto;
        padding-top: 1;
        color: #58a6ff;
        text-style: bold;
        border-left: solid #58a6ff;
    }

    #input {
        border: none;
        background: transparent;
        color: #f0f6ff;
        padding-top: 1;
    }

    #input:focus {
        background: transparent;
        border: none;
    }

    /* ── Side column ─────────────────────────────────────────────── */
    #side_column {
        width: 28;
        min-width: 22;
        height: 1fr;
        margin-left: 1;
    }

    .rail_card {
        height: auto;
        min-height: 8;
        margin-bottom: 1;
        padding: 1 2;
        background: #161b22;
        border: tall #30404f;
        color: #cdd9e5;
    }

    /* ── Divider ─────────────────────────────────────────────────── */
    #divider {
        height: 1;
        background: #30404f;
        margin: 0;
    }
    """

    config = reactive({})
    agent = None
    runtime_state = reactive("READY")
    runtime_note = reactive("Ready for input")
    session_id = reactive("")
    _log = get_logger("agent.tui")

    def _new_session_id(self) -> str:
        return f"tui_{uuid.uuid4().hex[:8]}"

    def load_config(self):
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                self.config = json.load(f)
            config_overrides = {
                "max_iterations": self.config.get("max_iterations", 5),
                "temperature": self.config.get("temperature", 0.7),
                "max_tokens": self.config.get("max_tokens", 2048),
                "vector_path": str(VECTOR_PATH),
            }
            for key in (
                "history_limit",
                "history_recent_tail",
                "compact_summary_max_chars",
                "tool_result_max_chars",
                "assistant_ctx_max_chars",
                "trace_context_max_messages",
                "trace_context_max_chars",
                "tool_memory_items",
                "rag_enabled",
                "rag_top_k",
            ):
                if key in self.config:
                    config_overrides[key] = self.config[key]
            # Push Firecrawl settings into the environment so the skill
            # bootstrap (## Bootstrap: firecrawl_config) picks them up.
            if "firecrawl_url" in self.config:
                os.environ["FIRECRAWL_URL"] = self.config["firecrawl_url"]
            if self.config.get("firecrawl_api_key"):
                os.environ["FIRECRAWL_API_KEY"] = self.config["firecrawl_api_key"]

            # Forward MCP server config so the agent can initialise clients.
            if "mcp" in self.config:
                config_overrides["mcp_servers"] = self.config["mcp"]

            self.agent = create_agent(
                llm_url=self.config.get("endpoint", "http://localhost:8080"),
                use_chat_api=self.config.get("use_chat_api", True),
                chat_template=self.config.get("chat_template", "chatml"),
                db_path=str(DB_PATH),
                config_overrides=config_overrides,
            )
            self.runtime_state = "READY"
            self.runtime_note = "Loaded"
        except Exception as exc:
            self._log.exception("Failed to load TUI config from %s", CONFIG_PATH)
            self.config = {}
            self.agent = None
            self.runtime_state = "ERROR"
            self.runtime_note = f"Config error: {exc}"
        if self.is_mounted:
            self._refresh_panels()

    def compose(self) -> ComposeResult:
        yield Container(
            Horizontal(
                Static("FOREBLOCKS", id="brand"),
                Static(" // AGENT", id="brand_dot"),
                Static("local coding workspace", id="tagline"),
                Static("", id="runtime_state"),
                id="topbar",
            ),
            Horizontal(
                Vertical(
                    Container(
                        Horizontal(
                            Static("Conversation", id="chat_title"),
                            Static("session transcript", id="chat_meta"),
                            id="chat_header",
                        ),
                        RichLog(
                            id="output",
                            wrap=True,
                            highlight=False,
                            markup=False,
                        ),
                        id="chat_panel",
                    ),
                    Horizontal(
                        Static(" ", id="composer_prompt"),
                        PromptInput(
                            placeholder=" Send a prompt…",
                            id="input",
                        ),
                        id="composer",
                    ),
                    id="main_column",
                ),
                Vertical(
                    Static("", id="session_card", classes="rail_card"),
                    Static("", id="config_card", classes="rail_card"),
                    Static("", id="help_card", classes="rail_card"),
                    id="side_column",
                ),
                id="workspace",
            ),
            id="app",
        )

    def on_mount(self):
        self.session_id = self._new_session_id()
        self.load_config()
        self._refresh_panels()
        if self.agent:
            self._write_system_message(
                "Workspace ready — type a prompt and press Enter."
            )
        else:
            self._write_system_message(self.runtime_note)
        self.call_after_refresh(self._focus_input)

    def action_reload_config(self):
        self.load_config()
        if self.agent:
            self._write_system_message("Configuration reloaded.")
        else:
            self._write_system_message(self.runtime_note)
        self._focus_input()

    def action_new_session(self):
        self.session_id = self._new_session_id()
        if self.agent is not None:
            self.agent.detach_runtime_state()
        self.runtime_state = "READY"
        self.runtime_note = "New session"
        self._refresh_panels()
        self._write_system_message(f"New session started  ·  {self.session_id}")
        self._focus_input()

    def on_input_submitted(self, event):
        cmd = event.value.strip()
        if not cmd:
            return
        input_box = self.query_one("#input", Input)
        input_box.value = ""
        self._focus_input()
        if cmd.startswith("/"):
            self._write_user_message(cmd)
            self._handle_slash_command(cmd)
            return
        if not self.agent:
            self.load_config()
        self._write_user_message(cmd)
        self.runtime_state = "THINKING"
        self.runtime_note = "Running"
        self._refresh_panels()
        self.run_worker(partial(self._get_agent_response, cmd), thread=True)

    def _get_agent_response(self, cmd):
        if self.agent:
            try:
                response = self.agent.chat(
                    cmd,
                    session_id=self.session_id or self._new_session_id(),
                    verbose=False,
                )
                state = "READY"
                note = "Idle"
            except Exception as e:
                response = f"[Agent error] {e}"
                state = "ERROR"
                note = "Request failed"
        else:
            response = "[Agent not initialized]"
            state = "OFFLINE"
            note = "Unavailable"
        self.call_from_thread(self._show_agent_response, response, state, note)

    def _show_agent_response(self, response, state, note):
        self.runtime_state = state
        self.runtime_note = note
        self._write_agent_message(response)
        self._refresh_panels()

    def _focus_input(self):
        input_box = self.query_one("#input", Input)
        input_box.focus()
        if input_box.value:
            input_box.action_select_all()

    def _refresh_panels(self):
        self.query_one("#runtime_state", Static).update(
            Text(f" {self.runtime_state} ", style=self._runtime_chip_style())
        )
        self.query_one("#session_card", Static).update(self._build_session_card())
        self.query_one("#config_card", Static).update(self._build_config_card())
        self.query_one("#help_card", Static).update(self._build_help_card())

    def _runtime_chip_style(self):
        if self.runtime_state == "THINKING":
            return "bold #0d1117 on #ffd173"
        if self.runtime_state == "ERROR":
            return "bold #ffeaea on #f07178"
        if self.runtime_state == "OFFLINE":
            return "bold #c9d1d9 on #2d333b"
        return "bold #0d1117 on #58a6ff"

    # ── Card builders ──────────────────────────────────────────────

    def _build_session_card(self):
        agent_name = self.config.get("agent_name", "ForeblocksAgent")
        ctx_info = self._session_context_info()
        msg_count = str(ctx_info.get("persisted_messages", 0))
        history_window = f"{ctx_info.get('loaded_message_budget', 0)}/{ctx_info.get('history_limit', 0)}"
        return self._build_card(
            "SESSION",
            [
                ("Agent", agent_name),
                ("ID", self.session_id or "—"),
                ("Msgs", msg_count),
                ("Window", history_window),
                ("State", self.runtime_state),
                ("Note", self.runtime_note),
            ],
        )

    def _build_config_card(self):
        if not self.config:
            return self._build_card(
                "CONFIG",
                [
                    ("Status", "Unavailable"),
                    ("Path", str(CONFIG_PATH)),
                ],
            )
        return self._build_card(
            "CONFIG",
            [
                ("Endpoint", self.config.get("endpoint", "localhost:8080")),
                ("Template", self.config.get("chat_template", "chatml")),
                ("Iterations", str(self.config.get("max_iterations", 5))),
                (
                    "History",
                    (
                        f"{self.config.get('history_limit', 18)} / "
                        f"tail {self.config.get('history_recent_tail', 8)}"
                    ),
                ),
                ("DB", DB_PATH.name),
            ],
        )

    def _build_help_card(self):
        return self._build_card(
            "HELP",
            [
                ("Enter", "Send prompt"),
                ("Ctrl+R", "Reload config"),
                ("Ctrl+N", "New session"),
                ("/help", "Show commands"),
                ("/context", "Inspect context"),
                ("/compact", "Checkpoint history"),
            ],
        )

    def _build_card(self, title: str, rows: list) -> Text:
        card = Text()
        # Title with subtle decoration
        card.append(" ", style="#58a6ff")
        card.append(f"{title}\n", style="bold #dce8f5")
        card.append("\n")
        for index, (label, value) in enumerate(rows):
            card.append(f"{label}", style="#3d5068")
            card.append("  ")
            card.append(f"{value}", style="#c9d4e0")
            if index != len(rows) - 1:
                card.append("\n")
        return card

    # ── Message writers ────────────────────────────────────────────

    def _write_message(
        self, label: str, label_style: str, message: str, message_style: str
    ):
        line = Text()
        line.append(f" {label} ", style=label_style)
        line.append("  ")
        line.append(message, style=message_style)
        log = self.query_one("#output", RichLog)
        log.write(line)
        log.write("")

    def _write_system_message(self, message: str):
        self._write_message("SYS", "bold #8b949e on #21262d", message, "#6e7681")

    def _write_user_message(self, message: str):
        self._write_message("YOU", "bold #0d1117 on #ffd173", message, "#cdd9e5")

    def _write_agent_message(self, message: str):
        self._write_message("AI ", "bold #0d1117 on #58a6ff", message, "#adbac7")

    def _session_context_info(self):
        if not self.agent or not self.session_id:
            history_limit = int(self.config.get("history_limit", 18))
            return {
                "persisted_messages": 0,
                "loaded_message_budget": 0,
                "history_limit": history_limit,
            }
        try:
            return self.agent.describe_runtime_context(self.session_id)
        except Exception:
            self._log.exception("Failed to inspect session context")
            history_limit = int(self.config.get("history_limit", 18))
            return {
                "persisted_messages": 0,
                "loaded_message_budget": 0,
                "history_limit": history_limit,
            }

    def _format_context_report(self, info):
        runtime = info.get("runtime", {})
        lines = [
            f"Session: {info.get('session_id') or '—'}",
            (
                "Persisted messages: "
                f"{info.get('persisted_messages', 0)} "
                f"(normal load budget: {info.get('loaded_message_budget', 0)}/"
                f"{info.get('history_limit', 0)})"
            ),
        ]
        if info.get("history_over_budget"):
            lines.append(
                "History status: older turns exceed the normal load window; compact if you want a checkpoint."
            )
        else:
            lines.append(
                "History status: current session fits in the normal load window."
            )

        if runtime.get("loaded"):
            cols = ", ".join(runtime.get("value_columns", [])[:6]) or "none"
            lines.append(
                "Runtime data: "
                f"{runtime.get('data_name') or 'unnamed dataset'} "
                f"rows={runtime.get('row_count', 0)} cols={cols}"
            )
            if runtime.get("freq"):
                lines.append(f"Frequency: {runtime.get('freq')}")
        else:
            lines.append("Runtime data: no dataset loaded")

        anomaly_series = runtime.get("anomaly_series", 0)
        if anomaly_series:
            lines.append(
                "Anomalies in memory: "
                f"{anomaly_series} series / {runtime.get('anomaly_points', 0)} points"
            )
        forecast_model = runtime.get("forecast_model")
        if forecast_model:
            lines.append(f"Forecast memory: best model {forecast_model}")

        return "\n".join(lines)

    def _handle_slash_command(self, raw: str):
        try:
            parts = shlex.split(raw)
        except ValueError as exc:
            self.runtime_state = "ERROR"
            self.runtime_note = "Bad command"
            self._write_system_message(f"Command parse error: {exc}")
            self._refresh_panels()
            return

        if not parts:
            return

        command = parts[0].lower()
        args = parts[1:]

        if command == "/help":
            self.runtime_state = "READY"
            self.runtime_note = "Commands"
            self._write_system_message(
                "\n".join(
                    [
                        "Commands:",
                        "/help  show this list",
                        "/context  inspect session history and runtime state",
                        "/compact [keep_last]  summarize older session history into one checkpoint",
                        "/reset  clear runtime tool/data state but keep the transcript",
                        "/new  start a new session",
                        "/reload  reload agent_config.json",
                    ]
                )
            )
        elif command == "/context":
            if not self.agent:
                self.load_config()
            info = self._session_context_info()
            self.runtime_state = "READY"
            self.runtime_note = "Context"
            self._write_system_message(self._format_context_report(info))
        elif command == "/compact":
            if not self.agent:
                self.load_config()
            keep_last = None
            if args:
                try:
                    keep_last = max(1, int(args[0]))
                except ValueError:
                    self.runtime_state = "ERROR"
                    self.runtime_note = "Bad command"
                    self._write_system_message("Usage: /compact [keep_last_messages]")
                    self._refresh_panels()
                    return
            result = self.agent.compact_session(
                self.session_id,
                keep_last_messages=keep_last,
            )
            self.runtime_state = "READY"
            self.runtime_note = "Compacted"
            self._write_system_message(result.get("message", "Compaction finished."))
        elif command == "/reset":
            if not self.agent:
                self.load_config()
            if self.agent is not None:
                self.agent.reset_runtime_state(self.session_id)
            self.runtime_state = "READY"
            self.runtime_note = "Runtime reset"
            self._write_system_message(
                "Cleared runtime tool/data context. Transcript history is unchanged."
            )
        elif command == "/new":
            self.action_new_session()
            return
        elif command == "/reload":
            self.action_reload_config()
            return
        else:
            self.runtime_state = "ERROR"
            self.runtime_note = "Bad command"
            self._write_system_message(f"Unknown command: {command}. Use /help.")

        self._refresh_panels()
        self._focus_input()


if __name__ == "__main__":
    AgentTUI().run()
