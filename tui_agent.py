import datetime
import json
import os
import shlex
import sys
import uuid
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from rich.theme import Theme as RichTheme
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

    /* ── Pipeline indicator ──────────────────────────────────────── */
    #pipeline_bar {
        height: 1;
        background: #1f2a1a;
        color: #7ee787;
        padding: 0 2;
        display: none;
    }

    #pipeline_bar.active {
        display: block;
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

    /* ── Streaming preview ──────────────────────────────────────────── */
    #stream_out {
        height: auto;
        min-height: 0;
        padding: 0 4;
        background: #0d1117;
        color: #7ee787;
    }

    /* ── Divider ─────────────────────────────────────────────────── */
    #divider {
        height: 1;
        background: #30404f;
        margin: 0;
    }
    """

    RICH_THEME = RichTheme(
        {
            # ── Headings ────────────────────────────────────────────
            "markdown.h1": "bold #58a6ff",
            "markdown.h1.border": "#58a6ff",
            "markdown.h2": "bold #7ee787",
            "markdown.h2.border": "#30404f",
            "markdown.h3": "bold #ffa657",
            "markdown.h4": "bold #cdd9e5",
            "markdown.h5": "italic #cdd9e5",
            "markdown.h6": "italic #8b949e",
            # ── Emphasis ─────────────────────────────────────────────
            "markdown.bold": "bold #f0f6ff",
            "markdown.strong": "bold #f0f6ff",
            "markdown.italic": "italic #cdd9e5",
            "markdown.em": "italic #cdd9e5",
            "markdown.strike": "strike #8b949e",
            "markdown.s": "strike #8b949e",
            # ── Code ─────────────────────────────────────────────────
            "markdown.code": "bold #ffa657",
            "markdown.code_block": "#cdd9e5",
            "markdown.fence": "#cdd9e5",
            # ── Block quote ──────────────────────────────────────────
            "markdown.block_quote": "italic #8b949e",
            "markdown.block_quote_border": "#30404f",
            # ── Links ────────────────────────────────────────────────
            "markdown.link": "#58a6ff underline",
            "markdown.link_url": "dim #58a6ff",
            # ── Lists ────────────────────────────────────────────────
            "markdown.bullet": "bold #ffa657",
            "markdown.item": "#cdd9e5",
            "markdown.item.bullet": "bold #ffa657",
            # ── Rules / horizontal lines ─────────────────────────────
            "markdown.rule": "#30404f",
            "markdown.hr": "#30404f",
            # ── Body text ────────────────────────────────────────────
            "markdown.paragraph": "#cdd9e5",
            "markdown.text": "#cdd9e5",
        }
    )

    config = reactive({})
    agent = None
    runtime_state = reactive("READY")
    runtime_note = reactive("Ready for input")
    session_id = reactive("")
    _log = get_logger("agent.tui")

    # ── Multi-agent state ─────────────────────────────────────────
    # All loaded agents: {name: Agent}
    _agents: dict = {}
    # Per-agent session IDs: {name: session_id}
    _agent_sessions: dict = {}
    # Name of the currently active agent
    _active_agent_name: str = ""
    # Pipeline mode: None or {"a": name, "b": name, "rounds": int, "left": int}
    _pipeline_mode: dict | None = None

    # ── Streaming state ───────────────────────────────────────────
    # Accumulated tokens from on_token callback (written from worker thread)
    _stream_buf: str = ""
    _stream_agent_name: str = ""
    _stream_flush_at: int = 0
    _had_tool_call: bool = False  # set True by _write_tool_call; cleared per turn

    def _new_session_id(self) -> str:
        return f"tui_{uuid.uuid4().hex[:8]}"

    # ── Agent creation helpers ────────────────────────────────────

    def _build_overrides_from_cfg(self, cfg_block: dict) -> dict:
        """Turn a flat agent-config dict into create_agent config_overrides."""
        overrides = {
            "max_iterations": cfg_block.get(
                "max_iterations", self.config.get("max_iterations", 5)
            ),
            "temperature": cfg_block.get(
                "temperature", self.config.get("temperature", 0.7)
            ),
            "max_tokens": cfg_block.get(
                "max_tokens", self.config.get("max_tokens", 2048)
            ),
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
            val = cfg_block.get(key, self.config.get(key))
            if val is not None:
                overrides[key] = val
        mcp = cfg_block.get("mcp", self.config.get("mcp"))
        if mcp:
            overrides["mcp_servers"] = mcp
        thinking = cfg_block.get("thinking", self.config.get("thinking"))
        if thinking:
            overrides["thinking"] = thinking
        return overrides

    def _create_agent_from_cfg(self, cfg_block: dict) -> object:
        """Instantiate a single Agent from a (possibly merged) config block."""
        overrides = self._build_overrides_from_cfg(cfg_block)
        firecrawl_url = cfg_block.get("firecrawl_url", self.config.get("firecrawl_url"))
        firecrawl_key = cfg_block.get(
            "firecrawl_api_key", self.config.get("firecrawl_api_key")
        )
        if firecrawl_url:
            os.environ["FIRECRAWL_URL"] = firecrawl_url
        if firecrawl_key:
            os.environ["FIRECRAWL_API_KEY"] = firecrawl_key
        return create_agent(
            llm_url=cfg_block.get(
                "endpoint", self.config.get("endpoint", "http://localhost:8080")
            ),
            system_prompt=cfg_block.get("system_prompt"),
            use_chat_api=cfg_block.get(
                "use_chat_api", self.config.get("use_chat_api", True)
            ),
            chat_template=cfg_block.get(
                "chat_template", self.config.get("chat_template", "chatml")
            ),
            db_path=str(DB_PATH),
            config_overrides=overrides,
        )

    def _switch_active_agent(self, name: str) -> None:
        """Change the active agent; syncs self.agent and self.session_id."""
        self._active_agent_name = name
        self.agent = self._agents[name]
        if name not in self._agent_sessions:
            self._agent_sessions[name] = self._new_session_id()
        self.session_id = self._agent_sessions[name]

    def load_config(self):
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                self.config = json.load(f)

            self._agents = {}
            self._agent_sessions = {}
            self._pipeline_mode = None

            raw_agent_map: dict = self.config.get("agents", {})
            if raw_agent_map:
                # Multi-agent mode: each key is an agent name with its own cfg block.
                # Top-level config fields act as defaults.
                for name, acfg in raw_agent_map.items():
                    merged = {**self.config, **acfg}
                    self._agents[name] = self._create_agent_from_cfg(merged)
                    self._log.info("Loaded agent '%s'", name)
            else:
                # Single-agent mode (backwards compat)
                name = self.config.get("agent_name", "main")
                self._agents[name] = self._create_agent_from_cfg(self.config)

            first_name = next(iter(self._agents))
            self._switch_active_agent(first_name)

            # Summarise MCP connections across all agents
            all_mcp = [
                c.name
                for a in self._agents.values()
                for c in getattr(a, "_mcp_clients", [])
            ]
            self.runtime_state = "READY"
            note = f"Loaded {len(self._agents)} agent(s)"
            if all_mcp:
                note += f" | MCP: {', '.join(all_mcp)}"
            self.runtime_note = note
        except Exception as exc:
            self._log.exception("Failed to load TUI config from %s", CONFIG_PATH)
            self.config = {}
            self._agents = {}
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
                    Static("", id="stream_out"),
                    Static("", id="pipeline_bar"),
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
                    Static("", id="agents_card", classes="rail_card"),
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
        self.console.push_theme(self.RICH_THEME)  # ← inject here

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
        new_sid = self._new_session_id()
        name = self._active_agent_name
        self._agent_sessions[name] = new_sid
        self.session_id = new_sid
        if self.agent is not None:
            self.agent.detach_runtime_state()
        self._pipeline_mode = None
        self._update_pipeline_bar()
        self.runtime_state = "READY"
        self.runtime_note = "New session"
        self._refresh_panels()
        self._write_system_message(
            f"New session started  ·  agent={name}  ·  {self.session_id}"
        )
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
        if self._pipeline_mode:
            self.run_worker(partial(self._run_pipeline, cmd), thread=True)
        else:
            self.run_worker(
                partial(self._get_agent_response, cmd, self._active_agent_name),
                thread=True,
            )

    def _get_agent_response(self, cmd: str, agent_name: str):
        """Thread worker: call agent with streaming and emit the result."""
        # Write the label immediately so the user sees the agent is responding
        self.call_from_thread(self._begin_agent_stream, agent_name)
        agent = self._agents.get(agent_name)
        sid = self._agent_sessions.get(agent_name) or self._new_session_id()
        if agent:
            try:
                response = agent.chat(
                    cmd,
                    session_id=sid,
                    verbose=False,
                    use_semantic_retrieval=True,
                    retrieval_mode="hybrid",
                    stream=self._on_stream_token,
                    tool_callback=lambda name, args: self.call_from_thread(
                        self._write_tool_call, name, args
                    ),
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
        self.call_from_thread(self._finalize_stream, response, state, note, agent_name)

    def _show_agent_response(
        self, response: str, state: str, note: str, agent_name: str = "AI"
    ):
        self.runtime_state = state
        self.runtime_note = note
        self._write_agent_message(response, agent_name)
        self._refresh_panels()

    # ── Pipeline execution ────────────────────────────────────────

    def _run_pipeline(self, seed: str) -> None:
        """Thread worker: alternate between two agents for N rounds."""
        pm = self._pipeline_mode
        if not pm:
            return
        name_a = pm["a"]
        name_b = pm["b"]
        rounds = pm["rounds"]
        agent_a = self._agents.get(name_a)
        agent_b = self._agents.get(name_b)
        if not agent_a or not agent_b:
            self.call_from_thread(
                self._show_agent_response,
                f"[Pipeline error] agent '{name_a}' or '{name_b}' not found.",
                "ERROR",
                "Pipeline failed",
                "SYS",
            )
            return

        message = seed
        for i in range(rounds):
            # Agent A turn
            sid_a = self._agent_sessions.get(name_a) or self._new_session_id()
            try:
                resp_a = agent_a.chat(
                    message,
                    session_id=sid_a,
                    verbose=False,
                    use_semantic_retrieval=True,
                    retrieval_mode="hybrid",
                )
            except Exception as exc:
                resp_a = f"[{name_a} error] {exc}"
            self.call_from_thread(
                self._show_agent_response,
                resp_a,
                "THINKING",
                f"Pipeline {i + 1}/{rounds}",
                name_a,
            )

            # Agent B turn — receives A's response as input
            sid_b = self._agent_sessions.get(name_b) or self._new_session_id()
            try:
                resp_b = agent_b.chat(
                    resp_a,
                    session_id=sid_b,
                    verbose=False,
                    use_semantic_retrieval=True,
                    retrieval_mode="hybrid",
                )
            except Exception as exc:
                resp_b = f"[{name_b} error] {exc}"
            last_note = f"Pipeline {i + 1}/{rounds}"
            self.call_from_thread(
                self._show_agent_response, resp_b, "THINKING", last_note, name_b
            )

            message = resp_b  # feed B's response into next round's A

        self.call_from_thread(self._pipeline_finished)

    def _pipeline_finished(self) -> None:
        self.runtime_state = "READY"
        self.runtime_note = "Pipeline done"
        self._write_system_message(
            f"Pipeline complete: {self._pipeline_mode['a']} ↔ {self._pipeline_mode['b']}  "
            f"({self._pipeline_mode['rounds']} round(s))"
        )
        self._pipeline_mode = None
        self._update_pipeline_bar()
        self._refresh_panels()

    def _update_pipeline_bar(self) -> None:
        """Show/hide the pipeline status bar at the bottom of the chat."""
        try:
            bar = self.query_one("#pipeline_bar", Static)
            if self._pipeline_mode:
                pm = self._pipeline_mode
                bar.update(
                    f" ⟳ PIPELINE  {pm['a']} → {pm['b']}  ·  {pm['rounds']} round(s)  "
                    f"·  /pipeline stop to cancel"
                )
                bar.add_class("active")
            else:
                bar.update("")
                bar.remove_class("active")
        except Exception:
            pass

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
        self.query_one("#agents_card", Static).update(self._build_agents_card())
        self.query_one("#config_card", Static).update(self._build_config_card())
        self.query_one("#help_card", Static).update(self._build_help_card())

    def _runtime_chip_style(self):
        if self.runtime_state == "THINKING":
            return "bold #0d1117 on #ffd173"
        if self.runtime_state == "STREAMING":
            return "bold #0d1117 on #7ee787"
        if self.runtime_state == "ERROR":
            return "bold #ffeaea on #f07178"
        if self.runtime_state == "OFFLINE":
            return "bold #c9d1d9 on #2d333b"
        return "bold #0d1117 on #58a6ff"

    # ── Card builders ──────────────────────────────────────────────

    def _build_session_card(self):
        ctx_info = self._session_context_info()
        msg_count = str(ctx_info.get("persisted_messages", 0))
        history_window = f"{ctx_info.get('loaded_message_budget', 0)}/{ctx_info.get('history_limit', 0)}"
        pipeline_note = ""
        if self._pipeline_mode:
            pm = self._pipeline_mode
            pipeline_note = f"{pm['a']}↔{pm['b']} ×{pm['rounds']}"
        return self._build_card(
            "SESSION",
            [
                ("Active", self._active_agent_name or "—"),
                ("ID", self.session_id or "—"),
                ("Msgs", msg_count),
                ("Window", history_window),
                ("State", self.runtime_state),
                ("Note", self.runtime_note),
            ]
            + ([("Pipeline", pipeline_note)] if pipeline_note else []),
        )

    def _build_agents_card(self):
        rows = []
        for name in self._agents:
            marker = "▶" if name == self._active_agent_name else " "
            rows.append((marker, name))
        if not rows:
            rows = [("—", "no agents")]
        return self._build_card("AGENTS", rows)

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
                ("/agents", "List agents"),
                ("/agent", "Switch agent"),
                ("/pipeline", "Agent pipeline"),
                ("/context", "Inspect context"),
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

    def _write_tool_call(self, name: str, args: dict) -> None:
        """Main-thread: write a tool-invocation banner into the log."""
        log = self.query_one("#output", RichLog)
        from rich.text import Text

        # ── header row: TOOL badge + tool name ──────────────────────
        header = Text()
        header.append(" TOOL ", style="bold #0d1117 on #ffa657")
        header.append("  ", style="")
        header.append(name, style="bold #ffa657")
        log.write(header)
        # ── argument rows (up to 4 most informative args) ───────────
        if args:
            _SKIP = {"output_path", "self"}
            shown = [(k, v) for k, v in args.items() if k not in _SKIP][:4]
            for k, v in shown:
                val_str = str(v)
                # Truncate long values (paths / JSON strings)
                if len(val_str) > 72:
                    val_str = val_str[:69] + "…"
                row = Text()
                row.append("        ", style="")
                row.append(f"{k}", style="#8b949e")
                row.append(" = ", style="#6e7681")
                row.append(val_str, style="#e6edf3")
                log.write(row)
        log.write("")
        self._had_tool_call = True

    def _write_user_message(self, message: str):
        self._write_message("YOU", "bold #0d1117 on #ffd173", message, "#cdd9e5")

    # Agent label colours — cycle through these for multi-agent
    _AGENT_COLOURS = [
        "#58a6ff",  # blue
        "#7ee787",  # green
        "#ffa657",  # orange
        "#e06c75",  # red
        "#c678dd",  # purple
        "#56b6c2",  # cyan
    ]

    def _agent_colour(self, name: str) -> str:
        names = list(self._agents.keys())
        try:
            idx = names.index(name)
        except ValueError:
            idx = 0
        return self._AGENT_COLOURS[idx % len(self._AGENT_COLOURS)]

    def _write_agent_message(self, message: str, agent_name: str = "AI"):
        log = self.query_one("#output", RichLog)
        colour = self._agent_colour(agent_name)
        label_line = Text()
        label = f" {agent_name[:6].upper():<6} "
        label_line.append(label, style=f"bold #0d1117 on {colour}")
        log.write(label_line)
        log.write(RichMarkdown(message, code_theme="github-dark"))
        log.write("")

    # ── Streaming helpers ─────────────────────────────────────────

    def _begin_agent_stream(self, agent_name: str) -> None:
        """Main-thread: write agent label and prime the stream display."""
        self._stream_buf = ""
        self._stream_flush_at = 0
        self._had_tool_call = False
        self._stream_agent_name = agent_name
        log = self.query_one("#output", RichLog)
        colour = self._agent_colour(agent_name)
        label_line = Text()
        label = f" {agent_name[:6].upper():<6} "
        label_line.append(label, style=f"bold #0d1117 on {colour}")
        log.write(label_line)
        self.runtime_state = "STREAMING"
        self.runtime_note = "…"
        self._refresh_panels()

    def _on_stream_token(self, token: str) -> None:
        """Worker-thread callback: accumulate token and throttle UI flushes."""
        self._stream_buf += token
        buf_len = len(self._stream_buf)
        if "\n" in token or buf_len - self._stream_flush_at >= 40:
            self._stream_flush_at = buf_len
            self.call_from_thread(self._refresh_stream_display)

    def _refresh_stream_display(self) -> None:
        """Main-thread: update the streaming preview widget."""
        try:
            self.query_one("#stream_out", Static).update(self._stream_buf)
            self.runtime_note = f"Streaming… {len(self._stream_buf)} chars"
        except Exception:
            pass

    def _finalize_stream(
        self, response: str, state: str, note: str, agent_name: str
    ) -> None:
        """Main-thread: clear stream preview and render final RichMarkdown."""
        try:
            self.query_one("#stream_out", Static).update("")
        except Exception:
            pass
        self._stream_buf = ""
        self._stream_flush_at = 0
        log = self.query_one("#output", RichLog)
        if self._had_tool_call:
            log.write("")
        log.write(RichMarkdown(response, code_theme="github-dark"))
        log.write("")
        self.runtime_state = state
        self.runtime_note = note
        self._refresh_panels()

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
                        "/help                         show this list",
                        "/agents                       list all loaded agents",
                        "/agent <name>                 switch active agent",
                        "/pipeline <a> <b> [rounds]    start inter-agent pipeline (next msg is seed)",
                        "/pipeline stop                cancel active pipeline",
                        "/context                      inspect session history and runtime state",
                        "/lastcontext                  dump the last LLM context window",
                        "/compact [keep_last]          summarize older session history",
                        "/reset                        clear runtime tool/data state",
                        "/new                          start a new session",
                        "/reload                       reload agent_config.json",
                        "/sessions                     list stored sessions",
                        "/load <session_id>            resume a previous session (prefix OK)",
                        "/export [path]                export session transcript to markdown",
                    ]
                )
            )
        elif command == "/agents":
            if not self._agents:
                self._write_system_message("No agents loaded. Use /reload.")
            else:
                lines = ["Loaded agents:"]
                for name, ag in self._agents.items():
                    active_tag = " ◀ active" if name == self._active_agent_name else ""
                    mcp_names = [c.name for c in getattr(ag, "_mcp_clients", [])]
                    mcp_tag = f"  MCP: {', '.join(mcp_names)}" if mcp_names else ""
                    lines.append(f"  {name}{active_tag}{mcp_tag}")
                self._write_system_message("\n".join(lines))
            self.runtime_state = "READY"
            self.runtime_note = "Agents listed"
        elif command == "/agent":
            if not args:
                self._write_system_message(
                    f"Active agent: {self._active_agent_name or '(none)'}  ·  use /agents to list"
                )
            elif args[0] not in self._agents:
                known = ", ".join(self._agents.keys()) or "none"
                self._write_system_message(f"Unknown agent '{args[0]}'. Known: {known}")
            else:
                self._switch_active_agent(args[0])
                self._write_system_message(
                    f"Switched to agent '{args[0]}'  ·  session {self.session_id}"
                )
            self.runtime_state = "READY"
            self.runtime_note = f"Agent: {self._active_agent_name}"
        elif command == "/pipeline":
            if args and args[0].lower() == "stop":
                self._pipeline_mode = None
                self._update_pipeline_bar()
                self._write_system_message("Pipeline cancelled.")
                self.runtime_state = "READY"
                self.runtime_note = "Idle"
            else:
                if len(args) < 2:
                    self._write_system_message(
                        "Usage: /pipeline <agent_a> <agent_b> [rounds=3]\n"
                        "Then send your seed message. Use /pipeline stop to cancel."
                    )
                elif args[0] not in self._agents or args[1] not in self._agents:
                    known = ", ".join(self._agents.keys()) or "none"
                    self._write_system_message(f"Unknown agent name(s). Known: {known}")
                else:
                    rounds = 3
                    if len(args) >= 3:
                        try:
                            rounds = max(1, int(args[2]))
                        except ValueError:
                            self._write_system_message("rounds must be an integer.")
                            self._refresh_panels()
                            self._focus_input()
                            return
                    self._pipeline_mode = {"a": args[0], "b": args[1], "rounds": rounds}
                    self._update_pipeline_bar()
                    self._write_system_message(
                        f"Pipeline ready: {args[0]} → {args[1]}  ×{rounds} round(s).\n"
                        f"Send your seed message to start."
                    )
                    self.runtime_state = "READY"
                    self.runtime_note = f"Pipeline armed: {args[0]}↔{args[1]}"
        elif command == "/context":
            if not self.agent:
                self.load_config()
            info = self._session_context_info()
            self.runtime_state = "READY"
            self.runtime_note = "Context"
            self._write_system_message(self._format_context_report(info))
        elif command == "/lastcontext":
            if not self.agent:
                self.load_config()
            text = (
                self.agent.get_last_context_text()
                if self.agent
                else "_Agent not initialised._"
            )
            self.runtime_state = "READY"
            self.runtime_note = "Last context"
            log = self.query_one("#output", RichLog)
            label_line = Text()
            label_line.append(" SYS ", style="bold #8b949e on #21262d")
            log.write(label_line)
            log.write(RichMarkdown(text))
            log.write("")
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
        elif command == "/sessions":
            if not self.agent:
                self.load_config()
            sessions = self.agent.list_sessions() if self.agent else []
            if not sessions:
                self._write_system_message("No sessions found.")
            else:
                active = getattr(self, "session_id", None)
                lines = [f"Stored sessions ({len(sessions)})  [most recent first]:"]
                for sid, last_ts in sessions:
                    short = sid[:20] + "..." if len(sid) > 20 else sid
                    tag = "  ◄ active" if sid == active else ""
                    lines.append(f"  {short}  last={last_ts}{tag}")
                self._write_system_message("\n".join(lines))
            self.runtime_state = "READY"
            self.runtime_note = "Sessions listed"
        elif command == "/load":
            if not args:
                self._write_system_message(
                    "Usage: /load <session_id>  (prefix matching supported)"
                )
            else:
                if not self.agent:
                    self.load_config()
                target = args[0]
                sessions = self.agent.list_sessions() if self.agent else []
                matched = [
                    s for s, _ in sessions if s == target or s.startswith(target)
                ]
                if not matched:
                    self._write_system_message(
                        f"No session matching '{target}'. Use /sessions to list."
                    )
                elif len(matched) > 1:
                    self._write_system_message(
                        f"Ambiguous prefix '{target}' matches {len(matched)} sessions. "
                        "Be more specific."
                    )
                else:
                    sid = matched[0]
                    self.session_id = sid
                    if self.agent:
                        self.agent._activate_session_runtime(sid)
                    n = (
                        self.agent.memory.count_session_messages(sid)
                        if self.agent
                        else 0
                    )
                    self._write_system_message(
                        f"Loaded session {sid[:20]}...\nFull ID : {sid}\nMessages: {n}"
                    )
            self.runtime_state = "READY"
            self.runtime_note = "Session loaded"
        elif command == "/export":
            if not self.agent:
                self.load_config()
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            sid_short = (self.session_id or "nosid")[:8]
            default_path = f"export_{sid_short}_{ts}.md"
            out_path = args[0] if args else default_path
            try:
                messages = (
                    self.agent.memory.get_session_messages(self.session_id)
                    if self.agent and self.session_id
                    else []
                )
                lines: list[str] = [
                    "# Session Export",
                    "",
                    f"**Session ID:** `{self.session_id}`  ",
                    f"**Exported:** {datetime.datetime.now().isoformat()}  ",
                    f"**Messages:** {len(messages)}",
                    "",
                    "---",
                    "",
                ]
                for msg in messages:
                    role = msg.role.value.upper()
                    name_tag = f" `{msg.name}`" if msg.name else ""
                    lines.append(f"### {role}{name_tag}")
                    lines.append("")
                    lines.append(msg.content or "")
                    lines.append("")
                    lines.append("---")
                    lines.append("")
                Path(out_path).write_text("\n".join(lines), encoding="utf-8")
                self._write_system_message(
                    f"Exported {len(messages)} messages to {out_path}"
                )
            except Exception as exc:
                self._write_system_message(f"Export failed: {exc}")
            self.runtime_state = "READY"
            self.runtime_note = "Exported"
        else:
            self.runtime_state = "ERROR"
            self.runtime_note = "Bad command"
            self._write_system_message(f"Unknown command: {command}. Use /help.")

        self._refresh_panels()
        self._focus_input()


if __name__ == "__main__":
    AgentTUI().run()
