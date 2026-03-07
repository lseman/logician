use chrono::Local;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use serde_json::Value;
use std::fmt;
use uuid::Uuid;

use crate::bridge::{BridgeEvent, BridgeState};
use crate::markdown::{render_markdown, render_streaming};

// ── Types ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    System,
    Trace,
    Tool,
    Skill,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Ready,
    Thinking,
    Bubbling,
    Jambering,
    Streaming,
    Error,
}

impl Phase {
    pub fn color(self) -> Color {
        match self {
            Phase::Ready => Color::Green,
            Phase::Thinking => Color::Yellow,
            Phase::Bubbling => Color::LightYellow,
            Phase::Jambering => Color::Magenta,
            Phase::Streaming => Color::Cyan,
            Phase::Error => Color::Red,
        }
    }
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "thinking" => Phase::Thinking,
            "bubbling" | "bubblering" => Phase::Bubbling,
            "jambering" | "hjambering" => Phase::Jambering,
            "streaming" => Phase::Streaming,
            "error" => Phase::Error,
            _ => Phase::Ready,
        }
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Phase::Ready => write!(f, "ready"),
            Phase::Thinking => write!(f, "thinking"),
            Phase::Bubbling => write!(f, "bubbling"),
            Phase::Jambering => write!(f, "jambering"),
            Phase::Streaming => write!(f, "streaming"),
            Phase::Error => write!(f, "error"),
        }
    }
}

// ── Message ───────────────────────────────────────────────────────────────────

pub struct Message {
    #[allow(dead_code)]
    pub id: String,
    pub role: Role,
    pub is_streaming: bool,
    #[allow(dead_code)]
    pub text: String,
    pub raw_stream: Option<String>,
    /// Pre-rendered lines (updated on text change)
    pub rendered: Vec<Line<'static>>,
}

trait MessageRenderer {
    fn render_header(&self, role: Role, streaming: bool) -> Line<'static>;
    fn render_body(&self, role: Role, text: &str, streaming: bool) -> Vec<Line<'static>>;

    fn render(&self, role: Role, text: &str, streaming: bool) -> Vec<Line<'static>> {
        let mut lines = vec![self.render_header(role, streaming)];
        lines.extend(self.render_body(role, text, streaming));
        lines.push(Line::raw(""));
        lines
    }
}

struct DefaultRenderer;

impl MessageRenderer for DefaultRenderer {
    fn render_header(&self, role: Role, streaming: bool) -> Line<'static> {
        if role == Role::Assistant && streaming {
            return Line::from(vec![
                Span::styled(
                    "assistant    ",
                    Style::default()
                        .fg(Color::Green)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "▸ streaming",
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::DIM),
                ),
            ]);
        }
        let now = Local::now().format("%H:%M:%S").to_string();
        let (label, color) = match role {
            Role::User => ("you          ", Color::Yellow),
            Role::Assistant => ("assistant    ", Color::Green),
            Role::System => ("system       ", Color::Blue),
            Role::Trace => ("trace        ", Color::DarkGray),
            Role::Tool => ("tool ⚙       ", Color::Cyan),
            Role::Skill => ("skill 🧠     ", Color::Magenta),
        };
        Line::from(vec![
            Span::styled(
                label,
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                now,
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::DIM),
            ),
        ])
    }

    fn render_body(&self, role: Role, text: &str, streaming: bool) -> Vec<Line<'static>> {
        match role {
            Role::Assistant if streaming => render_streaming(text),
            Role::Assistant => render_markdown(text),
            Role::System => render_markdown(text),
            Role::User => text
                .lines()
                .map(|l| {
                    Line::from(Span::styled(
                        l.to_string(),
                        Style::default().fg(Color::White),
                    ))
                })
                .collect(),
            Role::Trace => {
                vec![Line::from(Span::styled(
                    text.to_string(),
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::DIM),
                ))]
            }
            Role::Tool => {
                let mut tool_lines = Vec::new();
                let mut it = text.lines();
                if let Some(name) = it.next() {
                    tool_lines.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled(
                            name.to_string(),
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ),
                    ]));
                }
                for l in it {
                    tool_lines.push(Line::from(vec![
                        Span::styled("  ", Style::default()),
                        Span::styled(l.to_string(), Style::default().fg(Color::Cyan)),
                    ]));
                }
                tool_lines
            }
            Role::Skill => text
                .lines()
                .map(|l| {
                    Line::from(vec![
                        Span::raw("  "),
                        Span::styled(l.to_string(), Style::default().fg(Color::Magenta)),
                    ])
                })
                .collect(),
        }
    }
}

impl Message {
    fn new(role: Role, text: impl Into<String>) -> Self {
        Self::new_with_streaming(role, text, false)
    }

    fn new_streaming(role: Role) -> Self {
        Self::new_with_streaming(role, String::new(), true)
    }

    fn new_with_streaming(role: Role, text: impl Into<String>, is_streaming: bool) -> Self {
        let text = text.into();
        let renderer = DefaultRenderer;
        let rendered = renderer.render(role, &text, is_streaming);
        Self {
            id: Uuid::new_v4().to_string(),
            role,
            is_streaming,
            text,
            raw_stream: None,
            rendered,
        }
    }

    fn append_stream_chunk(&mut self, chunk: &str) {
        self.text.push_str(chunk);
        self.is_streaming = true;
        let renderer = DefaultRenderer;
        self.rendered = renderer.render(self.role, &self.text, true);
    }

    fn finalize_streaming(&mut self) {
        self.is_streaming = false;
        let renderer = DefaultRenderer;
        self.rendered = renderer.render(self.role, &self.text, false);
    }

    fn rendered_for_raw_mode(&self) -> Vec<Line<'static>> {
        if self.role == Role::Assistant {
            if let Some(raw) = &self.raw_stream {
                let renderer = DefaultRenderer;
                return renderer.render(self.role, raw, true);
            }
        }
        self.rendered.clone()
    }
}

// ── Key action returned to main loop ─────────────────────────────────────────

pub enum KeyAction {
    None,
    Quit,
    /// Text that needs to go to the bridge (non-slash chat or slash)
    Submit(String),
    /// Toggle trace output
    ToggleTrace,
    /// Toggle raw stream display
    ToggleRawStream,
}

#[derive(Clone, Copy)]
pub struct SlashPopupEntry {
    pub command: &'static str,
    pub description: &'static str,
}

#[derive(Clone, Copy)]
struct SlashCommandSpec {
    command: &'static str,
    description: &'static str,
}

const SLASH_POPUP_LIMIT: usize = 8;

const SLASH_COMMANDS: [SlashCommandSpec; 22] = [
    SlashCommandSpec {
        command: "/help",
        description: "Show command list",
    },
    SlashCommandSpec {
        command: "/?",
        description: "Alias for /help",
    },
    SlashCommandSpec {
        command: "/status",
        description: "Show runtime state snapshot",
    },
    SlashCommandSpec {
        command: "/changes",
        description: "Show git status and diff preview",
    },
    SlashCommandSpec {
        command: "/doctor",
        description: "Run local health checks",
    },
    SlashCommandSpec {
        command: "/bug",
        description: "Save reproducible bug report file",
    },
    SlashCommandSpec {
        command: "/trace",
        description: "Toggle trace messages",
    },
    SlashCommandSpec {
        command: "/clear",
        description: "Clear visible transcript only",
    },
    SlashCommandSpec {
        command: "/agents",
        description: "List loaded agents",
    },
    SlashCommandSpec {
        command: "/agent",
        description: "Switch active agent",
    },
    SlashCommandSpec {
        command: "/pipeline",
        description: "Set or stop inter-agent pipeline",
    },
    SlashCommandSpec {
        command: "/context",
        description: "Show session/data context",
    },
    SlashCommandSpec {
        command: "/sessions",
        description: "List previous sessions",
    },
    SlashCommandSpec {
        command: "/load",
        description: "Load a previous session",
    },
    SlashCommandSpec {
        command: "/export",
        description: "Export chat history",
    },
    SlashCommandSpec {
        command: "/upload",
        description: "Ingest one document into RAG",
    },
    SlashCommandSpec {
        command: "/upload-dir",
        description: "Bulk ingest docs into RAG",
    },
    SlashCommandSpec {
        command: "/docs",
        description: "Fetch Context7 library docs",
    },
    SlashCommandSpec {
        command: "/new",
        description: "Start a new session",
    },
    SlashCommandSpec {
        command: "/reload",
        description: "Reload config and agents",
    },
    SlashCommandSpec {
        command: "/quit",
        description: "Exit CLI",
    },
    SlashCommandSpec {
        command: "/exit",
        description: "Alias for /quit",
    },
];

// ── App state ─────────────────────────────────────────────────────────────────

pub struct App {
    // Messages
    pub messages: Vec<Message>,
    // Cached flattened transcript lines (normal view / raw view).
    cached_lines: Vec<Line<'static>>,
    cached_lines_raw: Vec<Line<'static>>,
    // Live streaming assistant message
    pub live: Option<Message>,
    // Raw stream buffer (for Ctrl+P view)
    pub raw_buf: String,

    // Input
    pub input: String,
    pub input_cursor: usize, // char offset
    slash_selected: usize,

    // Scroll  (lines from top)
    pub scroll_top: u16,
    pub at_bottom: bool,
    pub last_area_h: u16, // last known messages viewport height

    // Phase / status
    pub phase: Phase,
    pub phase_note: String,
    pub busy: bool,
    pub spinner_tick: usize,

    // Feature toggles
    pub trace_on: bool,
    pub raw_on: bool, // show raw stream

    // Bridge state
    pub bridge_state: BridgeState,
    pub connected: bool,

    pub should_quit: bool,

    // Trace buffer (not rendered into messages when trace_on=false)
    pub trace_log: Vec<String>,
}

impl App {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            cached_lines: Vec::new(),
            cached_lines_raw: Vec::new(),
            live: None,
            raw_buf: String::new(),
            input: String::new(),
            input_cursor: 0,
            slash_selected: 0,
            scroll_top: 0,
            at_bottom: true,
            last_area_h: 40,
            phase: Phase::Ready,
            phase_note: "ready".into(),
            busy: false,
            spinner_tick: 0,
            trace_on: false,
            raw_on: false,
            bridge_state: BridgeState::default(),
            connected: false,
            should_quit: false,
            trace_log: Vec::new(),
        }
    }

    // ── Init ─────────────────────────────────────────────────────────────────

    pub fn handle_init(&mut self, v: Value) {
        let state = v.get("state").unwrap_or(&v);
        self.bridge_state = BridgeState::from_value(state);
        self.connected = true;
        let agents = if self.bridge_state.agents.is_empty() {
            "-".to_string()
        } else {
            self.bridge_state.agents.join(", ")
        };
        let text = format!(
            "# Logician CLI powered by `rust`\n\n\
**Agents**: {agents}  \n\
**Rapidfuzz**: {}  \n\n\
Active: `{}` · Session: `{}` · Ctrl+O trace · Ctrl+P raw stream · Ctrl+C quit",
            if self.bridge_state.rapidfuzz {
                "enabled"
            } else {
                "disabled"
            },
            self.bridge_state.active,
            &self.bridge_state.session[..self.bridge_state.session.len().min(24)],
        );
        self.add_message(Role::System, text);
    }

    // ── Message helpers ───────────────────────────────────────────────────────

    pub fn add_message(&mut self, role: Role, text: impl Into<String>) {
        self.push_message(Message::new(role, text));
        if self.at_bottom {
            self.scroll_to_bottom();
        }
    }

    pub fn add_system_message(&mut self, text: impl Into<String>) {
        self.add_message(Role::System, text);
    }

    fn trace(&mut self, text: impl Into<String>) {
        let line = format!("{}  {}", Local::now().format("%H:%M:%S"), text.into());
        self.trace_log.push(line.clone());
        if self.trace_on {
            self.push_message(Message::new(Role::Trace, line));
            if self.at_bottom {
                self.scroll_to_bottom();
            }
        }
    }

    fn push_message(&mut self, message: Message) {
        self.cached_lines.extend(message.rendered.iter().cloned());
        self.cached_lines_raw.extend(message.rendered_for_raw_mode());
        self.messages.push(message);
    }

    fn clear_transcript(&mut self) {
        self.messages.clear();
        self.cached_lines.clear();
        self.cached_lines_raw.clear();
        self.live = None;
        self.raw_buf.clear();
        self.scroll_top = 0;
        self.at_bottom = true;
    }

    fn active_cached_lines(&self) -> &[Line<'static>] {
        if self.raw_on {
            self.cached_lines_raw.as_slice()
        } else {
            self.cached_lines.as_slice()
        }
    }

    fn clamp_scroll(&mut self) {
        let total = self.total_rendered_lines();
        let max_s = total.saturating_sub(self.last_area_h);
        if self.scroll_top > max_s {
            self.scroll_top = max_s;
        }
        self.at_bottom = self.scroll_top >= max_s;
    }

    // ── Scroll helpers ────────────────────────────────────────────────────────

    pub fn total_rendered_lines(&self) -> u16 {
        let message_lines = self.active_cached_lines().len().min(u16::MAX as usize) as u16;
        let live_lines = self
            .live
            .as_ref()
            .map(|lm| lm.rendered.len().min(u16::MAX as usize) as u16)
            .unwrap_or(0);
        message_lines.saturating_add(live_lines)
    }

    pub fn visible_lines(&self, height: u16) -> Vec<Line<'static>> {
        if height == 0 {
            return Vec::new();
        }

        let cached = self.active_cached_lines();
        let cached_len = cached.len();
        let live_slice: &[Line<'static>] = self
            .live
            .as_ref()
            .map(|m| m.rendered.as_slice())
            .unwrap_or(&[]);
        let live_len = live_slice.len();
        let total = cached_len.saturating_add(live_len);
        if total == 0 {
            return Vec::new();
        }

        let max_start = total.saturating_sub(height as usize);
        let start = (self.scroll_top as usize).min(max_start);
        let end = start.saturating_add(height as usize).min(total);
        let mut out = Vec::with_capacity(end.saturating_sub(start));

        if start < cached_len {
            let cached_end = end.min(cached_len);
            out.extend(cached[start..cached_end].iter().cloned());
        }
        if end > cached_len {
            let live_start = start.saturating_sub(cached_len);
            let live_end = end.saturating_sub(cached_len);
            out.extend(live_slice[live_start..live_end].iter().cloned());
        }

        out
    }

    pub fn scroll_to_bottom(&mut self) {
        let total = self.total_rendered_lines();
        let visible = self.last_area_h;
        self.scroll_top = total.saturating_sub(visible);
        self.at_bottom = true;
    }

    pub fn scroll_up(&mut self, n: u16) {
        self.scroll_top = self.scroll_top.saturating_sub(n);
        self.at_bottom = false;
    }

    pub fn scroll_down(&mut self, n: u16) {
        let total = self.total_rendered_lines();
        let visible = self.last_area_h;
        let max_s = total.saturating_sub(visible);
        self.scroll_top = (self.scroll_top + n).min(max_s);
        if self.scroll_top >= max_s {
            self.at_bottom = true;
        }
    }

    pub fn update_area_h(&mut self, h: u16) {
        self.last_area_h = h;
        if self.at_bottom {
            self.scroll_to_bottom();
        } else {
            self.clamp_scroll();
        }
    }

    pub fn stop_live_message(&mut self) {
        self.flush_live_message(None);
    }

    fn flush_live_message(&mut self, fallback_text: Option<String>) {
        let mut live = match self.live.take() {
            Some(m) => m,
            None => {
                if let Some(text) = fallback_text {
                    if text.len() > 2 && text.chars().any(|c| c.is_alphanumeric()) {
                        self.add_message(Role::Assistant, text);
                    }
                }
                return;
            }
        };

        if live.text.is_empty() {
            if let Some(text) = fallback_text {
                live.text = text;
            }
        }
        if live.text.is_empty() {
            return;
        }

        live.finalize_streaming();
        live.raw_stream = Some(std::mem::take(&mut self.raw_buf));
        self.push_message(live);
        if self.at_bottom {
            self.scroll_to_bottom();
        } else {
            self.clamp_scroll();
        }
    }

    // ── Bridge event handling ─────────────────────────────────────────────────

    pub fn handle_bridge_event(&mut self, event: BridgeEvent) {
        match event {
            BridgeEvent::Token(tok) => {
                self.raw_buf.push_str(&tok);
                self.busy = true;
                self.phase = Phase::Streaming;
                self.phase_note = "streaming".into();
                match &mut self.live {
                    Some(lm) => lm.append_stream_chunk(&tok),
                    None => {
                        let mut lm = Message::new_streaming(Role::Assistant);
                        lm.append_stream_chunk(&tok);
                        self.live = Some(lm);
                    }
                }
                if self.at_bottom {
                    self.scroll_to_bottom();
                }
            }

            BridgeEvent::Phase { state, note } => {
                self.stop_live_message();
                self.phase = Phase::from_str(&state);
                self.phase_note = if note.is_empty() {
                    state.clone()
                } else {
                    note.clone()
                };
                self.trace(format!("phase={state} note={note}"));
            }

            BridgeEvent::Tool { name, args } => {
                self.stop_live_message();
                let args_pretty = serde_json::to_string_pretty(&args).unwrap_or_default();
                self.trace(format!("tool={name}"));
                self.add_message(Role::Tool, format!("{name}\n{args_pretty}"));
            }

            BridgeEvent::Skill {
                skill_ids,
                selected_tools,
            } => {
                self.stop_live_message();
                let skills = if skill_ids.is_empty() {
                    "none".to_string()
                } else {
                    skill_ids.join(", ")
                };
                let tools = if selected_tools.is_empty() {
                    "none".to_string()
                } else {
                    selected_tools.join(", ")
                };
                self.trace(format!("skills={skills} tools={tools}"));
                self.add_message(
                    Role::Skill,
                    format!("Activated skills: {skills}\nAvailable tools: {tools}"),
                );
            }

            BridgeEvent::Stderr(text) => {
                let short = &text[..text.len().min(220)];
                self.trace(format!("stderr={short}"));
            }

            BridgeEvent::Exit(code) => {
                self.phase = Phase::Error;
                self.phase_note = "bridge exited".into();
                self.connected = false;
                self.trace(format!(
                    "bridge_exit code={}",
                    code.map_or("?".to_string(), |c| c.to_string())
                ));
            }
        }
    }

    // ── Command result handling ───────────────────────────────────────────────

    pub fn handle_chat_result(&mut self, result: Result<Value, anyhow::Error>) {
        // Finalize live message
        match result {
            Ok(v) => {
                let fallback = v["assistant"].as_str().map(|s| s.to_string());
                self.flush_live_message(fallback);

                // Pipeline turns
                if v["pipeline"].as_bool().unwrap_or(false) {
                    if let Some(turns) = v["turns"].as_array() {
                        for t in turns {
                            let agent = t["agent"].as_str().unwrap_or("?");
                            let text = t["text"].as_str().unwrap_or("");
                            self.add_message(Role::Assistant, format!("[{agent}] {text}"));
                        }
                    }
                }

                if let Some(state) = v.get("state") {
                    self.bridge_state = BridgeState::from_value(state);
                }
            }
            Err(e) => {
                self.live = None;
                self.raw_buf.clear();
                self.phase = Phase::Error;
                self.phase_note = "chat failed".into();
                self.add_system_message(format!("Chat failed: {e}"));
            }
        }
        self.set_idle();
    }

    pub fn handle_slash_result(&mut self, result: Result<Value, anyhow::Error>) {
        match result {
            Ok(v) => {
                if let Some(msgs) = v["messages"].as_array() {
                    for m in msgs {
                        if let Some(s) = m.as_str() {
                            self.add_system_message(s);
                        }
                    }
                }
                if let Some(state) = v.get("state") {
                    self.bridge_state = BridgeState::from_value(state);
                }
                if v["exit"].as_bool().unwrap_or(false) {
                    self.should_quit = true;
                }
            }
            Err(e) => {
                self.phase = Phase::Error;
                self.phase_note = "slash failed".into();
                self.add_system_message(format!("Slash command failed: {e}"));
            }
        }
        self.set_idle();
    }

    fn set_idle(&mut self) {
        self.busy = false;
        self.phase = Phase::Ready;
        self.phase_note = "ready".into();
        self.raw_buf.clear();
        if self.at_bottom {
            self.scroll_to_bottom();
        }
    }

    // ── Local slash commands ──────────────────────────────────────────────────

    /// Returns true if command was handled locally, false if it should go to bridge.
    #[allow(dead_code)]
    pub fn handle_local_slash(&mut self, cmd: &str) -> bool {
        let lower = cmd.trim().to_lowercase();
        let lower = lower.as_str();

        if lower == "/help" || lower == "/?" {
            self.add_system_message(Self::help_text());
            return true;
        }

        if lower == "/clear" {
            self.clear_transcript();
            self.add_system_message("Transcript cleared. Session state is unchanged.");
            return true;
        }

        if lower == "/trace" || lower.starts_with("/trace ") {
            let arg = lower.strip_prefix("/trace").unwrap_or("").trim();
            let next = match arg {
                "on" | "1" | "true" | "yes" => true,
                "off" | "0" | "false" | "no" => false,
                "" => !self.trace_on,
                _ => {
                    self.add_system_message("Usage: /trace [on|off]");
                    return true;
                }
            };
            self.trace_on = next;
            self.add_system_message(format!(
                "Trace {}.",
                if next { "enabled" } else { "disabled" }
            ));
            return true;
        }

        if lower == "/status" {
            // Status will be handled in main loop (needs bridge call)
            return false;
        }

        false
    }

    #[allow(dead_code)]
    fn help_text() -> String {
        [
            "## Commands",
            "",
            "| Command | What it does |",
            "| --- | --- |",
            "| `/help` | Show this command list |",
            "| `/status` | Show runtime state snapshot |",
            "| `/changes [path] [--staged]` | Show git status and diff preview |",
            "| `/doctor` | Run local health checks |",
            "| `/bug [note]` | Save a reproducible bug report file |",
            "| `/trace [on\\|off]` | Toggle trace messages in transcript |",
            "| `/clear` | Clear visible transcript only |",
            "| `/agents` | List loaded agents |",
            "| `/agent <name>` | Switch active agent |",
            "| `/pipeline <a> <b> [rounds]` | Enable inter-agent pipeline |",
            "| `/pipeline stop` | Disable current pipeline |",
            "| `/context` | Show session/data context |",
            "| `/sessions` / `/load <id>` | List and load previous sessions |",
            "| `/export [path]` | Export chat history to markdown |",
            "| `/upload <file> [label]` | Ingest one document into RAG |",
            "| `/upload-dir <dir>` | Bulk ingest documents into RAG |",
            "| `/docs <library> [query]` | Fetch Context7 library docs |",
            "| `/new` | Start a new session |",
            "| `/reload` | Reload config and agents |",
            "| `/quit` | Exit CLI |",
            "",
            "Shortcuts: `Ctrl+Q` quit · `Ctrl+O` trace toggle · `Ctrl+P` raw stream · `Ctrl+C` exit.",
        ]
        .join("\n")
    }

    // ── Status ────────────────────────────────────────────────────────────────

    #[allow(dead_code)]
    pub fn update_bridge_state(&mut self, v: &Value) {
        self.bridge_state = BridgeState::from_value(v);
        let s = &self.bridge_state;
        let agents = if s.agents.is_empty() {
            "-".to_string()
        } else {
            s.agents.join(", ")
        };
        let session = &s.session[..s.session.len().min(24)];
        let pipeline = if let Some(p) = &s.pipeline {
            format!(
                "{} -> {} x{}",
                p["a"].as_str().unwrap_or("?"),
                p["b"].as_str().unwrap_or("?"),
                p["rounds"]
            )
        } else {
            "off".to_string()
        };
        self.add_system_message(format!(
            "active: {}\nsession: {}\nmessages: {}\nagents: {}\npipeline: {}\nrapidfuzz: {}\ntrace: {}",
            s.active, session, s.msg_count, agents, pipeline,
            if s.rapidfuzz { "enabled" } else { "disabled" },
            if self.trace_on { "on" } else { "off" },
        ));
    }

    // ── Spinner tick ──────────────────────────────────────────────────────────

    pub fn tick(&mut self) {
        if self.busy {
            self.spinner_tick = (self.spinner_tick + 1) % 10;
        }
    }

    pub fn toggle_trace(&mut self) {
        self.trace_on = !self.trace_on;
    }

    pub fn toggle_raw_stream(&mut self) {
        self.raw_on = !self.raw_on;
        if self.at_bottom {
            self.scroll_to_bottom();
        } else {
            self.clamp_scroll();
        }
    }

    pub fn handle_paste(&mut self, text: &str) {
        if self.busy || text.is_empty() {
            return;
        }
        let normalized = text.replace('\r', "\n").replace('\n', " ");
        if normalized.is_empty() {
            return;
        }
        self.insert_input_text(&normalized);
    }

    pub fn slash_popup_visible(&self) -> bool {
        self.slash_token_span().is_some()
    }

    pub fn slash_popup_selected(&self) -> usize {
        self.slash_selected
    }

    pub fn slash_popup_entries(&self, limit: usize) -> Vec<SlashPopupEntry> {
        let Some((_, _, token)) = self.slash_token_span() else {
            return Vec::new();
        };

        let query = token.trim_start_matches('/').to_lowercase();
        let mut scored: Vec<(i32, usize, SlashPopupEntry)> = SLASH_COMMANDS
            .iter()
            .enumerate()
            .filter_map(|(order, spec)| {
                Self::slash_score(&query, *spec, order).map(|score| {
                    (
                        score,
                        order,
                        SlashPopupEntry {
                            command: spec.command,
                            description: spec.description,
                        },
                    )
                })
            })
            .collect();

        scored.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| a.2.command.len().cmp(&b.2.command.len()))
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.command.cmp(b.2.command))
        });

        let cap = limit.max(1);
        scored
            .into_iter()
            .take(cap)
            .map(|(_, _, entry)| entry)
            .collect()
    }

    fn insert_input_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let ci = self.char_idx();
        let before: String = self.input.chars().take(ci).collect();
        let after: String = self.input.chars().skip(ci).collect();
        self.input = format!("{before}{text}{after}");
        self.input_cursor += text.chars().count();
        self.normalize_slash_selection();
    }

    // ── Status bar text ───────────────────────────────────────────────────────

    pub fn status_text(&self) -> String {
        const SPINNERS: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let spin = if self.busy && !matches!(self.phase, Phase::Ready | Phase::Error) {
            format!("{} ", SPINNERS[self.spinner_tick])
        } else {
            String::new()
        };
        format!(
            "{}{} · {} · agent:{} · msgs:{} · tools:{} · skills:{} · trace:{} · raw:{} · Ctrl+O/P",
            spin,
            self.phase,
            self.phase_note,
            self.bridge_state.active,
            self.bridge_state.msg_count,
            self.bridge_state.tool_count,
            self.bridge_state.skill_count,
            if self.trace_on { "on" } else { "off" },
            if self.raw_on { "on" } else { "off" },
        )
    }

    // ── Keyboard handling ─────────────────────────────────────────────────────

    pub fn handle_key(&mut self, key: KeyEvent) -> KeyAction {
        // Global shortcuts (work even when busy)
        if key.modifiers == KeyModifiers::CONTROL {
            match key.code {
                KeyCode::Char('c') => {
                    self.should_quit = true;
                    return KeyAction::Quit;
                }
                KeyCode::Char('o') => {
                    return KeyAction::ToggleTrace;
                }
                KeyCode::Char('p') => {
                    return KeyAction::ToggleRawStream;
                }
                KeyCode::Char('l') => {
                    self.clear_transcript();
                    return KeyAction::None;
                }
                _ => {}
            }
        }

        // When slash popup is visible, Up/Down navigate suggestions.
        if !self.busy && key.modifiers.is_empty() && self.slash_popup_visible() {
            match key.code {
                KeyCode::Up => {
                    self.move_slash_selection(-1);
                    return KeyAction::None;
                }
                KeyCode::Down => {
                    self.move_slash_selection(1);
                    return KeyAction::None;
                }
                _ => {}
            }
        }

        // Scroll keys (work even when busy)
        match key.code {
            KeyCode::Up => {
                self.scroll_up(3);
                return KeyAction::None;
            }
            KeyCode::Down => {
                self.scroll_down(3);
                return KeyAction::None;
            }
            KeyCode::PageUp => {
                self.scroll_up(self.last_area_h.saturating_sub(2));
                return KeyAction::None;
            }
            KeyCode::PageDown => {
                self.scroll_down(self.last_area_h.saturating_sub(2));
                return KeyAction::None;
            }
            _ => {}
        }

        if self.busy {
            return KeyAction::None;
        }

        // Input editing
        match key.code {
            KeyCode::Enter => {
                if self.should_complete_slash_on_enter() && self.apply_selected_slash_command() {
                    return KeyAction::None;
                }
                let text = self.input.trim().to_string();
                if text.is_empty() {
                    return KeyAction::None;
                }
                self.input.clear();
                self.input_cursor = 0;
                self.slash_selected = 0;
                self.add_message(Role::User, text.clone());
                self.busy = true;
                self.phase = Phase::Thinking;
                self.phase_note = "thinking".into();
                return KeyAction::Submit(text);
            }
            KeyCode::Backspace => {
                if self.input_cursor > 0 {
                    // Remove char before cursor
                    let before: String = self.input.chars().take(self.char_idx()).collect();
                    let after: String = self.input.chars().skip(self.char_idx()).collect();
                    let new_before: String = before
                        .chars()
                        .take(before.chars().count().saturating_sub(1))
                        .collect();
                    self.input_cursor = self.input_cursor.saturating_sub(1);
                    self.input = format!("{new_before}{after}");
                    self.normalize_slash_selection();
                }
            }
            KeyCode::Delete => {
                let ci = self.char_idx();
                let total = self.input.chars().count();
                if ci < total {
                    let before: String = self.input.chars().take(ci).collect();
                    let after: String = self.input.chars().skip(ci + 1).collect();
                    self.input = format!("{before}{after}");
                    self.normalize_slash_selection();
                }
            }
            KeyCode::Left => {
                self.input_cursor = self.input_cursor.saturating_sub(1);
            }
            KeyCode::Right => {
                let total = self.input.chars().count();
                if self.input_cursor < total {
                    self.input_cursor += 1;
                }
            }
            KeyCode::Home => {
                self.input_cursor = 0;
            }
            KeyCode::End => {
                self.input_cursor = self.input.chars().count();
            }
            KeyCode::Char(c) => {
                if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT {
                    self.insert_input_text(&c.to_string());
                }
            }
            KeyCode::Tab => {
                if self.slash_popup_visible() {
                    let _ = self.apply_selected_slash_command();
                } else {
                    // Insert spaces
                    self.insert_input_text("    ");
                }
            }
            KeyCode::BackTab => {
                if self.slash_popup_visible() {
                    self.move_slash_selection(-1);
                }
            }
            _ => {}
        }

        // ── Keyboard shortcuts ──────────────────────────────────────────────
        match key.code {
            KeyCode::Char('q') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return KeyAction::Quit;
            }
            KeyCode::Char('o') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return KeyAction::ToggleTrace;
            }
            KeyCode::Char('p') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return KeyAction::ToggleRawStream;
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return KeyAction::Quit;
            }
            _ => {}
        }
        KeyAction::None
    }

    /// Character index (count of chars, not bytes) at cursor.
    fn char_idx(&self) -> usize {
        self.input_cursor
    }

    /// Byte offset of cursor in input string.
    pub fn input_byte_offset(&self) -> usize {
        self.input
            .char_indices()
            .nth(self.input_cursor)
            .map(|(b, _)| b)
            .unwrap_or(self.input.len())
    }

    fn slash_token_span(&self) -> Option<(usize, usize, String)> {
        let chars: Vec<char> = self.input.chars().collect();
        let start = chars.iter().position(|c| !c.is_whitespace())?;
        if chars.get(start).copied()? != '/' {
            return None;
        }

        let mut end = start + 1;
        while end < chars.len() && !chars[end].is_whitespace() {
            end += 1;
        }
        let token: String = chars[start..end].iter().collect();
        Some((start, end, token))
    }

    fn normalize_slash_selection(&mut self) {
        let len = self.slash_popup_entries(SLASH_POPUP_LIMIT).len();
        if len == 0 {
            self.slash_selected = 0;
            return;
        }
        if self.slash_selected >= len {
            self.slash_selected = 0;
        }
    }

    fn move_slash_selection(&mut self, delta: isize) {
        let len = self.slash_popup_entries(SLASH_POPUP_LIMIT).len();
        if len == 0 {
            self.slash_selected = 0;
            return;
        }
        let current = self.slash_selected.min(len - 1) as isize;
        let mut next = current + delta;
        if next < 0 {
            next = len as isize - 1;
        } else if next >= len as isize {
            next = 0;
        }
        self.slash_selected = next as usize;
    }

    fn apply_selected_slash_command(&mut self) -> bool {
        let entries = self.slash_popup_entries(SLASH_POPUP_LIMIT);
        if entries.is_empty() {
            return false;
        }
        let idx = self.slash_selected.min(entries.len().saturating_sub(1));
        self.replace_slash_token(entries[idx].command)
    }

    fn should_complete_slash_on_enter(&self) -> bool {
        let Some((_, _, token)) = self.slash_token_span() else {
            return false;
        };
        let normalized = token.trim().to_lowercase();
        if normalized == "/" {
            return true;
        }
        !Self::is_known_slash_command(&normalized)
    }

    fn is_known_slash_command(token: &str) -> bool {
        SLASH_COMMANDS
            .iter()
            .any(|spec| spec.command.eq_ignore_ascii_case(token))
    }

    fn replace_slash_token(&mut self, replacement: &str) -> bool {
        let Some((start, end, _)) = self.slash_token_span() else {
            return false;
        };
        let chars: Vec<char> = self.input.chars().collect();
        let before: String = chars[..start].iter().collect();
        let after: String = chars[end..].iter().collect();

        let mut insert = replacement.to_string();
        let next_char = after.chars().next();
        if next_char.is_none() || !next_char.is_some_and(|c| c.is_whitespace()) {
            insert.push(' ');
        }

        self.input = format!("{before}{insert}{after}");
        self.input_cursor = before.chars().count() + insert.chars().count();
        self.normalize_slash_selection();
        true
    }

    fn slash_score(query: &str, spec: SlashCommandSpec, order: usize) -> Option<i32> {
        let q = query.trim();
        if q.is_empty() {
            return Some(3_000 - order as i32);
        }

        let candidate = spec.command.trim_start_matches('/');
        let candidate_l = candidate.to_lowercase();
        let desc_l = spec.description.to_lowercase();

        if candidate_l == q {
            return Some(6_000 - order as i32);
        }
        if candidate_l.starts_with(q) {
            let rem = (candidate_l.len() as i32 - q.len() as i32).max(0);
            return Some(5_000 - rem - order as i32);
        }
        if let Some(pos) = candidate_l.find(q) {
            return Some(4_000 - (pos as i32 * 8) - order as i32);
        }
        if let Some(gap) = Self::subsequence_gap(q, &candidate_l) {
            return Some(3_200 - (gap * 6) - order as i32);
        }
        if desc_l.contains(q) {
            return Some(2_000 - order as i32);
        }
        None
    }

    fn subsequence_gap(query: &str, candidate: &str) -> Option<i32> {
        if query.is_empty() {
            return Some(0);
        }
        let mut q_iter = query.chars();
        let mut current = q_iter.next()?;
        let mut last_match: Option<usize> = None;
        let mut gap = 0i32;

        for (idx, c) in candidate.chars().enumerate() {
            if c != current {
                continue;
            }
            if let Some(prev) = last_match {
                gap += idx.saturating_sub(prev + 1) as i32;
            }
            last_match = Some(idx);
            if let Some(next) = q_iter.next() {
                current = next;
            } else {
                return Some(gap);
            }
        }

        None
    }
}
