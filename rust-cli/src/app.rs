use chrono::Local;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::{layout::Rect, Frame};
use serde_json::Value;
use std::fmt;
use uuid::Uuid;

use crate::bridge::{BridgeEvent, BridgeState};
use crate::image::ImageRenderer;
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

impl DefaultRenderer {
    fn tool_message_is_error(text: &str) -> bool {
        let lower = text.trim().to_lowercase();
        lower.starts_with("failed ")
            || lower.starts_with("error:")
            || lower.contains("\nerror:")
            || lower.contains(" status=error")
    }
}

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
                let is_error = Self::tool_message_is_error(text);
                let emphasis = if is_error { Color::Red } else { Color::Cyan };
                for (idx, line) in text.lines().enumerate() {
                    let (prefix, style) = if idx == 0 {
                        (
                            "  ▸ ",
                            Style::default()
                                .fg(emphasis)
                                .add_modifier(Modifier::BOLD),
                        )
                    } else if is_error && line.trim_start().starts_with("error:") {
                        (
                            "    ",
                            Style::default()
                                .fg(Color::Red)
                                .add_modifier(Modifier::BOLD),
                        )
                    } else {
                        ("    ", Style::default().fg(emphasis))
                    };
                    tool_lines.push(Line::from(vec![
                        Span::styled(prefix.to_string(), Style::default().fg(Color::DarkGray)),
                        Span::styled(line.to_string(), style),
                    ]));
                }
                tool_lines
            }
            Role::Skill => {
                let mut skill_lines = Vec::new();
                for (idx, line) in text.lines().enumerate() {
                    let (prefix, style) = if idx == 0 {
                        (
                            "  ◆ ",
                            Style::default()
                                .fg(Color::Magenta)
                                .add_modifier(Modifier::BOLD),
                        )
                    } else {
                        ("    ", Style::default().fg(Color::Magenta))
                    };
                    skill_lines.push(Line::from(vec![
                        Span::styled(prefix.to_string(), Style::default().fg(Color::DarkGray)),
                        Span::styled(line.to_string(), style),
                    ]));
                }
                skill_lines
            }
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
    /// Toggle task panel
    ToggleTasks,
}

#[derive(Clone)]
pub struct SlashPopupEntry {
    pub command: String,
    pub description: String,
}

#[derive(Clone, Copy)]
struct SlashCommandSpec {
    command: &'static str,
    description: &'static str,
}

const SLASH_POPUP_LIMIT: usize = 8;
const INPUT_HISTORY_LIMIT: usize = 200;

const LOCAL_ONLY_SLASH_COMMANDS: [SlashCommandSpec; 4] = [
    SlashCommandSpec {
        command: "/?",
        description: "Alias for /help",
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
        command: "/close",
        description: "Close image side panel",
    },
];

const SLASH_COMMANDS: [SlashCommandSpec; 28] = [
    SlashCommandSpec {
        command: "/help",
        description: "Show command list",
    },
    SlashCommandSpec {
        command: "/?",
        description: "Alias for /help",
    },
    SlashCommandSpec {
        command: "/version",
        description: "Show CLI and bridge version info",
    },
    SlashCommandSpec {
        command: "/status",
        description: "Show runtime state snapshot",
    },
    SlashCommandSpec {
        command: "/skills-health",
        description: "Show skill loader diagnostics",
    },
    SlashCommandSpec {
        command: "/changes",
        description: "Show git status and diff preview",
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
        command: "/close",
        description: "Close image side panel",
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
        command: "/compact",
        description: "Summarize older conversation history",
    },
    SlashCommandSpec {
        command: "/reset",
        description: "Reset runtime tool state for session",
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
        command: "/mount",
        description: "Mount codebase (context + RAG)",
    },
    SlashCommandSpec {
        command: "/mount-code",
        description: "Alias for /mount",
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
    SlashCommandSpec {
        command: "/q",
        description: "Alias for /quit",
    },
];

#[derive(Clone)]
struct SlashCommand {
    command: String,
    description: String,
}

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
    // Per-turn tool execution tracking (receipt-style UX).
    current_turn_tool_names: Vec<String>,
    last_turn_tool_names: Vec<String>,
    last_turn_iterations: u64,
    // Optional inline image preview for image-producing tools.
    image_renderer: Option<ImageRenderer>,
    image_path: Option<String>,

    // Input
    pub input: String,
    pub input_cursor: usize, // char offset
    slash_selected: usize,
    slash_commands: Vec<SlashCommand>,
    input_history: Vec<String>,
    input_history_index: Option<usize>,
    input_history_draft: Option<String>,

    // Scroll  (lines from top)
    pub scroll_top: u16,
    pub at_bottom: bool,
    pub last_area_h: u16, // last known messages viewport height

    // Phase / status
    pub phase: Phase,
    pub phase_note: String,
    pub last_tool_error: Option<String>,
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
    pub todo_on: bool,
}

impl App {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            cached_lines: Vec::new(),
            cached_lines_raw: Vec::new(),
            live: None,
            raw_buf: String::new(),
            current_turn_tool_names: Vec::new(),
            last_turn_tool_names: Vec::new(),
            last_turn_iterations: 0,
            image_renderer: ImageRenderer::new().ok(),
            image_path: None,
            input: String::new(),
            input_cursor: 0,
            slash_selected: 0,
            slash_commands: Self::default_slash_commands(),
            input_history: Vec::new(),
            input_history_index: None,
            input_history_draft: None,
            scroll_top: 0,
            at_bottom: true,
            last_area_h: 40,
            phase: Phase::Ready,
            phase_note: "ready".into(),
            last_tool_error: None,
            busy: false,
            spinner_tick: 0,
            trace_on: true,
            raw_on: false,
            bridge_state: BridgeState::default(),
            connected: false,
            should_quit: false,
            trace_log: Vec::new(),
            todo_on: false,
        }
    }

    fn append_slash_command(
        commands: &mut Vec<SlashCommand>,
        command: impl Into<String>,
        description: impl Into<String>,
    ) {
        let raw = command.into();
        let command = raw
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        if command.is_empty() || !command.starts_with('/') {
            return;
        }
        if commands
            .iter()
            .any(|item| item.command.eq_ignore_ascii_case(&command))
        {
            return;
        }
        let description = description.into().trim().to_string();
        commands.push(SlashCommand {
            command,
            description,
        });
    }

    fn default_slash_commands() -> Vec<SlashCommand> {
        let mut out: Vec<SlashCommand> = Vec::new();
        for spec in SLASH_COMMANDS {
            Self::append_slash_command(&mut out, spec.command, spec.description);
        }
        out
    }

    fn install_bridge_commands(&mut self, v: &Value) {
        let Some(items) = v.get("commands").and_then(|value| value.as_array()) else {
            return;
        };
        let mut merged: Vec<SlashCommand> = Vec::new();
        for item in items {
            let command = item
                .get("command")
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if command.is_empty() {
                continue;
            }
            let description = item
                .get("description")
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            Self::append_slash_command(&mut merged, command, description);
        }

        if merged.is_empty() {
            return;
        }

        for spec in LOCAL_ONLY_SLASH_COMMANDS {
            Self::append_slash_command(&mut merged, spec.command, spec.description);
        }
        self.slash_commands = merged;
        self.normalize_slash_selection();
    }

    // ── Init ─────────────────────────────────────────────────────────────────

    pub fn handle_init(&mut self, v: Value) {
        self.install_bridge_commands(&v);
        let state = v.get("state").unwrap_or(&v);
        self.bridge_state = BridgeState::from_value(state);
        self.connected = true;
        let agents = if self.bridge_state.agents.is_empty() {
            "-".to_string()
        } else {
            self.bridge_state.agents.join(", ")
        };
        let mcps = if self.bridge_state.mcp_servers.is_empty() {
            "-".to_string()
        } else {
            self.bridge_state.mcp_servers.join(", ")
        };
        let text = format!(
            "# Logician CLI powered by `rust`\n\n\
**Agents**: {agents}  \n\
**MCPs**: {mcps}  \n\
**Rapidfuzz**: {}  \n\
**Tiktoken**: {}  \n\n\
Active: `{}` · Session: `{}` · Ctrl+O trace · Ctrl+P raw stream · Ctrl+C quit",
            if self.bridge_state.rapidfuzz {
                "enabled"
            } else {
                "disabled"
            },
            if self.bridge_state.tiktoken {
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

    pub fn has_image_preview(&self) -> bool {
        self.image_path.is_some() && self.image_renderer.is_some()
    }

    pub fn image_preview_path(&self) -> Option<&str> {
        self.image_path.as_deref()
    }

    pub fn render_image_preview(&mut self, frame: &mut Frame, area: Rect) -> Result<(), String> {
        if area.width < 2 || area.height < 2 {
            return Err("preview area too small".to_string());
        }
        let renderer = self
            .image_renderer
            .as_mut()
            .ok_or_else(|| "image renderer unavailable".to_string())?;
        renderer
            .render_in_frame(frame, area)
            .map_err(|err| err.to_string())
    }

    fn clear_image_preview(&mut self) {
        if let Some(renderer) = self.image_renderer.as_mut() {
            renderer.clear();
        }
        self.image_path = None;
    }

    fn load_image_preview(&mut self, path: &str) -> Result<bool, String> {
        let normalized = path.trim();
        if normalized.is_empty() {
            return Err("empty image path".to_string());
        }
        let changed = self.image_path.as_deref() != Some(normalized);
        let renderer = self
            .image_renderer
            .as_mut()
            .ok_or_else(|| "image renderer unavailable".to_string())?;
        renderer
            .load_path(normalized)
            .map_err(|err| err.to_string())?;
        self.image_path = Some(normalized.to_string());
        Ok(changed)
    }

    fn truncate_inline(text: &str, max_chars: usize) -> String {
        if max_chars == 0 {
            return String::new();
        }
        let mut out = String::new();
        let mut count = 0usize;
        for ch in text.chars() {
            if count >= max_chars {
                out.push('…');
                return out;
            }
            out.push(ch);
            count += 1;
        }
        out
    }

    fn summarize_json_value(value: &Value) -> String {
        match value {
            Value::Null => "null".to_string(),
            Value::Bool(v) => v.to_string(),
            Value::Number(v) => v.to_string(),
            Value::String(v) => {
                let compact = v.split_whitespace().collect::<Vec<_>>().join(" ");
                format!("\"{}\"", Self::truncate_inline(&compact, 64))
            }
            Value::Array(items) => {
                if items.is_empty() {
                    "[]".to_string()
                } else if items.len() <= 3 {
                    let parts = items
                        .iter()
                        .map(Self::summarize_json_value)
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("[{parts}]")
                } else {
                    format!("[{} items]", items.len())
                }
            }
            Value::Object(map) => format!("{{{} keys}}", map.len()),
        }
    }

    fn summarize_name_list(items: &[String], limit: usize) -> String {
        if items.is_empty() {
            return "none".to_string();
        }
        let mut out = items.iter().take(limit).cloned().collect::<Vec<_>>();
        if items.len() > limit {
            out.push(format!("+{} more", items.len() - limit));
        }
        out.join(", ")
    }

    fn format_tool_event_text(name: &str, args: &Value, sequence: usize) -> String {
        let mut lines = vec![format!("call #{sequence} `{name}`")];

        match args {
            Value::Object(map) => {
                lines.push(format!("arguments: {} key(s)", map.len()));
                if map.is_empty() {
                    lines.push("• none".to_string());
                } else {
                    for (key, value) in map.iter().take(6) {
                        lines.push(format!("• {key} = {}", Self::summarize_json_value(value)));
                    }
                    if map.len() > 6 {
                        lines.push(format!("• ... +{} more key(s)", map.len() - 6));
                    }
                    if map.len() > 4 {
                        let raw = serde_json::to_string(args).unwrap_or_default();
                        if !raw.is_empty() {
                            lines.push(format!("raw: {}", Self::truncate_inline(&raw, 180)));
                        }
                    }
                }
            }
            _ => {
                lines.push(format!("arguments: {}", Self::summarize_json_value(args)));
            }
        }

        lines.join("\n")
    }

    fn format_tool_completion_text(
        name: &str,
        sequence: usize,
        status: &str,
        duration_ms: u64,
        cache_hit: bool,
        error: Option<&str>,
    ) -> String {
        let normalized = status.trim().to_lowercase();
        let label = if normalized == "ok" {
            "completed"
        } else {
            "failed"
        };
        let mut lines = vec![format!(
            "{label} #{sequence} `{name}` in {} ms{}",
            duration_ms,
            if cache_hit { " (cache hit)" } else { "" },
        )];
        if let Some(err) = error {
            let short = Self::truncate_inline(err.trim(), 220);
            if !short.is_empty() {
                lines.push(format!("error: {short}"));
            }
        }
        lines.join("\n")
    }

    fn format_skill_event_text(skill_ids: &[String], selected_tools: &[String]) -> String {
        let mut lines = Vec::new();
        if skill_ids.is_empty() {
            lines.push("activated skills: none".to_string());
        } else {
            lines.push(format!(
                "activated skills ({}): {}",
                skill_ids.len(),
                Self::summarize_name_list(skill_ids, 6)
            ));
        }

        if selected_tools.is_empty() {
            lines.push("routed tools: none".to_string());
        } else {
            lines.push(format!(
                "routed tools ({}): {}",
                selected_tools.len(),
                Self::summarize_name_list(selected_tools, 10)
            ));
        }

        lines.join("\n")
    }

    fn extract_tool_names_from_chat_result(v: &Value) -> Vec<String> {
        let mut out: Vec<String> = v["tool_calls"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| item["name"].as_str())
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                    .map(|name| name.to_string())
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();
        if out.is_empty() {
            let count = v["tool_call_count"].as_u64().unwrap_or(0);
            if count > 0 {
                out = vec!["(reported by runtime)".to_string(); count as usize];
            }
        }
        out
    }

    fn response_claims_completed_action(text: &str) -> bool {
        let lower = text.trim().to_lowercase();
        if lower.is_empty() {
            return false;
        }
        if [
            " i will ",
            " we'll ",
            " we will ",
            " i can ",
            " we can ",
            " could ",
            " should ",
            " might ",
            " may ",
        ]
        .iter()
        .any(|marker| lower.contains(marker))
        {
            return false;
        }
        let strong_phrases = [
            "i've generated",
            "i have generated",
            "i've created",
            "i have created",
            "has been generated",
            "has been created",
            "created successfully",
            "generated successfully",
            "is now available",
            "saved successfully",
            "loaded successfully",
            "in tool context",
            "in the tool context",
        ];
        if strong_phrases.iter().any(|p| lower.contains(p)) {
            return true;
        }
        let past_verbs = [
            "generated",
            "created",
            "loaded",
            "saved",
            "executed",
            "called",
            "used",
            "invoked",
            "applied",
            "updated",
            "finished",
            "completed",
            "done",
        ];
        let claim_subject = [
            "i ", "we ", "tool ", "tools ", "data ", "dataset ", "sample ",
        ];
        past_verbs.iter().any(|verb| lower.contains(verb))
            && claim_subject.iter().any(|subject| lower.contains(subject))
    }

    fn finalize_turn_receipt(&mut self, v: &Value, assistant_text: &str) {
        let reported_tool_names = Self::extract_tool_names_from_chat_result(v);
        let reported_count = if reported_tool_names.is_empty() {
            v["tool_call_count"].as_u64().unwrap_or(0) as usize
        } else {
            reported_tool_names.len()
        };
        let observed_count = self.current_turn_tool_names.len();
        let runtime_tools = if !reported_tool_names.is_empty() {
            reported_tool_names
        } else {
            self.current_turn_tool_names.clone()
        };
        self.last_turn_tool_names = runtime_tools.clone();
        self.last_turn_iterations = v["iterations"].as_u64().unwrap_or(0);

        if !runtime_tools.is_empty() {
            let mut uniq = Vec::<String>::new();
            for name in &runtime_tools {
                if !uniq.iter().any(|existing| existing == name) {
                    uniq.push(name.clone());
                }
            }
            let preview = uniq.iter().take(6).cloned().collect::<Vec<_>>().join(", ");
            self.add_system_message(format!(
                "Turn receipt: tools={} · iterations={} · {}",
                runtime_tools.len(),
                self.last_turn_iterations,
                preview
            ));
        }

        if reported_count != observed_count && (reported_count > 0 || observed_count > 0) {
            self.add_system_message(format!(
                "Runtime note: tool event mismatch (observed events={observed_count}, runtime reported={reported_count})."
            ));
        }

        if runtime_tools.is_empty() && Self::response_claims_completed_action(assistant_text) {
            self.add_system_message(
                "Runtime note: assistant claimed work was completed, but no tool call was recorded for this turn."
                    .to_string(),
            );
        }

        self.current_turn_tool_names.clear();
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
        self.cached_lines_raw
            .extend(message.rendered_for_raw_mode());
        self.messages.push(message);
    }

    fn clear_transcript(&mut self) {
        self.messages.clear();
        self.cached_lines.clear();
        self.cached_lines_raw.clear();
        self.live = None;
        self.raw_buf.clear();
        self.current_turn_tool_names.clear();
        self.last_turn_tool_names.clear();
        self.last_turn_iterations = 0;
        self.clear_image_preview();
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

            BridgeEvent::ToolStart {
                name,
                args,
                sequence,
            } => {
                self.stop_live_message();
                let call_number = if sequence == 0 {
                    self.current_turn_tool_names.len() + 1
                } else {
                    sequence as usize
                };
                self.trace(format!("tool#{call_number}={name}"));
                self.current_turn_tool_names.push(name.clone());
                self.add_message(
                    Role::Tool,
                    Self::format_tool_event_text(&name, &args, call_number),
                );
            }

            BridgeEvent::ToolEnd {
                name,
                sequence,
                status,
                duration_ms,
                cache_hit,
                error,
            } => {
                self.stop_live_message();
                let call_number = if sequence == 0 {
                    self.current_turn_tool_names.len()
                } else {
                    sequence as usize
                };
                let status_l = status.to_lowercase();
                self.trace(format!(
                    "tool_end#{call_number}={name} status={status_l} duration_ms={duration_ms}"
                ));
                self.add_message(
                    Role::Tool,
                    Self::format_tool_completion_text(
                        &name,
                        call_number,
                        &status,
                        duration_ms,
                        cache_hit,
                        error.as_deref(),
                    ),
                );
                if status_l == "error" || status_l == "failed" {
                    let detail = error
                        .as_deref()
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(|value| Self::truncate_inline(value, 220))
                        .unwrap_or_else(|| "tool execution failed".to_string());
                    self.last_tool_error = Some(format!("{name}: {detail}"));
                }
            }

            BridgeEvent::Image { tool, path } => {
                self.stop_live_message();
                self.trace(format!("image tool={tool} path={path}"));
                match self.load_image_preview(&path) {
                    Ok(changed) => {
                        if changed {
                            self.add_system_message(format!(
                                "Image preview updated from `{tool}`: `{path}`"
                            ));
                        }
                    }
                    Err(err) => {
                        self.add_system_message(format!(
                            "Image preview failed for `{path}` from `{tool}`: {err}"
                        ));
                    }
                }
            }

            BridgeEvent::Skill {
                skill_ids,
                selected_tools,
            } => {
                self.stop_live_message();
                let skills = Self::summarize_name_list(&skill_ids, 6);
                let tools = Self::summarize_name_list(&selected_tools, 10);
                self.trace(format!("skills={skills} tools={tools}"));
                self.add_message(
                    Role::Skill,
                    Self::format_skill_event_text(&skill_ids, &selected_tools),
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
                let assistant_text = v["assistant"].as_str().unwrap_or("").to_string();
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
                self.finalize_turn_receipt(&v, &assistant_text);
            }
            Err(e) => {
                self.live = None;
                self.raw_buf.clear();
                self.current_turn_tool_names.clear();
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
                self.install_bridge_commands(&v);
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
            self.add_system_message(self.help_text());
            return true;
        }

        if lower == "/clear" {
            self.clear_transcript();
            self.add_system_message("Transcript cleared. Session state is unchanged.");
            return true;
        }

        if lower == "/close" {
            let mut closed_any = false;
            if self.has_image_preview() {
                self.clear_image_preview();
                self.add_system_message("Image side panel closed.");
                closed_any = true;
            }
            if self.todo_on {
                self.todo_on = false;
                self.add_system_message("Task side panel closed.");
                closed_any = true;
            }

            if !closed_any {
                self.add_system_message("No side panels are open to close.");
            }
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

        if lower == "/version" {
            self.add_system_message(format!(
                "logician-cli {}\nbridge: use /version for full runtime details",
                env!("CARGO_PKG_VERSION")
            ));
            return true;
        }

        false
    }

    #[allow(dead_code)]
    fn help_text(&self) -> String {
        let mut lines: Vec<String> = vec![
            "# Command Palette".to_string(),
            "".to_string(),
            "## Available Commands".to_string(),
        ];
        for item in &self.slash_commands {
            let description = if item.description.is_empty() {
                "No description"
            } else {
                item.description.as_str()
            };
            lines.push(format!("- `{}`  {}", item.command, description));
        }
        lines.push("".to_string());
        lines.push("Shortcuts: `Up/Down` input history · mouse wheel / `PgUp/PgDn` scroll · `Shift+Drag` select text · `Ctrl+Q` quit · `Ctrl+O` trace · `Ctrl+P` raw stream · `Ctrl+C` exit.".to_string());
        lines.join("\n")
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
        let mcps = if s.mcp_servers.is_empty() {
            "-".to_string()
        } else {
            s.mcp_servers.join(", ")
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
            "active: {}\nsession: {}\nmessages: {}\nagents: {}\nmcp: {}\npipeline: {}\nrapidfuzz: {}\ntiktoken: {}\ntrace: {}",
            s.active, session, s.msg_count, agents, mcps, pipeline,
            if s.rapidfuzz { "enabled" } else { "disabled" },
            if s.tiktoken { "enabled" } else { "disabled" },
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

    pub fn toggle_todo(&mut self) {
        self.todo_on = !self.todo_on;
        self.trace(format!("todo_panel={}", self.todo_on));
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
        let mut scored: Vec<(i32, usize, SlashPopupEntry)> = self
            .slash_commands
            .iter()
            .enumerate()
            .filter_map(|(order, spec)| {
                Self::slash_score(&query, spec, order).map(|score| {
                    (
                        score,
                        order,
                        SlashPopupEntry {
                            command: spec.command.clone(),
                            description: spec.description.clone(),
                        },
                    )
                })
            })
            .collect();

        scored.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| a.2.command.len().cmp(&b.2.command.len()))
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.command.cmp(&b.2.command))
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
        self.clear_input_history_nav();
        let ci = self.char_idx();
        let before: String = self.input.chars().take(ci).collect();
        let after: String = self.input.chars().skip(ci).collect();
        self.input = format!("{before}{text}{after}");
        self.input_cursor += text.chars().count();
        self.normalize_slash_selection();
    }

    fn push_input_history(&mut self, text: &str) {
        let entry = text.trim();
        if entry.is_empty() {
            return;
        }
        self.input_history.push(entry.to_string());
        if self.input_history.len() > INPUT_HISTORY_LIMIT {
            let overflow = self.input_history.len() - INPUT_HISTORY_LIMIT;
            self.input_history.drain(0..overflow);
        }
        self.clear_input_history_nav();
    }

    fn clear_input_history_nav(&mut self) {
        self.input_history_index = None;
        self.input_history_draft = None;
    }

    fn set_input_from_history_value(&mut self, value: String) {
        self.input = value;
        self.input_cursor = self.input.chars().count();
        self.normalize_slash_selection();
    }

    fn history_prev(&mut self) {
        if self.input_history.is_empty() {
            return;
        }
        let next_idx = match self.input_history_index {
            Some(idx) => idx.saturating_sub(1),
            None => {
                self.input_history_draft = Some(self.input.clone());
                self.input_history.len().saturating_sub(1)
            }
        };
        self.input_history_index = Some(next_idx);
        if let Some(entry) = self.input_history.get(next_idx).cloned() {
            self.set_input_from_history_value(entry);
        }
    }

    fn history_next(&mut self) {
        let Some(idx) = self.input_history_index else {
            return;
        };
        if idx + 1 < self.input_history.len() {
            let next_idx = idx + 1;
            self.input_history_index = Some(next_idx);
            if let Some(entry) = self.input_history.get(next_idx).cloned() {
                self.set_input_from_history_value(entry);
            }
            return;
        }

        self.input_history_index = None;
        let draft = self.input_history_draft.take().unwrap_or_default();
        self.set_input_from_history_value(draft);
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
            "{}{} · {} · agent:{} · msgs:{} · tools:{} · skills:{} · last_turn_tools:{} · image:{} · trace:{} · raw:{} · Ctrl+O/P",
            spin,
            self.phase,
            self.phase_note,
            self.bridge_state.active,
            self.bridge_state.msg_count,
            self.bridge_state.tool_count,
            self.bridge_state.skill_count,
            self.last_turn_tool_names.len(),
            if self.has_image_preview() { "on" } else { "off" },
            if self.trace_on { "on" } else { "off" },
            if self.raw_on { "on" } else { "off" },
        )
    }

    pub fn last_turn_tool_count(&self) -> usize {
        self.last_turn_tool_names.len()
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
            KeyCode::Up if self.busy => {
                self.scroll_up(3);
                return KeyAction::None;
            }
            KeyCode::Down if self.busy => {
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
                self.push_input_history(&text);
                self.input.clear();
                self.input_cursor = 0;
                self.slash_selected = 0;
                self.last_tool_error = None;
                if text.starts_with('/') && self.handle_local_slash(&text) {
                    return KeyAction::None;
                }
                self.current_turn_tool_names.clear();
                self.add_message(Role::User, text.clone());
                self.busy = true;
                self.phase = Phase::Thinking;
                self.phase_note = "thinking".into();
                return KeyAction::Submit(text);
            }
            KeyCode::Backspace => {
                if self.input_cursor > 0 {
                    // Remove char before cursor
                    self.clear_input_history_nav();
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
                    self.clear_input_history_nav();
                    let before: String = self.input.chars().take(ci).collect();
                    let after: String = self.input.chars().skip(ci + 1).collect();
                    self.input = format!("{before}{after}");
                    self.normalize_slash_selection();
                }
            }
            KeyCode::Up => {
                self.history_prev();
            }
            KeyCode::Down => {
                self.history_next();
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
            KeyCode::Char('t') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return KeyAction::ToggleTasks;
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
        self.replace_slash_token(&entries[idx].command)
    }

    fn should_complete_slash_on_enter(&self) -> bool {
        let Some((_, _, token)) = self.slash_token_span() else {
            return false;
        };
        let normalized = token.trim().to_lowercase();
        if normalized == "/" {
            return true;
        }
        !self.is_known_slash_command(&normalized)
    }

    fn is_known_slash_command(&self, token: &str) -> bool {
        self.slash_commands
            .iter()
            .any(|spec| spec.command.eq_ignore_ascii_case(token))
    }

    fn replace_slash_token(&mut self, replacement: &str) -> bool {
        let Some((start, end, _)) = self.slash_token_span() else {
            return false;
        };
        self.clear_input_history_nav();
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

    fn slash_score(query: &str, spec: &SlashCommand, order: usize) -> Option<i32> {
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
