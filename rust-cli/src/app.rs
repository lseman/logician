use chrono::Local;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::{layout::Rect, Frame};
use serde_json::{Map, Value};
use std::fmt;
use uuid::Uuid;

use crate::bridge::{BridgeEvent, BridgeState};
use crate::image::ImageRenderer;
use crate::markdown::{render_diff, render_markdown, render_streaming, strip_think_blocks};

// ── Types ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelFocus {
    Messages,
    Input,
    Image,
    Todo,
    ToolOutput,
    Context,
    SlashPopup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    Thinking,
    System,
    Tool,
    Decision,
    Repair,
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
            Phase::Bubbling => write!(f, "planning"),
            Phase::Jambering => write!(f, "running"),
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
        if streaming {
            let (label, label_color) = match role {
                Role::Assistant => ("assistant    ", Color::Green),
                Role::Thinking => ("thinking     ", Color::Yellow),
                _ => ("stream       ", Color::Cyan),
            };
            return Line::from(vec![
                Span::styled(
                    label,
                    Style::default()
                        .fg(label_color)
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
            Role::Thinking => ("thinking     ", Color::Yellow),
            Role::System => ("system       ", Color::Blue),
            Role::Tool => ("tool ⚙       ", Color::Cyan),
            Role::Decision => ("decision ⎇   ", Color::LightBlue),
            Role::Repair => ("repair ⟳     ", Color::LightYellow),
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
            Role::Thinking if streaming => render_streaming(text),
            Role::Thinking => text
                .lines()
                .map(|l| {
                    Line::from(Span::styled(
                        l.to_string(),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::DIM),
                    ))
                })
                .collect(),
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
            Role::Tool => {
                let mut tool_lines = Vec::new();
                let is_error = Self::tool_message_is_error(text);
                let emphasis = if is_error { Color::Red } else { Color::Cyan };
                for (idx, line) in text.lines().enumerate() {
                    let (prefix, style) = if idx == 0 {
                        (
                            "  ▸ ",
                            Style::default().fg(emphasis).add_modifier(Modifier::BOLD),
                        )
                    } else if is_error && line.trim_start().starts_with("error:") {
                        (
                            "    ",
                            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
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
            Role::Decision => {
                let mut decision_lines = Vec::new();
                for (idx, line) in text.lines().enumerate() {
                    let (prefix, style) = if idx == 0 {
                        (
                            "  ◇ ",
                            Style::default()
                                .fg(Color::LightBlue)
                                .add_modifier(Modifier::BOLD),
                        )
                    } else {
                        (
                            "    ",
                            Style::default().fg(Color::Gray).add_modifier(Modifier::DIM),
                        )
                    };
                    decision_lines.push(Line::from(vec![
                        Span::styled(prefix.to_string(), Style::default().fg(Color::DarkGray)),
                        Span::styled(line.to_string(), style),
                    ]));
                }
                decision_lines
            }
            Role::Repair => {
                let mut repair_lines = Vec::new();
                let lower = text.to_lowercase();
                let emphasis = if lower.contains("failed") || lower.contains("invalid") {
                    Color::LightRed
                } else if lower.contains("repaired") {
                    Color::LightGreen
                } else {
                    Color::LightYellow
                };
                for (idx, line) in text.lines().enumerate() {
                    let (prefix, style) = if idx == 0 {
                        (
                            "  ↺ ",
                            Style::default().fg(emphasis).add_modifier(Modifier::BOLD),
                        )
                    } else {
                        (
                            "    ",
                            Style::default().fg(Color::Gray).add_modifier(Modifier::DIM),
                        )
                    };
                    repair_lines.push(Line::from(vec![
                        Span::styled(prefix.to_string(), Style::default().fg(Color::DarkGray)),
                        Span::styled(line.to_string(), style),
                    ]));
                }
                repair_lines
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

    fn new_pre_rendered(role: Role, text: impl Into<String>, body: Vec<Line<'static>>) -> Self {
        let text = text.into();
        let renderer = DefaultRenderer;
        let mut rendered = vec![renderer.render_header(role, false)];
        rendered.extend(body);
        rendered.push(Line::raw(""));
        Self {
            id: Uuid::new_v4().to_string(),
            role,
            is_streaming: false,
            text,
            raw_stream: None,
            rendered,
        }
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
    /// Cancel the in-flight request (Escape or Ctrl+C while busy)
    Interrupt,
    /// Text that needs to go to the bridge (non-slash chat or slash)
    Submit(String),
    /// Toggle trace output
    ToggleTrace,
    /// Toggle raw stream display
    ToggleRawStream,
    /// Toggle task panel
    ToggleTasks,
    /// Toggle context explorer panel
    ToggleContextExplorer,
    /// Toggle tool output expansion (Ctrl+R)
    ToggleToolOutput,
}

#[derive(Clone, Copy)]
pub struct ShortcutHint {
    pub chord: &'static str,
    pub description: &'static str,
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

pub const SHORTCUT_HINTS: [ShortcutHint; 14] = [
    ShortcutHint {
        chord: "Ctrl+O",
        description: "trace in inspector",
    },
    ShortcutHint {
        chord: "Ctrl+P",
        description: "toggle raw stream",
    },
    ShortcutHint {
        chord: "Ctrl+T",
        description: "toggle task panel",
    },
    ShortcutHint {
        chord: "Ctrl+E",
        description: "toggle context panel",
    },
    ShortcutHint {
        chord: "Ctrl+R",
        description: "toggle inspector",
    },
    ShortcutHint {
        chord: "Ctrl+L",
        description: "clear transcript",
    },
    ShortcutHint {
        chord: "Ctrl+Q",
        description: "quit",
    },
    ShortcutHint {
        chord: "Ctrl+C",
        description: "interrupt, clear input, or quit",
    },
    ShortcutHint {
        chord: "Ctrl+K",
        description: "clear input line",
    },
    ShortcutHint {
        chord: "Ctrl+W",
        description: "delete previous word",
    },
    ShortcutHint {
        chord: "Ctrl+←",
        description: "previous word",
    },
    ShortcutHint {
        chord: "Ctrl+→",
        description: "next word",
    },
    ShortcutHint {
        chord: "Shift+Enter",
        description: "insert newline",
    },
    ShortcutHint {
        chord: "Alt+Enter",
        description: "insert newline",
    },
];

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

#[derive(Clone)]
pub struct ChangedFile {
    pub tool: String,
    pub path: String,
    pub diff: String,
    pub expanded: bool,
}

#[derive(Clone, Default)]
struct UiHitboxes {
    messages: Option<Rect>,
    input: Option<Rect>,
    image: Option<Rect>,
    todo: Option<Rect>,
    tool_output: Option<Rect>,
    context: Option<Rect>,
    slash_popup: Option<Rect>,
    slash_popup_items: Vec<Rect>,
    change_headers: Vec<(usize, Rect)>,
    todo_viewport_h: u16,
    todo_content_h: u16,
    tool_output_viewport_h: u16,
    tool_output_content_h: u16,
    context_viewport_h: u16,
    context_content_h: u16,
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
    // Store tool arguments for expanded output
    current_turn_tool_args: Vec<Value>,
    last_turn_tool_names: Vec<String>,
    last_turn_iterations: u64,
    current_turn_tool_errors: u64,
    total_tool_errors: u64,
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
    pub decision_mode: String,
    pub decision_stage: String,
    pub last_tool_error: Option<String>,
    pub busy: bool,
    pub spinner_tick: usize,

    // Feature toggles
    pub trace_on: bool,
    pub raw_on: bool,         // show raw stream
    pub tool_output_on: bool, // show expanded tool output panel
    pub context_on: bool,     // show context explorer panel

    // Bridge state
    pub bridge_state: BridgeState,
    pub connected: bool,

    pub should_quit: bool,

    // Trace buffer (not rendered into messages when trace_on=false)
    pub trace_log: Vec<String>,
    pub todo_on: bool,
    active_skill_ids: Vec<String>,
    active_selected_tools: Vec<String>,

    // Expanded tool output buffer (for Ctrl+R toggle panel)
    tool_output_buffer: Vec<String>,
    changed_files: Vec<ChangedFile>,
    current_change_idx: usize,
    focused_panel: PanelFocus,
    todo_scroll: u16,
    tool_output_scroll: u16,
    context_scroll: u16,
    ui_hitboxes: UiHitboxes,
}

impl App {
    fn sanitize_assistant_text(text: &str) -> String {
        strip_think_blocks(text)
    }

    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            cached_lines: Vec::new(),
            cached_lines_raw: Vec::new(),
            live: None,
            raw_buf: String::new(),
            current_turn_tool_names: Vec::new(),
            current_turn_tool_args: Vec::new(),
            last_turn_tool_names: Vec::new(),
            last_turn_iterations: 0,
            current_turn_tool_errors: 0,
            total_tool_errors: 0,
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
            decision_mode: String::new(),
            decision_stage: String::new(),
            last_tool_error: None,
            busy: false,
            spinner_tick: 0,
            trace_on: true,
            raw_on: false,
            tool_output_on: false,
            context_on: false,
            bridge_state: BridgeState::default(),
            connected: false,
            should_quit: false,
            trace_log: Vec::new(),
            todo_on: false,
            active_skill_ids: Vec::new(),
            active_selected_tools: Vec::new(),
            tool_output_buffer: Vec::new(),
            changed_files: Vec::new(),
            current_change_idx: 0,
            focused_panel: PanelFocus::Input,
            todo_scroll: 0,
            tool_output_scroll: 0,
            context_scroll: 0,
            ui_hitboxes: UiHitboxes::default(),
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
        self.apply_bridge_state(state);
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
        let mut text = format!(
            "# Logician\n\n\
**Agents**: {agents}  \n\
**MCPs**: {mcps}  \n\
**Rapidfuzz**: {}  \n\
**Tiktoken**: {}  \n\n\
Active: `{}` · Session: `{}` · Ctrl+O trace in Inspector · Ctrl+P raw stream · Ctrl+E context · Ctrl+C quit",
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

        // ── Project memory summary ────────────────────────────────────────────
        if let Some(mem) = v.get("memory_summary") {
            let has_memories = mem
                .get("has_memories")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if has_memories {
                let total = mem
                    .get("total_entries")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                text.push_str(&format!("\n\n---\n\n**Memory** ({total} observations)"));
                if let Some(sections) = mem.get("sections").and_then(|v| v.as_array()) {
                    for section in sections {
                        let heading = section
                            .get("heading")
                            .and_then(|h| h.as_str())
                            .unwrap_or("Other");
                        if let Some(entries) = section.get("entries").and_then(|e| e.as_array()) {
                            if !entries.is_empty() {
                                text.push_str(&format!("\n\n**{heading}**"));
                                for entry in entries {
                                    if let Some(s) = entry.as_str() {
                                        // Strip markdown link syntax [name](file): desc → name: desc
                                        let display = Self::format_memory_entry(s);
                                        text.push_str(&format!("\n- {display}"));
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                text.push_str("\n\n---\n\n*No project memories yet — use `write_file` to save observations to `.logician/memory/`.*");
            }
        }

        self.add_message(Role::System, text);
    }

    fn format_memory_entry(entry: &str) -> String {
        // Convert "[name](file.md): description" → "name: description"
        // Also handle plain "- text" entries
        if entry.starts_with('[') {
            if let Some(close_bracket) = entry.find("](") {
                let name = &entry[1..close_bracket];
                if let Some(paren_end) = entry[close_bracket..].find(')') {
                    let after = entry[close_bracket + paren_end + 1..]
                        .trim_start_matches(':')
                        .trim();
                    if after.is_empty() {
                        return name.to_string();
                    }
                    return format!("{name}: {after}");
                }
            }
        }
        entry.to_string()
    }

    // ── Message helpers ───────────────────────────────────────────────────────

    pub fn add_message(&mut self, role: Role, text: impl Into<String>) {
        self.push_message(Message::new(role, text));
        if self.at_bottom {
            self.scroll_to_bottom();
        }
    }

    pub fn add_rendered_message(
        &mut self,
        role: Role,
        text: impl Into<String>,
        body: Vec<Line<'static>>,
    ) {
        self.push_message(Message::new_pre_rendered(role, text, body));
        if self.at_bottom {
            self.scroll_to_bottom();
        }
    }

    pub fn add_system_message(&mut self, text: impl Into<String>) {
        self.add_message(Role::System, text);
    }

    pub fn add_preformatted_system_message(&mut self, text: impl Into<String>) {
        let text = text.into();
        let body = text
            .split('\n')
            .map(|line| {
                Line::from(Span::styled(
                    line.to_string(),
                    Style::default().fg(Color::White),
                ))
            })
            .collect::<Vec<_>>();
        self.add_rendered_message(Role::System, text, body);
    }

    pub fn trace_entries(&self) -> &[String] {
        &self.trace_log
    }

    pub fn active_skill_ids(&self) -> &[String] {
        &self.active_skill_ids
    }

    pub fn active_selected_tools(&self) -> &[String] {
        &self.active_selected_tools
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

    pub fn route_label(&self) -> String {
        match (
            self.decision_mode.trim().is_empty(),
            self.decision_stage.trim().is_empty(),
        ) {
            (true, _) => "-".to_string(),
            (false, true) => self.decision_mode.clone(),
            (false, false) => format!("{}/{}", self.decision_mode, self.decision_stage),
        }
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
        for (count, ch) in text.chars().enumerate() {
            if count >= max_chars {
                out.push('…');
                return out;
            }
            out.push(ch);
        }
        out
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

    fn compact_inline_text(text: &str, max_chars: usize) -> String {
        let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
        Self::truncate_inline(&compact, max_chars)
    }

    fn summarize_result_preview(result_preview: Option<&str>, max_chars: usize) -> Option<String> {
        let trimmed = result_preview?.trim();
        if trimmed.is_empty() {
            return None;
        }

        if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(trimmed) {
            let mut parts = Vec::new();
            for key in [
                "summary",
                "message",
                "path",
                "file_path",
                "lines",
                "line_count",
                "matches",
                "count",
                "file_count",
                "chunks_added",
                "rows",
            ] {
                if let Some(value) = map.get(key) {
                    match value {
                        Value::String(text) if !text.trim().is_empty() => {
                            parts.push(format!(
                                "{key}={}",
                                Self::compact_inline_text(text, max_chars / 2)
                            ));
                        }
                        Value::Number(num) => parts.push(format!("{key}={num}")),
                        Value::Bool(flag) => parts.push(format!("{key}={flag}")),
                        _ => {}
                    }
                }
                if parts.len() >= 3 {
                    break;
                }
            }
            if !parts.is_empty() {
                return Some(Self::truncate_inline(&parts.join(" · "), max_chars));
            }
        }

        Some(Self::compact_inline_text(trimmed, max_chars))
    }

    fn should_render_slash_result_preformatted(text: &str) -> bool {
        let trimmed = text.trim();
        !trimmed.is_empty()
            && trimmed.contains("## System Prompt")
            && trimmed.contains("## Message Window")
    }

    fn format_tool_event_text(name: &str, args: &Value, sequence: usize) -> String {
        let mut lines = vec![format!("call #{sequence} `{name}`")];

        match args {
            Value::Object(map) => {
                lines.push(format!("arguments: {} key(s)", map.len()));
                if map.is_empty() {
                    lines.push("• none".to_string());
                } else {
                    for (key, value) in map.iter().take(4) {
                        lines.push(format!("• {key} = {}", Self::summarize_json_value(value)));
                    }
                    if map.len() > 4 {
                        lines.push(format!("• ... +{} more key(s)", map.len() - 4));
                    }
                }
            }
            _ => lines.push(format!("arguments: {}", Self::summarize_json_value(args))),
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
        result_preview: Option<&str>,
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
        if normalized == "ok" {
            if let Some(summary) = Self::summarize_result_preview(result_preview, 120) {
                lines.push(format!("result: {summary}"));
            }
        }
        if let Some(err) = error {
            let short = Self::truncate_inline(err.trim(), 220);
            if !short.is_empty() {
                lines.push(format!("error: {short}"));
            }
        }
        lines.join("\n")
    }

    fn truncate_json_strings(value: &Value, max_chars: usize) -> Value {
        match value {
            Value::String(text) => Value::String(Self::truncate_inline(text, max_chars)),
            Value::Array(items) => Value::Array(
                items
                    .iter()
                    .map(|item| Self::truncate_json_strings(item, max_chars))
                    .collect(),
            ),
            Value::Object(map) => Value::Object(
                map.iter()
                    .map(|(key, value)| {
                        (key.clone(), Self::truncate_json_strings(value, max_chars))
                    })
                    .collect::<Map<String, Value>>(),
            ),
            _ => value.clone(),
        }
    }

    fn parse_tool_preview_value(text: &str, max_chars: usize) -> Value {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Value::Null;
        }

        if let Ok(parsed) = serde_json::from_str::<Value>(trimmed) {
            return Self::truncate_json_strings(&parsed, max_chars);
        }

        Value::String(Self::truncate_inline(trimmed, max_chars))
    }

    /// Format tool output with proper JSON pretty-printing for multi-line display
    fn format_tool_expanded_text(
        name: &str,
        args: &Value,
        status: &str,
        duration_ms: u64,
        error: Option<&str>,
        result_preview: Option<&str>,
    ) -> String {
        let mut payload = Map::new();
        payload.insert("tool".to_string(), Value::String(name.to_string()));
        payload.insert(
            "arguments".to_string(),
            Self::truncate_json_strings(args, 180),
        );
        payload.insert("status".to_string(), Value::String(status.to_string()));
        payload.insert("duration_ms".to_string(), Value::from(duration_ms));

        if let Some(err) = error.map(str::trim).filter(|value| !value.is_empty()) {
            payload.insert(
                "error".to_string(),
                Value::String(Self::truncate_inline(err, 800)),
            );
        }

        if let Some(result) = result_preview
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            payload.insert(
                "result".to_string(),
                Self::parse_tool_preview_value(result, 300),
            );
        }

        serde_json::to_string_pretty(&Value::Object(payload))
            .unwrap_or_else(|_| format!("{{\"tool\":\"{name}\",\"status\":\"{status}\"}}"))
    }

    fn format_tool_repair_text(
        stage: &str,
        attempt: u64,
        tool: &str,
        error_type: &str,
        message: &str,
    ) -> String {
        let stage_l = stage.trim().to_lowercase();
        let label = match stage_l.as_str() {
            "invalid" => "invalid tool call detected",
            "attempt" => "attempting tool-call repair",
            "repaired" => "tool call repaired",
            "failed" => "tool-call repair failed",
            "nudge" => "tool-call correction requested",
            _ => "tool-call repair update",
        };

        let mut lines = vec![if attempt > 0 {
            format!("{label} for `{tool}` (attempt #{attempt})")
        } else {
            format!("{label} for `{tool}`")
        }];

        if !error_type.trim().is_empty() {
            lines.push(format!(
                "error_type: {}",
                Self::truncate_inline(error_type.trim(), 120)
            ));
        }
        if !message.trim().is_empty() {
            lines.push(Self::truncate_inline(message.trim(), 260));
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

    fn format_decision_event_text(mode: &str, stage: &str, message: &str) -> String {
        let route = if mode.trim().is_empty() {
            "decision"
        } else {
            mode.trim()
        };
        let stage = if stage.trim().is_empty() {
            "update"
        } else {
            stage.trim()
        };
        let mut lines = vec![format!("{route}: {stage}")];
        if !message.trim().is_empty() {
            lines.push(Self::truncate_inline(message.trim(), 220));
        }
        lines.join("\n")
    }

    fn remember_changed_file(&mut self, tool: String, path: String, diff: String) {
        let normalized_path = path.trim().to_string();
        if normalized_path.is_empty() || diff.trim().is_empty() {
            return;
        }

        if let Some(existing_idx) = self
            .changed_files
            .iter()
            .position(|item| item.path == normalized_path)
        {
            self.changed_files.remove(existing_idx);
        }

        self.changed_files.push(ChangedFile {
            tool,
            path: normalized_path,
            diff,
            expanded: true,
        });
        self.current_change_idx = self.changed_files.len().saturating_sub(1);
        self.tool_output_on = true;
        self.focused_panel = PanelFocus::ToolOutput;
    }

    fn extract_tool_names_from_chat_result(v: &Value) -> Vec<String> {
        let out: Vec<String> = v["tool_calls"]
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
        let reported_count = reported_tool_names.len();
        let observed_count = self.current_turn_tool_names.len();
        let runtime_tools = if !reported_tool_names.is_empty() {
            reported_tool_names
        } else {
            self.current_turn_tool_names.clone()
        };
        self.last_turn_tool_names = runtime_tools.clone();
        self.last_turn_iterations = v["iterations"].as_u64().unwrap_or(0);
        // Use reported tool_errors from Python; fall back to events tracked here
        let reported_errors = v["tool_errors"]
            .as_u64()
            .unwrap_or(self.current_turn_tool_errors);
        self.total_tool_errors = self
            .total_tool_errors
            .saturating_sub(self.current_turn_tool_errors)
            .saturating_add(reported_errors);
        if !runtime_tools.is_empty()
            && (runtime_tools.len() > 1 || self.last_turn_iterations > 1 || reported_errors > 0)
        {
            let mut uniq = Vec::<String>::new();
            for name in &runtime_tools {
                if !uniq.iter().any(|existing| existing == name) {
                    uniq.push(name.clone());
                }
            }
            let preview = Self::summarize_name_list(&uniq, 6);
            let mut receipt = format!(
                "Activity summary: {} tool{}",
                runtime_tools.len(),
                if runtime_tools.len() == 1 { "" } else { "s" }
            );
            if self.last_turn_iterations > 1 {
                receipt.push_str(&format!(" across {} iterations", self.last_turn_iterations));
            }
            if preview != "none" {
                receipt.push_str(&format!(" · {preview}"));
            }
            if reported_errors > 0 {
                receipt.push_str(&format!(" · errors={reported_errors}"));
            }
            self.add_system_message(receipt);
        }

        if observed_count > reported_count && (reported_count > 0 || observed_count > 0) {
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
        self.current_turn_tool_args.clear();
        self.current_turn_tool_errors = 0;
    }

    fn trace(&mut self, text: impl Into<String>) {
        let line = format!("{}  {}", Local::now().format("%H:%M:%S"), text.into());
        self.trace_log.push(line);
        const TRACE_LOG_LIMIT: usize = 400;
        if self.trace_log.len() > TRACE_LOG_LIMIT {
            let overflow = self.trace_log.len() - TRACE_LOG_LIMIT;
            self.trace_log.drain(0..overflow);
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
        self.current_turn_tool_errors = 0;
        self.total_tool_errors = 0;
        self.active_skill_ids.clear();
        self.active_selected_tools.clear();
        self.tool_output_buffer.clear();
        self.changed_files.clear();
        self.current_change_idx = 0;
        self.clear_image_preview();
        self.scroll_top = 0;
        self.at_bottom = true;
        self.tool_output_on = false;
        self.todo_scroll = 0;
        self.tool_output_scroll = 0;
        self.context_scroll = 0;
        self.decision_mode.clear();
        self.decision_stage.clear();
        self.focused_panel = PanelFocus::Input;
    }

    pub fn tool_output_buffer(&self) -> &[String] {
        &self.tool_output_buffer
    }

    pub fn changed_files(&self) -> &[ChangedFile] {
        &self.changed_files
    }

    pub fn current_changed_file(&self) -> Option<&ChangedFile> {
        self.changed_files.get(
            self.current_change_idx
                .min(self.changed_files.len().saturating_sub(1)),
        )
    }

    pub fn is_panel_focused(&self, panel: PanelFocus) -> bool {
        self.focused_panel == panel
    }

    pub fn todo_scroll(&self) -> u16 {
        self.todo_scroll
    }

    pub fn tool_output_scroll(&self) -> u16 {
        self.tool_output_scroll
    }

    pub fn context_scroll(&self) -> u16 {
        self.context_scroll
    }

    pub fn has_change_records(&self) -> bool {
        !self.changed_files.is_empty()
    }

    pub fn has_context_records(&self) -> bool {
        !self.bridge_state.active_repos.is_empty()
            || !self.bridge_state.retrieval_insights.is_empty()
            || !self.bridge_state.repo_library.is_empty()
            || !self.bridge_state.mounted_paths.is_empty()
            || !self.bridge_state.rag_docs.is_empty()
    }

    pub fn context_token_estimate_total(&self) -> u64 {
        self.bridge_state
            .mounted_paths
            .iter()
            .map(|item| item.token_count)
            .sum::<u64>()
            + self
                .bridge_state
                .rag_docs
                .iter()
                .map(|item| item.token_count)
                .sum::<u64>()
    }

    pub fn has_panel_content(&self) -> bool {
        self.has_change_records()
            || !self.tool_output_buffer.is_empty()
            || (self.trace_on && !self.trace_log.is_empty())
            || !self.active_skill_ids.is_empty()
    }

    pub fn begin_frame(&mut self) {
        self.ui_hitboxes = UiHitboxes::default();
    }

    pub fn register_messages_area(&mut self, area: Rect) {
        self.ui_hitboxes.messages = Some(area);
    }

    pub fn register_input_area(&mut self, area: Rect) {
        self.ui_hitboxes.input = Some(area);
    }

    pub fn register_image_area(&mut self, area: Rect) {
        self.ui_hitboxes.image = Some(area);
    }

    pub fn register_todo_area(&mut self, area: Rect, viewport_h: u16, content_h: u16) {
        self.ui_hitboxes.todo = Some(area);
        self.ui_hitboxes.todo_viewport_h = viewport_h;
        self.ui_hitboxes.todo_content_h = content_h;
        self.clamp_aux_scroll();
    }

    pub fn register_tool_output_area(
        &mut self,
        area: Rect,
        viewport_h: u16,
        content_h: u16,
        change_headers: Vec<(usize, Rect)>,
    ) {
        self.ui_hitboxes.tool_output = Some(area);
        self.ui_hitboxes.tool_output_viewport_h = viewport_h;
        self.ui_hitboxes.tool_output_content_h = content_h;
        self.ui_hitboxes.change_headers = change_headers;
        self.clamp_aux_scroll();
    }

    pub fn register_context_area(&mut self, area: Rect, viewport_h: u16, content_h: u16) {
        self.ui_hitboxes.context = Some(area);
        self.ui_hitboxes.context_viewport_h = viewport_h;
        self.ui_hitboxes.context_content_h = content_h;
        self.clamp_aux_scroll();
    }

    pub fn register_slash_popup(&mut self, area: Rect, item_rects: Vec<Rect>) {
        self.ui_hitboxes.slash_popup = Some(area);
        self.ui_hitboxes.slash_popup_items = item_rects;
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

    fn clamp_aux_scroll(&mut self) {
        self.todo_scroll = self.todo_scroll.min(
            self.ui_hitboxes
                .todo_content_h
                .saturating_sub(self.ui_hitboxes.todo_viewport_h),
        );
        self.tool_output_scroll = self.tool_output_scroll.min(
            self.ui_hitboxes
                .tool_output_content_h
                .saturating_sub(self.ui_hitboxes.tool_output_viewport_h),
        );
        self.context_scroll = self.context_scroll.min(
            self.ui_hitboxes
                .context_content_h
                .saturating_sub(self.ui_hitboxes.context_viewport_h),
        );
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

    fn rect_contains(rect: Rect, column: u16, row: u16) -> bool {
        column >= rect.x
            && column < rect.x.saturating_add(rect.width)
            && row >= rect.y
            && row < rect.y.saturating_add(rect.height)
    }

    fn focus_panel_at(&mut self, column: u16, row: u16) -> Option<PanelFocus> {
        if let Some(idx) = self
            .ui_hitboxes
            .slash_popup_items
            .iter()
            .position(|rect| Self::rect_contains(*rect, column, row))
        {
            self.focused_panel = PanelFocus::SlashPopup;
            self.slash_selected = idx;
            return Some(PanelFocus::SlashPopup);
        }

        let panel = if self
            .ui_hitboxes
            .slash_popup
            .is_some_and(|rect| Self::rect_contains(rect, column, row))
        {
            Some(PanelFocus::SlashPopup)
        } else if self
            .ui_hitboxes
            .input
            .is_some_and(|rect| Self::rect_contains(rect, column, row))
        {
            Some(PanelFocus::Input)
        } else if self
            .ui_hitboxes
            .image
            .is_some_and(|rect| Self::rect_contains(rect, column, row))
        {
            Some(PanelFocus::Image)
        } else if self
            .ui_hitboxes
            .todo
            .is_some_and(|rect| Self::rect_contains(rect, column, row))
        {
            Some(PanelFocus::Todo)
        } else if self
            .ui_hitboxes
            .tool_output
            .is_some_and(|rect| Self::rect_contains(rect, column, row))
        {
            Some(PanelFocus::ToolOutput)
        } else if self
            .ui_hitboxes
            .context
            .is_some_and(|rect| Self::rect_contains(rect, column, row))
        {
            Some(PanelFocus::Context)
        } else if self
            .ui_hitboxes
            .messages
            .is_some_and(|rect| Self::rect_contains(rect, column, row))
        {
            Some(PanelFocus::Messages)
        } else {
            None
        };

        if let Some(panel) = panel {
            self.focused_panel = panel;
        }
        panel
    }

    pub fn handle_mouse_click(&mut self, column: u16, row: u16) {
        if let Some(idx) = self
            .ui_hitboxes
            .slash_popup_items
            .iter()
            .position(|rect| Self::rect_contains(*rect, column, row))
        {
            self.focused_panel = PanelFocus::SlashPopup;
            self.slash_selected = idx;
            let _ = self.apply_selected_slash_command();
            self.focused_panel = PanelFocus::Input;
            return;
        }

        if let Some(file_idx) = self
            .ui_hitboxes
            .change_headers
            .iter()
            .find(|(_, rect)| Self::rect_contains(*rect, column, row))
            .map(|(file_idx, _)| *file_idx)
        {
            self.focused_panel = PanelFocus::ToolOutput;
            self.toggle_changed_file(file_idx);
            return;
        }

        let _ = self.focus_panel_at(column, row);
    }

    pub fn handle_mouse_scroll(&mut self, column: u16, row: u16, up: bool) {
        let _ = self.focus_panel_at(column, row);
        let amount = 3;

        match self.focused_panel {
            PanelFocus::Todo if self.ui_hitboxes.todo_viewport_h > 0 => {
                self.scroll_aux_panel(PanelFocus::Todo, amount, up);
            }
            PanelFocus::ToolOutput if self.ui_hitboxes.tool_output_viewport_h > 0 => {
                self.scroll_aux_panel(PanelFocus::ToolOutput, amount, up);
            }
            PanelFocus::Context if self.ui_hitboxes.context_viewport_h > 0 => {
                self.scroll_aux_panel(PanelFocus::Context, amount, up);
            }
            PanelFocus::SlashPopup if self.slash_popup_visible() => {
                self.move_slash_selection(if up { -1 } else { 1 });
            }
            _ => {
                if up {
                    self.scroll_up(amount);
                } else {
                    self.scroll_down(amount);
                }
            }
        }
    }

    fn scroll_aux_panel(&mut self, panel: PanelFocus, amount: u16, up: bool) {
        let (offset, content_h, viewport_h) = match panel {
            PanelFocus::Todo => (
                &mut self.todo_scroll,
                self.ui_hitboxes.todo_content_h,
                self.ui_hitboxes.todo_viewport_h,
            ),
            PanelFocus::ToolOutput => (
                &mut self.tool_output_scroll,
                self.ui_hitboxes.tool_output_content_h,
                self.ui_hitboxes.tool_output_viewport_h,
            ),
            PanelFocus::Context => (
                &mut self.context_scroll,
                self.ui_hitboxes.context_content_h,
                self.ui_hitboxes.context_viewport_h,
            ),
            _ => return,
        };

        if up {
            *offset = offset.saturating_sub(amount);
        } else {
            let max_offset = content_h.saturating_sub(viewport_h);
            *offset = offset.saturating_add(amount).min(max_offset);
        }
    }

    fn toggle_changed_file(&mut self, idx: usize) {
        if let Some(item) = self.changed_files.get_mut(idx) {
            self.current_change_idx = idx;
            item.expanded = !item.expanded;
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
                    let text = Self::sanitize_assistant_text(&text);
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
        if live.role == Role::Assistant {
            live.text = Self::sanitize_assistant_text(&live.text);
        }
        if live.text.is_empty() {
            return;
        }

        live.finalize_streaming();
        if live.role == Role::Assistant {
            live.raw_stream = Some(std::mem::take(&mut self.raw_buf));
        }
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

            BridgeEvent::ThinkingToken(tok) => {
                self.busy = true;
                self.phase = Phase::Thinking;
                self.phase_note = "pre-turn plan".into();
                match &mut self.live {
                    Some(lm) if lm.role == Role::Thinking => lm.append_stream_chunk(&tok),
                    Some(_) => {
                        self.stop_live_message();
                        let mut lm = Message::new_streaming(Role::Thinking);
                        lm.append_stream_chunk(&tok);
                        self.live = Some(lm);
                    }
                    None => {
                        let mut lm = Message::new_streaming(Role::Thinking);
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
                self.current_turn_tool_args
                    .push(args.clone().unwrap_or_default());
                self.add_message(
                    Role::Tool,
                    Self::format_tool_event_text(&name, &args.unwrap_or_default(), call_number),
                );
                self.phase = Phase::Jambering;
                self.phase_note = format!("running {name}");
            }

            BridgeEvent::ToolEnd {
                name,
                sequence,
                status,
                duration_ms,
                cache_hit,
                error,
                result_preview,
                args,
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
                        result_preview.as_deref(),
                    ),
                );
                // Add expanded output to toggle panel - use args from event
                let expanded = Self::format_tool_expanded_text(
                    &name,
                    args.as_ref()
                        .unwrap_or(&Value::Object(serde_json::Map::new())),
                    &status,
                    duration_ms,
                    error.as_deref(),
                    result_preview.as_deref(),
                );
                self.tool_output_buffer.push(expanded);
                if status_l == "error" || status_l == "failed" {
                    let detail = error
                        .as_deref()
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(|value| Self::truncate_inline(value, 220))
                        .unwrap_or_else(|| "tool execution failed".to_string());
                    self.last_tool_error = Some(format!("{name}: {detail}"));
                    self.current_turn_tool_errors += 1;
                    self.total_tool_errors += 1;
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
                let skill_text = Self::format_skill_event_text(&skill_ids, &selected_tools);
                self.active_skill_ids = skill_ids;
                self.active_selected_tools = selected_tools;
                self.add_message(Role::Skill, skill_text);
            }

            BridgeEvent::Decision {
                mode,
                stage,
                message,
            } => {
                self.stop_live_message();
                self.trace(format!("decision mode={} stage={}", mode, stage));
                let decision_text = Self::format_decision_event_text(&mode, &stage, &message);
                self.decision_mode = mode;
                self.decision_stage = stage;
                self.add_message(Role::Decision, decision_text);
                if !message.trim().is_empty() {
                    self.trace(format!(
                        "decision_note={}",
                        Self::truncate_inline(&message, 160)
                    ));
                }
            }

            BridgeEvent::ToolRepair {
                stage,
                attempt,
                tool,
                error_type,
                message,
            } => {
                self.stop_live_message();
                self.trace(format!(
                    "tool_repair stage={} tool={} attempt={}",
                    stage, tool, attempt
                ));
                if stage == "attempt" {
                    self.phase = Phase::Thinking;
                    self.phase_note = format!("repairing {tool}");
                }
                self.add_message(
                    Role::Repair,
                    Self::format_tool_repair_text(&stage, attempt, &tool, &error_type, &message),
                );
            }

            BridgeEvent::Stderr(text) => {
                let short = &text[..text.len().min(220)];
                self.trace(format!("stderr={short}"));
                // Surface bridge warnings/errors so they're visible without trace mode.
                let lower = text.to_ascii_lowercase();
                if lower.contains("error")
                    || lower.contains("warning")
                    || lower.contains("traceback")
                    || lower.contains("exception")
                {
                    self.add_system_message(format!("[bridge] {}", &text[..text.len().min(300)]));
                }
            }
            BridgeEvent::FileDiff { tool, path, diff } => {
                self.stop_live_message();
                self.trace(format!("file_diff tool={tool} path={path}"));
                self.remember_changed_file(tool.clone(), path.clone(), diff);
                let mut lines = Vec::new();
                lines.push(Line::from(vec![
                    Span::styled("  ▸ ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("changed via `{tool}`"),
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]));
                lines.push(Line::raw(""));
                lines.extend(render_diff(
                    &path,
                    &self
                        .current_changed_file()
                        .map(|item| item.diff.clone())
                        .unwrap_or_default(),
                ));
                self.add_rendered_message(Role::Tool, format!("inline diff for {path}"), lines);
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
                let assistant_text = v["final_response"]
                    .as_str()
                    .or_else(|| v["assistant"].as_str())
                    .unwrap_or("")
                    .to_string();
                let visible_assistant_text = Self::sanitize_assistant_text(&assistant_text);
                let fallback =
                    (!visible_assistant_text.is_empty()).then_some(visible_assistant_text.clone());
                self.flush_live_message(fallback);

                // Pipeline turns
                if v["pipeline"].as_bool().unwrap_or(false) {
                    if let Some(turns) = v["turns"].as_array() {
                        for t in turns {
                            let agent = t["agent"].as_str().unwrap_or("?");
                            let text =
                                Self::sanitize_assistant_text(t["text"].as_str().unwrap_or(""));
                            if !text.trim().is_empty() {
                                self.add_message(Role::Assistant, format!("[{agent}] {text}"));
                            }
                        }
                    }
                }

                if let Some(state) = v.get("state") {
                    self.apply_bridge_state(state);
                }
                self.finalize_turn_receipt(&v, &visible_assistant_text);
            }
            Err(e) => {
                self.live = None;
                self.raw_buf.clear();
                self.current_turn_tool_names.clear();
                self.current_turn_tool_errors = 0;
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
                    let text = msgs
                        .iter()
                        .filter_map(|m| m.as_str())
                        .collect::<Vec<_>>()
                        .join("\n");
                    if !text.trim().is_empty() {
                        if Self::should_render_slash_result_preformatted(&text) {
                            self.add_preformatted_system_message(text);
                        } else {
                            self.add_system_message(text);
                        }
                    }
                }
                self.install_bridge_commands(&v);
                if let Some(state) = v.get("state") {
                    self.apply_bridge_state(state);
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
        self.decision_mode.clear();
        self.decision_stage.clear();
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
            if next {
                self.tool_output_on = true;
                self.focused_panel = PanelFocus::ToolOutput;
            }
            self.add_system_message(format!(
                "Trace {}{}.",
                if next { "enabled" } else { "disabled" },
                if next { " in Inspector" } else { "" }
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
        lines.push("## Keyboard Shortcuts".to_string());
        for item in self.shortcut_hints() {
            lines.push(format!("- `{}`  {}", item.chord, item.description));
        }
        lines.push("- `Enter`  submit input".to_string());
        lines.push("- `Shift+Enter` or `Alt+Enter`  insert newline".to_string());
        lines.push("- `Up/Down`  move across input lines, then history at top/bottom".to_string());
        lines.push("- `PgUp/PgDn`  scroll transcript".to_string());
        lines.push("- `Esc`  interrupt when busy, otherwise clear input".to_string());
        lines.push("- `Tab`  accept slash command or insert spaces".to_string());
        lines.push("- `Shift+Drag`  select text".to_string());
        lines.join("\n")
    }

    pub fn shortcut_hints(&self) -> &'static [ShortcutHint] {
        &SHORTCUT_HINTS
    }

    // ── Status ────────────────────────────────────────────────────────────────

    #[allow(dead_code)]
    pub fn update_bridge_state(&mut self, v: &Value) {
        self.apply_bridge_state(v);
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
            "active: {}\nsession: {}\nmessages: {}\nagents: {}\nmcp: {}\npipeline: {}\nrapidfuzz: {}\ntiktoken: {}\ntrace: {}\nactive_repos: {}\nretrievals: {}\nrepo_library: {}\nmounts: {}\nrag_docs: {}\nskills: {}\ncontext_tokens≈{}",
            s.active, session, s.msg_count, agents, mcps, pipeline,
            if s.rapidfuzz { "enabled" } else { "disabled" },
            if s.tiktoken { "enabled" } else { "disabled" },
            if self.trace_on { "on" } else { "off" },
            s.active_repos.len(),
            s.retrieval_insights.len(),
            s.repo_library.len(),
            s.mounted_paths.len(),
            s.rag_docs.len(),
            Self::summarize_name_list(&s.loaded_skills, 8),
            self.context_token_estimate_total(),
        ));
    }

    fn apply_bridge_state(&mut self, v: &Value) {
        let previous_todo_count = self.bridge_state.todo.len();
        let previous_context_count = self.bridge_state.active_repos.len()
            + self.bridge_state.retrieval_insights.len()
            + self.bridge_state.repo_library.len()
            + self.bridge_state.mounted_paths.len()
            + self.bridge_state.rag_docs.len();
        self.bridge_state = BridgeState::from_value(v);
        if previous_todo_count == 0 && !self.bridge_state.todo.is_empty() {
            self.todo_on = true;
        }
        if previous_context_count == 0 && self.has_context_records() {
            self.context_on = true;
        }
    }

    // ── Spinner tick ──────────────────────────────────────────────────────────

    pub fn tick(&mut self) {
        if self.busy {
            self.spinner_tick = (self.spinner_tick + 1) % 10;
        }
    }

    pub fn toggle_trace(&mut self) {
        self.trace_on = !self.trace_on;
        if self.trace_on {
            self.tool_output_on = true;
            self.focused_panel = PanelFocus::ToolOutput;
        }
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
        if self.todo_on {
            self.focused_panel = PanelFocus::Todo;
        }
        self.trace(format!("todo_panel={}", self.todo_on));
    }

    pub fn toggle_context_explorer(&mut self) {
        self.context_on = !self.context_on;
        if self.context_on {
            self.focused_panel = PanelFocus::Context;
        }
        self.trace(format!("context_panel={}", self.context_on));
    }

    pub fn toggle_tool_output(&mut self) {
        if !self.has_panel_content() {
            self.tool_output_on = false;
            return;
        }
        self.tool_output_on = !self.tool_output_on;
        if self.tool_output_on {
            self.focused_panel = PanelFocus::ToolOutput;
        }
        self.trace(format!("tool_output_panel={}", self.tool_output_on));
    }

    pub fn handle_paste(&mut self, text: &str) {
        if self.busy || text.is_empty() {
            return;
        }
        let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
        if normalized.is_empty() {
            return;
        }
        self.focused_panel = PanelFocus::Input;
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

    fn reset_input(&mut self) {
        self.input.clear();
        self.input_cursor = 0;
        self.slash_selected = 0;
    }

    fn push_input_history(&mut self, text: &str) {
        if text.trim().is_empty() {
            return;
        }
        self.input_history.push(text.to_string());
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

    fn input_char_len(&self) -> usize {
        self.input.chars().count()
    }

    pub fn input_line_count(&self) -> usize {
        self.input.split('\n').count().max(1)
    }

    pub fn input_cursor_line_col(&self) -> (usize, usize) {
        let mut row = 0usize;
        let mut col = 0usize;
        for ch in self.input.chars().take(self.input_cursor) {
            if ch == '\n' {
                row += 1;
                col = 0;
            } else {
                col += 1;
            }
        }
        (row, col)
    }

    fn line_start_before(&self, cursor: usize) -> usize {
        let chars: Vec<char> = self.input.chars().collect();
        let mut pos = cursor.min(chars.len());
        while pos > 0 && chars[pos - 1] != '\n' {
            pos -= 1;
        }
        pos
    }

    fn line_end_after(&self, cursor: usize) -> usize {
        let chars: Vec<char> = self.input.chars().collect();
        let mut pos = cursor.min(chars.len());
        while pos < chars.len() && chars[pos] != '\n' {
            pos += 1;
        }
        pos
    }

    fn move_cursor_up_line(&mut self) -> bool {
        let chars: Vec<char> = self.input.chars().collect();
        let cursor = self.input_cursor.min(chars.len());
        let line_start = self.line_start_before(cursor);
        if line_start == 0 {
            return false;
        }

        let current_col = cursor - line_start;
        let prev_line_end = line_start - 1;
        let prev_line_start = self.line_start_before(prev_line_end);
        let prev_line_len = prev_line_end - prev_line_start;
        self.input_cursor = prev_line_start + current_col.min(prev_line_len);
        true
    }

    fn move_cursor_down_line(&mut self) -> bool {
        let chars: Vec<char> = self.input.chars().collect();
        let cursor = self.input_cursor.min(chars.len());
        let line_end = self.line_end_after(cursor);
        if line_end >= chars.len() {
            return false;
        }

        let line_start = self.line_start_before(cursor);
        let current_col = cursor - line_start;
        let next_line_start = line_end + 1;
        let next_line_end = self.line_end_after(next_line_start);
        let next_line_len = next_line_end - next_line_start;
        self.input_cursor = next_line_start + current_col.min(next_line_len);
        true
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
        let tools_chip = if self.total_tool_errors > 0 {
            format!(
                "{} ({}x err)",
                self.bridge_state.tool_count, self.total_tool_errors
            )
        } else {
            self.bridge_state.tool_count.to_string()
        };
        format!(
            "{}{} · {} · agent:{} · msgs:{} · tools:{} · skills:{} · last_turn_tools:{} · route:{} · image:{} · trace:{} · raw:{} · Ctrl+O trace · Ctrl+P raw · Ctrl+T tasks · Ctrl+R inspector",
            spin,
            self.phase,
            self.phase_note,
            self.bridge_state.active,
            self.bridge_state.msg_count,
            tools_chip,
            self.active_skill_ids.len(),
            self.last_turn_tool_names.len(),
            self.route_label(),
            if self.has_image_preview() { "on" } else { "off" },
            if self.trace_on { "on" } else { "off" },
            if self.raw_on { "on" } else { "off" },
        )
    }

    pub fn last_turn_tool_count(&self) -> usize {
        self.last_turn_tool_names.len()
    }

    // ── Interrupt / cancel ────────────────────────────────────────────────────

    /// Called when the user presses Escape or Ctrl+C while the agent is busy.
    /// Marks the app as no longer busy immediately so the user can type again.
    /// The Python bridge finishes its current turn in background; its response
    /// is silently discarded once we've dropped the oneshot receiver.
    pub fn handle_interrupt(&mut self) {
        self.busy = false;
        self.phase = Phase::Ready;
        self.phase_note = "ready".into();
        self.add_system_message("Interrupted.".to_string());
    }

    // ── Word navigation helpers ───────────────────────────────────────────────

    /// Returns the char-index of the start of the word before `cursor`.
    fn word_start_before(&self, cursor: usize) -> usize {
        let chars: Vec<char> = self.input.chars().collect();
        let mut pos = cursor;
        // skip whitespace immediately before cursor
        while pos > 0 && chars[pos - 1].is_whitespace() {
            pos -= 1;
        }
        // skip word characters
        while pos > 0 && !chars[pos - 1].is_whitespace() {
            pos -= 1;
        }
        pos
    }

    /// Returns the char-index after the end of the word starting at `cursor`.
    fn word_end_after(&self, cursor: usize) -> usize {
        let chars: Vec<char> = self.input.chars().collect();
        let len = chars.len();
        let mut pos = cursor;
        // skip whitespace
        while pos < len && chars[pos].is_whitespace() {
            pos += 1;
        }
        // skip word characters
        while pos < len && !chars[pos].is_whitespace() {
            pos += 1;
        }
        pos
    }

    // ── Keyboard handling ─────────────────────────────────────────────────────

    pub fn handle_key(&mut self, key: KeyEvent) -> KeyAction {
        // Global shortcuts (work even when busy)
        if key.modifiers == KeyModifiers::CONTROL {
            match key.code {
                KeyCode::Char('c') => {
                    if self.busy {
                        // First press while busy → cancel
                        return KeyAction::Interrupt;
                    }
                    // When idle: clear input if non-empty, otherwise quit
                    if !self.input.is_empty() {
                        self.reset_input();
                        self.clear_input_history_nav();
                        return KeyAction::None;
                    }
                    self.should_quit = true;
                    return KeyAction::Quit;
                }
                KeyCode::Char('k') => {
                    // Clear input line (readline-style Ctrl+K)
                    if !self.busy {
                        self.reset_input();
                        self.clear_input_history_nav();
                    }
                    return KeyAction::None;
                }
                KeyCode::Char('w') => {
                    // Delete word before cursor (readline-style Ctrl+W)
                    if !self.busy && self.input_cursor > 0 {
                        let new_cursor = self.word_start_before(self.input_cursor);
                        let before: String = self.input.chars().take(new_cursor).collect();
                        let after: String = self.input.chars().skip(self.input_cursor).collect();
                        self.input = format!("{before}{after}");
                        self.input_cursor = new_cursor;
                        self.clear_input_history_nav();
                    }
                    return KeyAction::None;
                }
                KeyCode::Backspace => {
                    // Ctrl+Backspace → delete word before cursor
                    if !self.busy && self.input_cursor > 0 {
                        let new_cursor = self.word_start_before(self.input_cursor);
                        let before: String = self.input.chars().take(new_cursor).collect();
                        let after: String = self.input.chars().skip(self.input_cursor).collect();
                        self.input = format!("{before}{after}");
                        self.input_cursor = new_cursor;
                        self.clear_input_history_nav();
                    }
                    return KeyAction::None;
                }
                KeyCode::Left => {
                    // Ctrl+Left → jump to start of previous word
                    if !self.busy {
                        self.input_cursor = self.word_start_before(self.input_cursor);
                    }
                    return KeyAction::None;
                }
                KeyCode::Right => {
                    // Ctrl+Right → jump to end of next word
                    if !self.busy {
                        self.input_cursor = self.word_end_after(self.input_cursor);
                    }
                    return KeyAction::None;
                }
                KeyCode::Char('o') => {
                    return KeyAction::ToggleTrace;
                }
                KeyCode::Char('p') => {
                    return KeyAction::ToggleRawStream;
                }
                KeyCode::Char('e') => {
                    return KeyAction::ToggleContextExplorer;
                }
                KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    return KeyAction::ToggleToolOutput;
                }
                KeyCode::Char('l') => {
                    self.clear_transcript();
                    return KeyAction::None;
                }
                _ => {}
            }
        }

        // Escape: cancel when busy, clear input when idle
        if key.code == KeyCode::Esc && key.modifiers.is_empty() {
            if self.busy {
                return KeyAction::Interrupt;
            }
            if !self.input.is_empty() {
                self.reset_input();
                self.clear_input_history_nav();
            }
            return KeyAction::None;
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

        if matches!(
            key.code,
            KeyCode::Enter
                | KeyCode::Backspace
                | KeyCode::Delete
                | KeyCode::Up
                | KeyCode::Down
                | KeyCode::Left
                | KeyCode::Right
                | KeyCode::Home
                | KeyCode::End
                | KeyCode::Char(_)
                | KeyCode::Tab
                | KeyCode::BackTab
        ) {
            self.focused_panel = PanelFocus::Input;
        }

        // Input editing
        match key.code {
            KeyCode::Enter => {
                if key
                    .modifiers
                    .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT)
                {
                    self.insert_input_text("\n");
                    return KeyAction::None;
                }
                if !key.modifiers.is_empty() {
                    return KeyAction::None;
                }
                if self.should_complete_slash_on_enter() && self.apply_selected_slash_command() {
                    return KeyAction::None;
                }
                let raw_text = self.input.clone();
                if raw_text.trim().is_empty() {
                    return KeyAction::None;
                }
                let submit_text = if raw_text.trim_start().starts_with('/') {
                    raw_text.trim().to_string()
                } else {
                    raw_text
                };
                self.push_input_history(&submit_text);
                self.reset_input();
                self.last_tool_error = None;
                if submit_text.starts_with('/') && self.handle_local_slash(&submit_text) {
                    return KeyAction::None;
                }
                self.current_turn_tool_names.clear();
                self.current_turn_tool_errors = 0;
                self.decision_mode.clear();
                self.decision_stage.clear();
                self.active_skill_ids.clear();
                self.active_selected_tools.clear();
                self.add_message(Role::User, submit_text.clone());
                self.busy = true;
                self.phase = Phase::Thinking;
                self.phase_note = "thinking".into();
                return KeyAction::Submit(submit_text);
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
                if !self.move_cursor_up_line() {
                    self.history_prev();
                }
            }
            KeyCode::Down => {
                if !self.move_cursor_down_line() {
                    self.history_next();
                }
            }
            KeyCode::Left => {
                self.input_cursor = self.input_cursor.saturating_sub(1);
            }
            KeyCode::Right => {
                let total = self.input_char_len();
                if self.input_cursor < total {
                    self.input_cursor += 1;
                }
            }
            KeyCode::Home => {
                self.input_cursor = self.line_start_before(self.input_cursor);
            }
            KeyCode::End => {
                self.input_cursor = self.line_end_after(self.input_cursor);
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
            KeyCode::Char('t') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return KeyAction::ToggleTasks;
            }
            KeyCode::Char('e') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return KeyAction::ToggleContextExplorer;
            }
            KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if !self.has_panel_content() {
                    self.tool_output_on = false;
                } else {
                    self.tool_output_on = !self.tool_output_on;
                }
                return KeyAction::None;
            }
            _ => {}
        }
        KeyAction::None
    }

    /// Character index (count of chars, not bytes) at cursor.
    fn char_idx(&self) -> usize {
        self.input_cursor
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

    // ── Unified diff generator for file edits ──────────────────────────────────
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn slash_result_keeps_multiline_output_together() {
        let mut app = App::new();
        app.handle_slash_result(Ok(json!({
            "messages": ["alpha", "beta", "gamma"],
            "state": {},
        })));

        assert_eq!(app.messages.len(), 1);
        assert_eq!(app.messages[0].role, Role::System);
        assert_eq!(app.messages[0].text, "alpha\nbeta\ngamma");
    }

    #[test]
    fn slash_result_applies_loaded_skills_from_state() {
        let mut app = App::new();
        app.handle_slash_result(Ok(json!({
            "messages": [],
            "state": {
                "active": "main",
                "session": "sess-1",
                "loaded_tools": ["read_file", "edit_file"],
                "skill_count": 2,
                "loaded_skills": ["explore", "patch"],
            },
        })));

        assert_eq!(app.bridge_state.skill_count, 2);
        assert_eq!(
            app.bridge_state.loaded_tools,
            vec!["read_file".to_string(), "edit_file".to_string()]
        );
        assert_eq!(
            app.bridge_state.loaded_skills,
            vec!["explore".to_string(), "patch".to_string()]
        );
    }

    #[test]
    fn trace_stays_out_of_transcript() {
        let mut app = App::new();
        app.trace("phase=thinking");
        assert!(app.messages.is_empty());
        assert_eq!(app.trace_entries().len(), 1);
    }

    #[test]
    fn skill_event_updates_inspector_state_without_transcript_noise() {
        let mut app = App::new();
        app.handle_bridge_event(BridgeEvent::Skill {
            skill_ids: vec!["explore".to_string()],
            selected_tools: vec!["read_file".to_string(), "rg_search".to_string()],
        });

        assert_eq!(app.messages.len(), 1);
        assert_eq!(app.messages[0].role, Role::Skill);
        assert_eq!(app.active_skill_ids(), &["explore".to_string()]);
        assert_eq!(
            app.active_selected_tools(),
            &["read_file".to_string(), "rg_search".to_string()]
        );
    }
}
