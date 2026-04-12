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
use crate::markdown::{
    render_markdown, render_streaming,
    sanitize_assistant_text as sanitize_assistant_markdown_text,
    sanitize_thinking_text as sanitize_thinking_stream_text,
};

// ── Types ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelFocus {
    Messages,
    Input,
    Image,
    Todo,
    ToolOutput,
    Context,
    Rag,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgressState {
    Pending,
    Active,
    Complete,
    Error,
}

impl ProgressState {
    pub fn color(self) -> Color {
        match self {
            ProgressState::Pending => Color::Gray,
            ProgressState::Active => Color::Cyan,
            ProgressState::Complete => Color::Green,
            ProgressState::Error => Color::Red,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgressEntry {
    pub label: String,
    pub meta: Option<String>,
    pub detail: Option<String>,
    pub state: ProgressState,
    pub is_last: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveStepSummary {
    pub text: String,
    pub state: ProgressState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivityState {
    Running,
    Complete,
    Error,
}

impl ActivityState {
    pub fn color(self) -> Color {
        match self {
            ActivityState::Running => Color::Cyan,
            ActivityState::Complete => Color::Green,
            ActivityState::Error => Color::Red,
        }
    }
}

impl From<ActivityState> for ProgressState {
    fn from(value: ActivityState) -> Self {
        match value {
            ActivityState::Running => ProgressState::Active,
            ActivityState::Complete => ProgressState::Complete,
            ActivityState::Error => ProgressState::Error,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivityEntry {
    pub sequence: usize,
    pub name: String,
    pub summary: Option<String>,
    pub detail: Option<String>,
    pub status: ActivityState,
    pub duration_ms: Option<u64>,
    pub cache_hit: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CachedLineMeta {
    message_idx: usize,
    thinking_toggle: bool,
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
    pub collapsed: bool,
    pub thinking_label_index: Option<usize>,
    pub thinking_label_ticks: usize,
    /// Pre-rendered lines (updated on text change)
    pub rendered: Vec<Line<'static>>,
}

trait MessageRenderer {
    fn render_header(&self, role: Role, streaming: bool, collapsed: bool) -> Line<'static>;
    fn render_body(
        &self,
        role: Role,
        text: &str,
        streaming: bool,
        collapsed: bool,
        thinking_label_index: Option<usize>,
    ) -> Vec<Line<'static>>;

    fn render(
        &self,
        role: Role,
        text: &str,
        streaming: bool,
        collapsed: bool,
        thinking_label_index: Option<usize>,
    ) -> Vec<Line<'static>> {
        let mut lines = vec![self.render_header(role, streaming, collapsed)];
        lines.extend(self.render_body(role, text, streaming, collapsed, thinking_label_index));
        lines.push(Line::raw(""));
        lines
    }
}

struct DefaultRenderer;

impl DefaultRenderer {
    fn role_style(role: Role) -> (Color, &'static str, &'static str) {
        match role {
            Role::User => (Color::Yellow, "●", "you"),
            Role::Assistant => (Color::Green, "●", "assistant"),
            Role::Thinking => (Color::Yellow, "◌", "thinking"),
            Role::System => (Color::Blue, "●", "system"),
            Role::Tool => (Color::Cyan, "●", "tool"),
            Role::Decision => (Color::LightBlue, "◆", "decision"),
            Role::Repair => (Color::LightYellow, "↺", "repair"),
            Role::Skill => (Color::Magenta, "◆", "skill"),
        }
    }

    fn prefix_lines(
        lines: Vec<Line<'static>>,
        prefix: &'static str,
        prefix_style: Style,
    ) -> Vec<Line<'static>> {
        lines
            .into_iter()
            .map(|line| {
                if line.spans.is_empty() {
                    return line;
                }

                let mut spans = Vec::with_capacity(line.spans.len() + 1);
                spans.push(Span::styled(prefix.to_string(), prefix_style));
                spans.extend(line.spans);
                Line::from(spans)
            })
            .collect()
    }

    fn tool_message_is_error(text: &str) -> bool {
        let lower = text.trim().to_lowercase();
        lower.starts_with("failed ")
            || lower.starts_with("error:")
            || lower.contains("\nerror:")
            || lower.contains(" status=error")
    }

    const THINKING_VERBS: &[&str] = &[
        "Accomplishing",
        "Actioning",
        "Actualizing",
        "Architecting",
        "Baking",
        "Beaming",
        "Beboppin'",
        "Befuddling",
        "Billowing",
        "Blanching",
        "Bloviating",
        "Boogieing",
        "Boondoggling",
        "Booping",
        "Bootstrapping",
        "Brewing",
        "Bunning",
        "Burrowing",
        "Calculating",
        "Canoodling",
        "Caramelizing",
        "Cascading",
        "Catapulting",
        "Cerebrating",
        "Channeling",
        "Channelling",
        "Choreographing",
        "Churning",
        "Clauding",
        "Coalescing",
        "Cogitating",
        "Combobulating",
        "Composing",
        "Computing",
        "Concocting",
        "Considering",
        "Contemplating",
        "Cooking",
        "Crafting",
        "Creating",
        "Crunching",
        "Crystallizing",
        "Cultivating",
        "Deciphering",
        "Deliberating",
        "Determining",
        "Dilly-dallying",
        "Discombobulating",
        "Doing",
        "Doodling",
        "Drizzling",
        "Ebbing",
        "Effecting",
        "Elucidating",
        "Embellishing",
        "Enchanting",
        "Envisioning",
        "Evaporating",
        "Fermenting",
        "Fiddle-faddling",
        "Finagling",
        "Flambéing",
        "Flibbertigibbeting",
        "Flowing",
        "Flummoxing",
        "Fluttering",
        "Forging",
        "Forming",
        "Frolicking",
        "Frosting",
        "Gallivanting",
        "Galloping",
        "Garnishing",
        "Generating",
        "Gesticulating",
        "Germinating",
        "Gitifying",
        "Grooving",
        "Gusting",
        "Harmonizing",
        "Hashing",
        "Hatching",
        "Herding",
        "Honking",
        "Hullaballooing",
        "Hyperspacing",
        "Ideating",
        "Imagining",
        "Improvising",
        "Incubating",
        "Inferring",
        "Infusing",
        "Ionizing",
        "Jitterbugging",
        "Julienning",
        "Kneading",
        "Leavening",
        "Levitating",
        "Lollygagging",
        "Manifesting",
        "Marinating",
        "Meandering",
        "Metamorphosing",
        "Misting",
        "Moonwalking",
        "Moseying",
        "Mulling",
        "Mustering",
        "Musing",
        "Nebulizing",
        "Nesting",
        "Newspapering",
        "Noodling",
        "Nucleating",
        "Orbiting",
        "Orchestrating",
        "Osmosing",
        "Perambulating",
        "Percolating",
        "Perusing",
        "Philosophising",
        "Photosynthesizing",
        "Pollinating",
        "Pondering",
        "Pontificating",
        "Pouncing",
        "Precipitating",
        "Prestidigitating",
        "Processing",
        "Proofing",
        "Propagating",
        "Puttering",
        "Puzzling",
        "Quantumizing",
        "Razzle-dazzling",
        "Razzmatazzing",
        "Recombobulating",
        "Reticulating",
        "Roosting",
        "Ruminating",
        "Sautéing",
        "Scampering",
        "Schlepping",
        "Scurrying",
        "Seasoning",
        "Shenaniganing",
        "Shimmying",
        "Simmering",
        "Skedaddling",
        "Sketching",
        "Slithering",
        "Smooshing",
        "Sock-hopping",
        "Spelunking",
        "Spinning",
        "Sprouting",
        "Stewing",
        "Sublimating",
        "Swirling",
        "Swooping",
        "Symbioting",
        "Synthesizing",
        "Tempering",
        "Thinking",
        "Thundering",
        "Tinkering",
        "Tomfoolering",
        "Topsy-turvying",
        "Transfiguring",
        "Transmuting",
        "Twisting",
        "Undulating",
        "Unfurling",
        "Unravelling",
        "Vibing",
        "Waddling",
        "Wandering",
        "Warping",
        "Whatchamacalliting",
        "Whirlpooling",
        "Whirring",
        "Whisking",
        "Wibbling",
        "Working",
        "Wrangling",
        "Zesting",
        "Zigzagging",
    ];

    fn thinking_category_label(text: &str, label_index: Option<usize>) -> String {
        let index = label_index.unwrap_or_else(|| {
            let seed = text
                .lines()
                .next()
                .unwrap_or("")
                .chars()
                .take(64)
                .collect::<String>();
            let hash = seed
                .bytes()
                .fold(0u64, |acc, b| acc.wrapping_mul(131).wrapping_add(b as u64));
            (hash % Self::THINKING_VERBS.len() as u64) as usize
        });
        Self::THINKING_VERBS[index].to_string()
    }

    fn thinking_streaming_body(text: &str, label_index: Option<usize>) -> Vec<Line<'static>> {
        let label = Self::thinking_category_label(text, label_index);
        vec![Line::from(vec![
            Span::styled(
                "⋮ ",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::DIM),
            ),
            Span::styled(
                label,
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::DIM),
            ),
        ])]
    }

    fn thinking_summary_text(text: &str, max_chars: usize, label_index: Option<usize>) -> String {
        let label = Self::thinking_category_label(text, label_index);
        App::truncate_inline(&label, max_chars)
    }

    fn collapsed_thinking_body(
        text: &str,
        label_index: Option<usize>,
    ) -> Vec<Line<'static>> {
        let sanitized = sanitize_thinking_stream_text(text);
        let line_count = sanitized.lines().filter(|line| !line.trim().is_empty()).count();
        let preview = Self::thinking_summary_text(&sanitized, 88, label_index);

        vec![Line::from(vec![
            Span::styled(
                "⋮ ",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::DIM),
            ),
            Span::styled(
                format!(
                    "{} hidden{}",
                    line_count.max(1),
                    if line_count == 1 { " line" } else { " lines" }
                ),
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::DIM),
            ),
            Span::styled(
                "  ",
                Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM),
            ),
            Span::styled(
                preview,
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::DIM),
            ),
        ])]
    }
}

impl MessageRenderer for DefaultRenderer {
    fn render_header(&self, role: Role, streaming: bool, collapsed: bool) -> Line<'static> {
        let (color, icon, label) = Self::role_style(role);
        if streaming {
            return Line::from(vec![
                Span::styled(
                    format!("{icon} {label}"),
                    Style::default()
                        .fg(color)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "  live",
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::DIM),
                ),
            ]);
        }
        let now = Local::now().format("%H:%M:%S").to_string();
        Line::from(vec![
            Span::styled(
                format!("{icon} {label}"),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            if role == Role::Thinking && collapsed {
                Span::styled(
                    "  collapsed",
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::DIM),
                )
            } else {
                Span::styled(
                    "  ",
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::DIM),
                )
            },
            Span::styled(
                if role == Role::Thinking && collapsed { "  " } else { "" },
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::DIM),
            ),
            Span::styled(
                now,
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::DIM),
            ),
        ])
    }

    fn render_body(
        &self,
        role: Role,
        text: &str,
        streaming: bool,
        collapsed: bool,
        thinking_label_index: Option<usize>,
    ) -> Vec<Line<'static>> {
        match role {
            Role::Assistant if streaming => Self::prefix_lines(
                render_streaming(text),
                "│ ",
                Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM),
            ),
            Role::Assistant => Self::prefix_lines(
                render_markdown(text),
                "│ ",
                Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM),
            ),
            Role::Thinking if streaming => Self::thinking_streaming_body(text, thinking_label_index),
            Role::Thinking if collapsed => Self::collapsed_thinking_body(text, thinking_label_index),
            Role::Thinking => Self::prefix_lines(
                sanitize_thinking_stream_text(text)
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
                "⋮ ",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::DIM),
            ),
            Role::System => Self::prefix_lines(
                render_markdown(text),
                "│ ",
                Style::default().fg(Color::Blue).add_modifier(Modifier::DIM),
            ),
            Role::User => Self::prefix_lines(
                text.lines()
                    .map(|l| {
                        Line::from(Span::styled(
                            l.to_string(),
                            Style::default().fg(Color::White),
                        ))
                    })
                    .collect(),
                "> ",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ),
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
        let mut rendered = vec![renderer.render_header(role, false, false)];
        rendered.extend(body);
        rendered.push(Line::raw(""));
        Self {
            id: Uuid::new_v4().to_string(),
            role,
            is_streaming: false,
            text,
            raw_stream: None,
            collapsed: false,
            thinking_label_index: None,
            thinking_label_ticks: 0,
            rendered,
        }
    }

    fn new_streaming(role: Role) -> Self {
        Self::new_with_streaming(role, String::new(), true)
    }

    fn initial_thinking_label_index(text: &str) -> usize {
        let seed = text
            .lines()
            .next()
            .unwrap_or("")
            .chars()
            .take(64)
            .collect::<String>();
        let hash = seed
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(131).wrapping_add(b as u64));
        (hash % DefaultRenderer::THINKING_VERBS.len() as u64) as usize
    }

    fn render_current(&mut self) {
        let renderer = DefaultRenderer;
        self.rendered = renderer.render(
            self.role,
            &self.text,
            self.is_streaming,
            self.collapsed,
            self.thinking_label_index,
        );
    }

    fn advance_thinking_label(&mut self) {
        if self.role != Role::Thinking || !self.is_streaming {
            return;
        }
        self.thinking_label_ticks = self.thinking_label_ticks.saturating_add(1);
        const LABEL_ROTATION_TICKS: usize = 10;
        if self.thinking_label_ticks < LABEL_ROTATION_TICKS {
            return;
        }
        self.thinking_label_ticks = 0;
        let next_index = self
            .thinking_label_index
            .unwrap_or_else(|| Self::initial_thinking_label_index(&self.text))
            .wrapping_add(1)
            % DefaultRenderer::THINKING_VERBS.len();
        self.thinking_label_index = Some(next_index);
        self.render_current();
    }

    fn new_with_streaming(role: Role, text: impl Into<String>, is_streaming: bool) -> Self {
        let text = text.into();
        let thinking_label_index = if role == Role::Thinking {
            Some(Self::initial_thinking_label_index(&text))
        } else {
            None
        };
        let renderer = DefaultRenderer;
        let collapsed = role == Role::Thinking && !is_streaming;
        let rendered = renderer.render(
            role,
            &text,
            is_streaming,
            collapsed,
            thinking_label_index,
        );
        Self {
            id: Uuid::new_v4().to_string(),
            role,
            is_streaming,
            text,
            raw_stream: None,
            collapsed,
            thinking_label_index,
            thinking_label_ticks: 0,
            rendered,
        }
    }

    fn append_stream_chunk(&mut self, chunk: &str) {
        self.text.push_str(chunk);
        self.is_streaming = true;
        self.collapsed = false;
        let renderer = DefaultRenderer;
        self.rendered = renderer.render(
            self.role,
            &self.text,
            true,
            false,
            self.thinking_label_index,
        );
    }

    fn replace_stream_text(&mut self, text: impl Into<String>) {
        self.text = text.into();
        self.is_streaming = true;
        self.collapsed = false;
        let renderer = DefaultRenderer;
        self.rendered = renderer.render(
            self.role,
            &self.text,
            true,
            false,
            self.thinking_label_index,
        );
    }

    fn finalize_streaming(&mut self) {
        self.is_streaming = false;
        if self.role == Role::Thinking {
            self.collapsed = true;
        }
        let renderer = DefaultRenderer;
        self.rendered = renderer.render(
            self.role,
            &self.text,
            false,
            self.collapsed,
            self.thinking_label_index,
        );
    }

    fn toggle_collapsed(&mut self) {
        if self.role != Role::Thinking || self.is_streaming {
            return;
        }
        self.set_collapsed(!self.collapsed);
    }

    fn set_collapsed(&mut self, collapsed: bool) {
        if self.role != Role::Thinking || self.is_streaming || self.collapsed == collapsed {
            return;
        }
        self.collapsed = collapsed;
        self.render_current();
    }

    fn line_meta(&self, message_idx: usize, line_count: usize) -> Vec<CachedLineMeta> {
        (0..line_count)
            .map(|idx| CachedLineMeta {
                message_idx,
                thinking_toggle: self.role == Role::Thinking && !self.is_streaming && idx == 0,
            })
            .collect()
    }

    fn rendered_for_raw_mode(&self) -> Vec<Line<'static>> {
        if self.role == Role::Assistant {
            if let Some(raw) = &self.raw_stream {
                let renderer = DefaultRenderer;
                return renderer.render(self.role, raw, true, false, self.thinking_label_index);
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
    /// Toggle dedicated RAG panel
    ToggleRagPanel,
    /// Toggle tool output expansion (Ctrl+R)
    ToggleToolOutput,
    /// Expand all finalized thinking messages
    ExpandThinking,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlashDispatch {
    Local,
    Bridge,
    State,
    Quit,
}

#[derive(Clone, Copy)]
pub struct ShortcutHint {
    pub chord: &'static str,
    pub description: &'static str,
}

#[derive(Clone)]
pub struct SlashPopupEntry {
    pub command: String,
    pub usage: String,
    pub description: String,
}

#[derive(Clone, Copy)]
struct SlashArgumentSpec {
    name: &'static str,
    required: bool,
    repeats: bool,
    help: &'static str,
    choices: &'static [&'static str],
}

#[derive(Clone, Copy)]
struct SlashCommandSpec {
    command: &'static str,
    usage: &'static str,
    description: &'static str,
    dispatch: SlashDispatch,
    args: &'static [SlashArgumentSpec],
}

#[derive(Clone)]
struct SlashCommand {
    command: String,
    usage: String,
    description: String,
    dispatch: SlashDispatch,
    args: Vec<SlashArgumentSpec>,
    strict_schema: bool,
}

#[derive(Debug, Clone)]
pub struct ParsedSlashCommand {
    command: String,
    usage: String,
    dispatch: SlashDispatch,
    pub positionals: Vec<String>,
    pub named_args: Map<String, Value>,
}

impl ParsedSlashCommand {
    pub fn command(&self) -> &str {
        &self.command
    }

    pub fn usage(&self) -> &str {
        &self.usage
    }

    pub fn dispatch(&self) -> SlashDispatch {
        self.dispatch
    }

    pub fn arg(&self, name: &str) -> Option<&str> {
        self.named_args.get(name).and_then(Value::as_str)
    }
}

const NO_SLASH_ARGS: &[SlashArgumentSpec] = &[];
const TRACE_SLASH_ARGS: &[SlashArgumentSpec] = &[SlashArgumentSpec {
    name: "mode",
    required: false,
    repeats: false,
    help: "Optional trace mode",
    choices: &["on", "off", "1", "0", "true", "false", "yes", "no"],
}];
const CLOSE_SLASH_ARGS: &[SlashArgumentSpec] = &[SlashArgumentSpec {
    name: "target",
    required: false,
    repeats: false,
    help: "Optional panel target",
    choices: &["all", "image", "todo", "tasks", "context", "rag", "inspector"],
}];
const SINGLE_WORD_SLASH_ARGS: &[SlashArgumentSpec] = &[SlashArgumentSpec {
    name: "value",
    required: true,
    repeats: false,
    help: "Required value",
    choices: &[],
}];
const OPTIONAL_WORD_SLASH_ARGS: &[SlashArgumentSpec] = &[SlashArgumentSpec {
    name: "value",
    required: false,
    repeats: false,
    help: "Optional value",
    choices: &[],
}];
const PATH_SLASH_ARGS: &[SlashArgumentSpec] = &[SlashArgumentSpec {
    name: "path",
    required: true,
    repeats: false,
    help: "Path argument",
    choices: &[],
}];
const PATH_AND_GLOB_SLASH_ARGS: &[SlashArgumentSpec] = &[
    SlashArgumentSpec {
        name: "path",
        required: true,
        repeats: false,
        help: "Path argument",
        choices: &[],
    },
    SlashArgumentSpec {
        name: "glob",
        required: false,
        repeats: true,
        help: "Optional glob override",
        choices: &[],
    },
];
const DOCS_SLASH_ARGS: &[SlashArgumentSpec] = &[
    SlashArgumentSpec {
        name: "library",
        required: true,
        repeats: false,
        help: "Context7 library id or package name",
        choices: &[],
    },
    SlashArgumentSpec {
        name: "topic",
        required: false,
        repeats: true,
        help: "Optional topic terms",
        choices: &[],
    },
];
const PIPELINE_SLASH_ARGS: &[SlashArgumentSpec] = &[
    SlashArgumentSpec {
        name: "from",
        required: true,
        repeats: false,
        help: "Source agent",
        choices: &[],
    },
    SlashArgumentSpec {
        name: "to",
        required: true,
        repeats: false,
        help: "Target agent",
        choices: &[],
    },
    SlashArgumentSpec {
        name: "rounds",
        required: false,
        repeats: false,
        help: "Optional round count",
        choices: &[],
    },
];

const SLASH_POPUP_LIMIT: usize = 8;
const INPUT_HISTORY_LIMIT: usize = 200;

pub const SHORTCUT_HINTS: [ShortcutHint; 16] = [
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
        chord: "Ctrl+G",
        description: "toggle RAG panel",
    },
    ShortcutHint {
        chord: "Ctrl+R",
        description: "toggle inspector",
    },
    ShortcutHint {
        chord: "Ctrl+Y",
        description: "expand thinking details",
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
        usage: "/?",
        description: "Alias for /help",
        dispatch: SlashDispatch::Local,
        args: NO_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/trace",
        usage: "/trace [on|off]",
        description: "Toggle trace messages",
        dispatch: SlashDispatch::Local,
        args: TRACE_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/clear",
        usage: "/clear",
        description: "Clear visible transcript only",
        dispatch: SlashDispatch::Local,
        args: NO_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/close",
        usage: "/close [all|image|todo|context|rag|inspector]",
        description: "Close image side panel",
        dispatch: SlashDispatch::Local,
        args: CLOSE_SLASH_ARGS,
    },
];

const SLASH_COMMANDS: [SlashCommandSpec; 29] = [
    SlashCommandSpec {
        command: "/help",
        usage: "/help",
        description: "Show command list",
        dispatch: SlashDispatch::Local,
        args: NO_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/?",
        usage: "/?",
        description: "Alias for /help",
        dispatch: SlashDispatch::Local,
        args: NO_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/version",
        usage: "/version",
        description: "Show CLI and bridge version info",
        dispatch: SlashDispatch::Local,
        args: NO_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/login",
        usage: "/login [github]",
        description: "Authenticate with GitHub Models via GitHub login",
        dispatch: SlashDispatch::Bridge,
        args: OPTIONAL_WORD_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/status",
        usage: "/status",
        description: "Show runtime state snapshot",
        dispatch: SlashDispatch::State,
        args: NO_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/skills-health",
        usage: "/skills-health",
        description: "Show skill loader diagnostics",
        dispatch: SlashDispatch::Bridge,
        args: NO_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/changes",
        usage: "/changes",
        description: "Show git status and diff preview",
        dispatch: SlashDispatch::Bridge,
        args: NO_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/trace",
        usage: "/trace [on|off]",
        description: "Toggle trace messages",
        dispatch: SlashDispatch::Local,
        args: TRACE_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/clear",
        usage: "/clear",
        description: "Clear visible transcript only",
        dispatch: SlashDispatch::Local,
        args: NO_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/close",
        usage: "/close [all|image|todo|context|rag|inspector]",
        description: "Close image side panel",
        dispatch: SlashDispatch::Local,
        args: CLOSE_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/agents",
        usage: "/agents",
        description: "List loaded agents",
        dispatch: SlashDispatch::Bridge,
        args: NO_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/agent",
        usage: "/agent <name>",
        description: "Switch active agent",
        dispatch: SlashDispatch::Bridge,
        args: SINGLE_WORD_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/pipeline",
        usage: "/pipeline <from> <to> [rounds]",
        description: "Set or stop inter-agent pipeline",
        dispatch: SlashDispatch::Bridge,
        args: PIPELINE_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/context",
        usage: "/context",
        description: "Show session/data context",
        dispatch: SlashDispatch::Bridge,
        args: NO_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/compact",
        usage: "/compact",
        description: "Summarize older conversation history",
        dispatch: SlashDispatch::Bridge,
        args: NO_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/reset",
        usage: "/reset",
        description: "Reset runtime tool state for session",
        dispatch: SlashDispatch::Bridge,
        args: NO_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/sessions",
        usage: "/sessions [query]",
        description: "List previous sessions",
        dispatch: SlashDispatch::Bridge,
        args: OPTIONAL_WORD_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/load",
        usage: "/load <session_id>",
        description: "Load a previous session",
        dispatch: SlashDispatch::Bridge,
        args: SINGLE_WORD_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/export",
        usage: "/export <path>",
        description: "Export chat history",
        dispatch: SlashDispatch::Bridge,
        args: PATH_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/upload",
        usage: "/upload <path>",
        description: "Ingest one document into RAG",
        dispatch: SlashDispatch::Bridge,
        args: PATH_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/mount",
        usage: "/mount <path> [glob]",
        description: "Mount codebase (context + RAG)",
        dispatch: SlashDispatch::Bridge,
        args: PATH_AND_GLOB_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/mount-code",
        usage: "/mount-code <path> [glob]",
        description: "Alias for /mount",
        dispatch: SlashDispatch::Bridge,
        args: PATH_AND_GLOB_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/upload-dir",
        usage: "/upload-dir <path>",
        description: "Bulk ingest docs into RAG",
        dispatch: SlashDispatch::Bridge,
        args: PATH_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/docs",
        usage: "/docs <library> [topic]",
        description: "Fetch Context7 library docs",
        dispatch: SlashDispatch::Bridge,
        args: DOCS_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/new",
        usage: "/new",
        description: "Start a new session",
        dispatch: SlashDispatch::Bridge,
        args: NO_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/reload",
        usage: "/reload",
        description: "Reload config and agents",
        dispatch: SlashDispatch::Bridge,
        args: NO_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/quit",
        usage: "/quit",
        description: "Exit CLI",
        dispatch: SlashDispatch::Quit,
        args: NO_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/exit",
        usage: "/exit",
        description: "Alias for /quit",
        dispatch: SlashDispatch::Quit,
        args: NO_SLASH_ARGS,
    },
    SlashCommandSpec {
        command: "/q",
        usage: "/q",
        description: "Alias for /quit",
        dispatch: SlashDispatch::Quit,
        args: NO_SLASH_ARGS,
    },
];

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
    thinking_headers: Vec<(usize, Rect)>,
    input: Option<Rect>,
    image: Option<Rect>,
    todo: Option<Rect>,
    tool_output: Option<Rect>,
    context: Option<Rect>,
    rag: Option<Rect>,
    slash_popup: Option<Rect>,
    slash_popup_items: Vec<Rect>,
    change_headers: Vec<(usize, Rect)>,
    todo_viewport_h: u16,
    todo_content_h: u16,
    tool_output_viewport_h: u16,
    tool_output_content_h: u16,
    context_viewport_h: u16,
    context_content_h: u16,
    rag_viewport_h: u16,
    rag_content_h: u16,
}

// ── App state ─────────────────────────────────────────────────────────────────

pub struct App {
    // Messages
    pub messages: Vec<Message>,
    // Cached flattened transcript lines (normal view / raw view).
    cached_lines: Vec<Line<'static>>,
    cached_lines_raw: Vec<Line<'static>>,
    cached_meta: Vec<CachedLineMeta>,
    cached_meta_raw: Vec<CachedLineMeta>,
    // Live streaming assistant message
    pub live: Option<Message>,
    // Raw stream buffer (for Ctrl+P view)
    pub raw_buf: String,
    // Per-turn tool execution tracking (receipt-style UX).
    current_turn_tool_names: Vec<String>,
    // Store tool arguments for expanded output
    current_turn_tool_args: Vec<Value>,
    current_turn_activity: Vec<ActivityEntry>,
    last_turn_activity: Vec<ActivityEntry>,
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
    pub rag_on: bool,         // show dedicated rag panel

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
    rag_scroll: u16,
    ui_hitboxes: UiHitboxes,
}

impl App {
    fn slash_command_from_spec(spec: SlashCommandSpec) -> SlashCommand {
        SlashCommand {
            command: spec.command.to_string(),
            usage: spec.usage.to_string(),
            description: spec.description.to_string(),
            dispatch: spec.dispatch,
            args: spec.args.to_vec(),
            strict_schema: true,
        }
    }

    fn sanitize_assistant_text(text: &str) -> String {
        sanitize_assistant_markdown_text(text)
    }

    fn sanitize_thinking_text(text: &str) -> String {
        sanitize_thinking_stream_text(text)
    }

    fn normalize_reasoning_char(ch: char) -> char {
        match ch {
            '“' | '”' => '"',
            '‘' | '’' => '\'',
            _ => ch,
        }
    }

    fn strip_duplicate_thinking_prefix(candidate: &str, thinking: &str) -> Option<String> {
        let candidate = candidate.trim_start();
        let thinking = thinking.trim();
        if candidate.is_empty() || thinking.is_empty() {
            return None;
        }

        let mut candidate_iter = candidate.char_indices().peekable();
        let mut thinking_iter = thinking.chars().peekable();
        let mut consumed = 0usize;

        while let Some(tch) = thinking_iter.next() {
            let Some((idx, cch)) = candidate_iter.next() else {
                return None;
            };
            if Self::normalize_reasoning_char(cch) != Self::normalize_reasoning_char(tch) {
                return None;
            }
            consumed = idx + cch.len_utf8();
        }

        let stripped = candidate[consumed..].trim_start().to_string();
        if stripped.is_empty() {
            None
        } else {
            Some(stripped)
        }
    }

    fn sanitize_assistant_text_with_context(&self, text: &str) -> String {
        let visible = Self::sanitize_assistant_text(text);
        let Some(last_message) = self.messages.last() else {
            return visible;
        };
        if last_message.role != Role::Thinking {
            return visible;
        }
        let thinking = Self::sanitize_thinking_text(&last_message.text);
        Self::strip_duplicate_thinking_prefix(&visible, &thinking).unwrap_or(visible)
    }

    fn repair_previous_assistant_from_thinking(&mut self, thinking_text: &str) {
        let thinking = Self::sanitize_thinking_text(thinking_text);
        if thinking.is_empty() {
            return;
        }
        let Some(previous) = self.messages.last_mut() else {
            return;
        };
        if previous.role != Role::Assistant {
            return;
        }

        let visible = Self::sanitize_assistant_text(&previous.text);
        match Self::strip_duplicate_thinking_prefix(&visible, &thinking) {
            Some(stripped) => {
                previous.text = stripped;
                if previous.is_streaming {
                    previous.replace_stream_text(previous.text.clone());
                } else {
                    let renderer = DefaultRenderer;
                    previous.rendered = renderer.render(
                        previous.role,
                        &previous.text,
                        false,
                        false,
                        previous.thinking_label_index,
                    );
                }
            }
            None if !visible.is_empty() && Self::strip_duplicate_thinking_prefix(&(visible.clone() + " "), &thinking).is_none() => {}
            None => {
                self.messages.pop();
            }
        }
    }

    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            cached_lines: Vec::new(),
            cached_lines_raw: Vec::new(),
            cached_meta: Vec::new(),
            cached_meta_raw: Vec::new(),
            live: None,
            raw_buf: String::new(),
            current_turn_tool_names: Vec::new(),
            current_turn_tool_args: Vec::new(),
            current_turn_activity: Vec::new(),
            last_turn_activity: Vec::new(),
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
            rag_on: false,
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
            rag_scroll: 0,
            ui_hitboxes: UiHitboxes::default(),
        }
    }

    fn upsert_slash_command(commands: &mut Vec<SlashCommand>, command: SlashCommand) {
        if command.command.trim().is_empty() || !command.command.starts_with('/') {
            return;
        }
        if let Some(existing) = commands
            .iter_mut()
            .find(|item| item.command.eq_ignore_ascii_case(&command.command))
        {
            if !command.description.trim().is_empty() {
                existing.description = command.description;
            }
            if !command.usage.trim().is_empty() {
                existing.usage = command.usage;
            }
            if command.strict_schema {
                existing.dispatch = command.dispatch;
                existing.args = command.args;
                existing.strict_schema = true;
            }
            return;
        }
        commands.push(command);
    }

    fn default_slash_commands() -> Vec<SlashCommand> {
        let mut out: Vec<SlashCommand> = Vec::new();
        for spec in SLASH_COMMANDS {
            Self::upsert_slash_command(&mut out, Self::slash_command_from_spec(spec));
        }
        Self::upsert_slash_command(
            &mut out,
            SlashCommand {
                command: "/repo".to_string(),
                usage: "/repo add|list|use|ingest|remove ...".to_string(),
                description: "Manage repo memory and active repo selection".to_string(),
                dispatch: SlashDispatch::Bridge,
                args: Vec::new(),
                strict_schema: false,
            },
        );
        Self::upsert_slash_command(
            &mut out,
            SlashCommand {
                command: "/rag".to_string(),
                usage: "/rag list [repo] | /rag search <query> [top_k] | /rag clear"
                    .to_string(),
                description: "Inspect the live RAG store and run direct RAG searches"
                    .to_string(),
                dispatch: SlashDispatch::Bridge,
                args: Vec::new(),
                strict_schema: false,
            },
        );
        out
    }

    fn install_bridge_commands(&mut self, v: &Value) {
        let Some(items) = v.get("commands").and_then(|value| value.as_array()) else {
            return;
        };
        let mut merged = Self::default_slash_commands();
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
            Self::upsert_slash_command(
                &mut merged,
                SlashCommand {
                    usage: command.clone(),
                    command,
                    description,
                    dispatch: SlashDispatch::Bridge,
                    args: Vec::new(),
                    strict_schema: false,
                },
            );
        }

        if merged.is_empty() {
            return;
        }

        for spec in LOCAL_ONLY_SLASH_COMMANDS {
            Self::upsert_slash_command(&mut merged, Self::slash_command_from_spec(spec));
        }
        self.slash_commands = merged;
        self.normalize_slash_selection();
    }

    fn slash_usage_error(spec: &SlashCommand, reason: &str) -> String {
        format!("{reason}\nUsage: {}", spec.usage)
    }

    pub fn parse_slash_command(&self, input: &str) -> Result<ParsedSlashCommand, String> {
        let trimmed = input.trim();
        if !trimmed.starts_with('/') {
            return Err("Slash commands must start with '/'".to_string());
        }

        let mut parts = trimmed.split_whitespace();
        let token = parts
            .next()
            .ok_or_else(|| "Slash command is empty".to_string())?
            .to_lowercase();
        let spec = self
            .slash_commands
            .iter()
            .find(|item| item.command.eq_ignore_ascii_case(&token))
            .ok_or_else(|| format!("Unknown slash command `{token}`. Use /help for the full list."))?;
        let positionals = parts.map(|item| item.to_string()).collect::<Vec<_>>();
        let mut named_args = Map::new();

        if spec.strict_schema {
            let mut index = 0usize;
            for arg in &spec.args {
                if arg.repeats {
                    let remaining = positionals[index..]
                        .iter()
                        .map(|item| Value::String(item.clone()))
                        .collect::<Vec<_>>();
                    if arg.required && remaining.is_empty() {
                        return Err(Self::slash_usage_error(
                            spec,
                            &format!("Missing required argument `{}`.", arg.name),
                        ));
                    }
                    if !remaining.is_empty() {
                        named_args.insert(arg.name.to_string(), Value::Array(remaining));
                    }
                    index = positionals.len();
                    break;
                }

                let Some(value) = positionals.get(index) else {
                    if arg.required {
                        return Err(Self::slash_usage_error(
                            spec,
                            &format!("Missing required argument `{}`.", arg.name),
                        ));
                    }
                    continue;
                };

                if !arg.choices.is_empty()
                    && !arg
                        .choices
                        .iter()
                        .any(|choice| choice.eq_ignore_ascii_case(value))
                {
                    return Err(Self::slash_usage_error(
                        spec,
                        &format!(
                            "Invalid value `{value}` for `{}`. Expected one of: {}.",
                            arg.name,
                            arg.choices.join(", ")
                        ),
                    ));
                }

                named_args.insert(arg.name.to_string(), Value::String(value.clone()));
                index += 1;
            }

            if index < positionals.len() {
                return Err(Self::slash_usage_error(
                    spec,
                    "Too many arguments provided.",
                ));
            }
        } else {
            for (index, value) in positionals.iter().enumerate() {
                named_args.insert(format!("arg{}", index + 1), Value::String(value.clone()));
            }
        }

        Ok(ParsedSlashCommand {
            command: spec.command.clone(),
            usage: spec.usage.clone(),
            dispatch: spec.dispatch,
            positionals,
            named_args,
        })
    }

    // ── Init ─────────────────────────────────────────────────────────────────

    pub fn handle_init(&mut self, v: Value) {
        self.install_bridge_commands(&v);
        let state = v.get("state").unwrap_or(&v);
        self.apply_bridge_state(state);
        self.connected = true;
        self.phase = Phase::Ready;
        self.phase_note = "ready".into();
        self.busy = false;
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
        let mcp_state = self.bridge_state.lifecycle.as_ref()
            .map_or("unknown".to_string(), |lifecycle| Self::bridge_lifecycle_state(lifecycle, "mcp"));
        let plugin_state = self.bridge_state.lifecycle.as_ref()
            .map_or("unknown".to_string(), |lifecycle| Self::bridge_lifecycle_state(lifecycle, "plugin"));
        let mut text = format!(
            "# Logician\n\n\
**Agents**: {agents}  \n\
**MCPs**: {mcps} ({mcp_state})  \n\
**Plugins**: {plugin_state}  \n\
**Rapidfuzz**: {}  \n\
**Tiktoken**: {}  \n\n\
Active: `{}` · Session: `{}` · Ctrl+O trace in Inspector · Ctrl+P raw stream · Ctrl+E context · Ctrl+G RAG · Ctrl+C quit",
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

    fn summarize_progress_args(args: &Value) -> Option<String> {
        match args {
            Value::Object(map) if !map.is_empty() => {
                let mut parts = Vec::new();
                for (key, value) in map.iter().take(2) {
                    parts.push(format!(
                        "{key}={}",
                        Self::truncate_inline(&Self::summarize_json_value(value), 28)
                    ));
                }
                if map.len() > 2 {
                    parts.push(format!("+{} more", map.len() - 2));
                }
                Some(parts.join(" · "))
            }
            Value::Null => None,
            other => Some(Self::truncate_inline(&Self::summarize_json_value(other), 48)),
        }
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

    fn extract_file_diff_from_result_output(
        tool: &str,
        args: &Value,
        result_output: Option<&str>,
    ) -> Option<(String, String)> {
        let tool_name = tool.trim();
        let is_file_tool = matches!(
            tool_name,
            "write_file"
                | "edit_file"
                | "apply_edit_block"
                | "smart_edit"
                | "edit_file_libcst"
                | "replace_function_body"
                | "replace_docstring"
                | "replace_decorators"
                | "replace_argument"
                | "insert_after_function"
                | "delete_function"
        );
        if !is_file_tool {
            return None;
        }
        let parsed = serde_json::from_str::<Value>(result_output?.trim()).ok()?;
        let diff = parsed.get("diff")?.as_str()?.trim().to_string();
        if diff.is_empty() {
            return None;
        }
        let path = parsed
            .get("path")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(ToString::to_string)
            .or_else(|| {
                args.get("path")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
            })
            .or_else(|| {
                args.get("file_path")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
            })?;
        Some((path, diff))
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

    fn bridge_lifecycle_state(lifecycle: &Value, subsystem: &str) -> String {
        lifecycle
            .get(subsystem)
            .and_then(|item| item.get("state"))
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string()
    }

    fn format_lifecycle_event_text(subsystem: &str, payload: &Value) -> String {
        let state = payload
            .get("state")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let mut lines = vec![format!("{subsystem} lifecycle: {state}")];
        if let Some(total) = payload.get("total").and_then(Value::as_u64) {
            let healthy = payload.get("healthy").and_then(Value::as_u64).unwrap_or(0);
            let degraded = payload.get("degraded").and_then(Value::as_u64).unwrap_or(0);
            let failed = payload.get("failed").and_then(Value::as_u64).unwrap_or(0);
            lines.push(format!(
                "total={total} healthy={healthy} degraded={degraded} failed={failed}"
            ));
        }
        if let Some(count) = payload.get("count").and_then(Value::as_u64) {
            lines.push(format!("count={count}"));
        }
        if let Some(names) = payload.get("names").and_then(Value::as_array) {
            let values = names
                .iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            if !values.is_empty() {
                lines.push(format!(
                    "targets: {}",
                    Self::summarize_name_list(&values, 6)
                ));
            }
        }
        lines.join("\n")
    }

    fn format_compaction_event_text(payload: &Value) -> String {
        let kept = payload.get("kept_messages").and_then(Value::as_u64);
        let removed = payload.get("removed_messages").and_then(Value::as_u64);
        let summary_chars = payload.get("summary_chars").and_then(Value::as_u64);
        let mut lines = vec!["session compaction recorded".to_string()];
        let mut stats = Vec::new();
        if let Some(value) = kept {
            stats.push(format!("kept={value}"));
        }
        if let Some(value) = removed {
            stats.push(format!("removed={value}"));
        }
        if let Some(value) = summary_chars {
            stats.push(format!("summary_chars={value}"));
        }
        if !stats.is_empty() {
            lines.push(stats.join(" "));
        }
        if let Some(reason) = payload.get("reason").and_then(Value::as_str) {
            if !reason.trim().is_empty() {
                lines.push(Self::truncate_inline(reason.trim(), 220));
            }
        }
        lines.join("\n")
    }

    fn format_summary_event_text(payload: &Value) -> String {
        let kind = payload
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("summary");
        let mut lines = vec![format!("summary event: {kind}")];
        if let Some(session_state) = payload.get("session_state").and_then(Value::as_str) {
            if !session_state.trim().is_empty() {
                lines.push(format!("session_state={session_state}"));
            }
        }
        if let Some(message) = payload.get("message").and_then(Value::as_str) {
            if !message.trim().is_empty() {
                lines.push(Self::truncate_inline(message.trim(), 220));
            }
        } else if let Some(summary) = payload.get("summary").and_then(Value::as_str) {
            if !summary.trim().is_empty() {
                lines.push(Self::truncate_inline(summary.trim(), 220));
            }
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

    fn begin_tool_activity(&mut self, sequence: usize, name: &str, args: &Value) {
        let summary = Self::summarize_progress_args(args);
        self.current_turn_activity.push(ActivityEntry {
            sequence,
            name: name.to_string(),
            summary,
            detail: None,
            status: ActivityState::Running,
            duration_ms: None,
            cache_hit: false,
        });
    }

    fn complete_tool_activity(
        &mut self,
        sequence: usize,
        name: &str,
        status: &str,
        duration_ms: u64,
        cache_hit: bool,
        error: Option<&str>,
        result_preview: Option<&str>,
        args: &Value,
    ) {
        let status_l = status.trim().to_lowercase();
        let detail = if status_l == "error" || status_l == "failed" {
            error
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(|value| Self::truncate_inline(value, 120))
        } else {
            Self::summarize_result_preview(result_preview, 96)
        };

        if let Some(entry) = self
            .current_turn_activity
            .iter_mut()
            .find(|entry| entry.sequence == sequence)
        {
            entry.status = if status_l == "error" || status_l == "failed" {
                ActivityState::Error
            } else {
                ActivityState::Complete
            };
            entry.duration_ms = Some(duration_ms);
            entry.cache_hit = cache_hit;
            entry.detail = detail;
            if entry.summary.is_none() {
                entry.summary = Self::summarize_progress_args(args);
            }
            return;
        }

        self.current_turn_activity.push(ActivityEntry {
            sequence,
            name: name.to_string(),
            summary: Self::summarize_progress_args(args),
            detail,
            status: if status_l == "error" || status_l == "failed" {
                ActivityState::Error
            } else {
                ActivityState::Complete
            },
            duration_ms: Some(duration_ms),
            cache_hit,
        });
    }

    fn live_step_text_for_entry(entry: &ActivityEntry, max_chars: usize) -> String {
        let mut parts = vec![format!("#{} {}", entry.sequence, entry.name)];

        if let Some(detail) = entry.detail.as_deref().or(entry.summary.as_deref()) {
            let trimmed = detail.trim();
            if !trimmed.is_empty() {
                parts.push(trimmed.to_string());
            }
        }

        if entry.cache_hit {
            parts.push("cached".to_string());
        }

        if let Some(duration_ms) = entry.duration_ms {
            parts.push(format!("{}ms", duration_ms));
        }

        let joined = parts.join(" · ");
        Self::truncate_inline(&joined, max_chars.max(16))
    }

    fn format_turn_receipt_text(
        entries: &[ActivityEntry],
        iterations: u64,
        error_count: u64,
    ) -> String {
        if entries.is_empty() {
            return "step receipt · no tool activity recorded".to_string();
        }

        let mut lines = Vec::new();
        let step_count = entries.len();
        let error_entries: Vec<&ActivityEntry> = entries
            .iter()
            .filter(|entry| matches!(entry.status, ActivityState::Error))
            .collect();
        let completed = step_count.saturating_sub(error_entries.len());

        let mut summary = format!(
            "step receipt · {} step{}",
            step_count,
            if step_count == 1 { "" } else { "s" }
        );
        if iterations > 1 {
            summary.push_str(&format!(" · {} iterations", iterations));
        }
        if error_count > 0 {
            summary.push_str(&format!(
                " · {} error{}",
                error_count,
                if error_count == 1 { "" } else { "s" }
            ));
        } else {
            summary.push_str(&format!(
                " · {} complete",
                completed
            ));
        }
        lines.push(summary);

        if let Some(first_error) = error_entries.first() {
            lines.push(format!(
                "error: {}",
                Self::live_step_text_for_entry(first_error, 140)
            ));
        }

        let success_entries: Vec<&ActivityEntry> = entries
            .iter()
            .filter(|entry| !matches!(entry.status, ActivityState::Error))
            .collect();
        if !success_entries.is_empty() {
            let preview = success_entries
                .iter()
                .take(2)
                .map(|entry| Self::live_step_text_for_entry(entry, 72))
                .collect::<Vec<_>>()
                .join(" · ");
            let remainder = success_entries.len().saturating_sub(2);
            if remainder > 0 {
                lines.push(format!("ok: {} · +{} more", preview, remainder));
            } else {
                lines.push(format!("ok: {}", preview));
            }
        }

        // List all unique tool names used, in first-call order.
        let mut seen: std::collections::HashSet<&str> = Default::default();
        let mut unique_tools: Vec<&str> = Vec::new();
        for entry in entries.iter() {
            let name = entry.name.trim();
            if !name.is_empty() && seen.insert(name) {
                unique_tools.push(name);
            }
        }
        if !unique_tools.is_empty() {
            lines.push(format!("tools: {}", unique_tools.join(", ")));
        }

        lines.join("\n")
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
        self.last_turn_activity = self.current_turn_activity.clone();
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
        if !self.last_turn_activity.is_empty() {
            self.add_message(
                Role::Tool,
                Self::format_turn_receipt_text(
                    &self.last_turn_activity,
                    self.last_turn_iterations,
                    reported_errors,
                ),
            );
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
        self.current_turn_activity.clear();
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
        if message.role == Role::Thinking {
            self.repair_previous_assistant_from_thinking(&message.text);
            self.messages.push(message);
            self.rebuild_cached_transcript();
            return;
        }
        let message_idx = self.messages.len();
        self.cached_lines.extend(message.rendered.iter().cloned());
        self.cached_meta
            .extend(message.line_meta(message_idx, message.rendered.len()));
        let raw_lines = message.rendered_for_raw_mode();
        self.cached_meta_raw
            .extend(message.line_meta(message_idx, raw_lines.len()));
        self.cached_lines_raw.extend(raw_lines);
        self.messages.push(message);
    }

    fn rebuild_cached_transcript(&mut self) {
        self.cached_lines.clear();
        self.cached_lines_raw.clear();
        self.cached_meta.clear();
        self.cached_meta_raw.clear();

        for (message_idx, message) in self.messages.iter().enumerate() {
            self.cached_lines.extend(message.rendered.iter().cloned());
            self.cached_meta
                .extend(message.line_meta(message_idx, message.rendered.len()));
            let raw_lines = message.rendered_for_raw_mode();
            self.cached_meta_raw
                .extend(message.line_meta(message_idx, raw_lines.len()));
            self.cached_lines_raw.extend(raw_lines);
        }
    }

    fn clear_transcript(&mut self) {
        self.messages.clear();
        self.cached_lines.clear();
        self.cached_lines_raw.clear();
        self.cached_meta.clear();
        self.cached_meta_raw.clear();
        self.live = None;
        self.raw_buf.clear();
        self.current_turn_tool_names.clear();
        self.current_turn_activity.clear();
        self.last_turn_activity.clear();
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
        self.rag_on = false;
        self.rag_scroll = 0;
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

    pub fn rag_scroll(&self) -> u16 {
        self.rag_scroll
    }

    pub fn has_change_records(&self) -> bool {
        !self.changed_files.is_empty()
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

    pub fn dashboard_tabs(&self) -> Vec<PanelFocus> {
        let mut tabs = Vec::new();
        if self.has_image_preview() {
            tabs.push(PanelFocus::Image);
        }
        if self.todo_on {
            tabs.push(PanelFocus::Todo);
        }
        if self.tool_output_on {
            tabs.push(PanelFocus::ToolOutput);
        }
        if self.context_on {
            tabs.push(PanelFocus::Context);
        }
        if self.rag_on {
            tabs.push(PanelFocus::Rag);
        }
        tabs
    }

    pub fn active_dashboard_tab(&self) -> Option<PanelFocus> {
        let visible_tabs = self.dashboard_tabs();
        if visible_tabs.is_empty() {
            return None;
        }
        if visible_tabs.contains(&self.focused_panel) {
            Some(self.focused_panel)
        } else {
            visible_tabs.into_iter().next()
        }
    }

    pub fn begin_frame(&mut self) {
        self.ui_hitboxes = UiHitboxes::default();
    }

    pub fn register_messages_area(&mut self, area: Rect, thinking_headers: Vec<(usize, Rect)>) {
        self.ui_hitboxes.messages = Some(area);
        self.ui_hitboxes.thinking_headers = thinking_headers;
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

    pub fn register_rag_area(&mut self, area: Rect, viewport_h: u16, content_h: u16) {
        self.ui_hitboxes.rag = Some(area);
        self.ui_hitboxes.rag_viewport_h = viewport_h;
        self.ui_hitboxes.rag_content_h = content_h;
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

    fn active_cached_meta(&self) -> &[CachedLineMeta] {
        if self.raw_on {
            self.cached_meta_raw.as_slice()
        } else {
            self.cached_meta.as_slice()
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
        self.rag_scroll = self.rag_scroll.min(
            self.ui_hitboxes
                .rag_content_h
                .saturating_sub(self.ui_hitboxes.rag_viewport_h),
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

    pub fn visible_transcript(&self, height: u16) -> (Vec<Line<'static>>, Vec<(usize, u16)>) {
        if height == 0 {
            return (Vec::new(), Vec::new());
        }

        let cached = self.active_cached_lines();
        let cached_meta = self.active_cached_meta();
        let cached_len = cached.len();
        let live_slice: &[Line<'static>] = self
            .live
            .as_ref()
            .map(|m| m.rendered.as_slice())
            .unwrap_or(&[]);
        let live_len = live_slice.len();
        let total = cached_len.saturating_add(live_len);
        if total == 0 {
            return (Vec::new(), Vec::new());
        }

        let max_start = total.saturating_sub(height as usize);
        let start = (self.scroll_top as usize).min(max_start);
        let end = start.saturating_add(height as usize).min(total);
        let mut out = Vec::with_capacity(end.saturating_sub(start));
        let mut thinking_headers = Vec::new();

        if start < cached_len {
            let cached_end = end.min(cached_len);
            for idx in start..cached_end {
                if cached_meta
                    .get(idx)
                    .is_some_and(|meta| meta.thinking_toggle)
                {
                    thinking_headers.push((
                        cached_meta[idx].message_idx,
                        out.len().min(u16::MAX as usize) as u16,
                    ));
                }
                out.push(cached[idx].clone());
            }
        }
        if end > cached_len {
            let live_start = start.saturating_sub(cached_len);
            let live_end = end.saturating_sub(cached_len);
            out.extend(live_slice[live_start..live_end].iter().cloned());
        }

        (out, thinking_headers)
    }

    pub fn activity_entries(&self) -> &[ActivityEntry] {
        if self.busy && !self.current_turn_activity.is_empty() {
            &self.current_turn_activity
        } else {
            &self.last_turn_activity
        }
    }

    pub fn live_step_summary(&self, max_chars: usize) -> LiveStepSummary {
        if let Some(entry) = self.activity_entries().last() {
            return LiveStepSummary {
                text: Self::live_step_text_for_entry(entry, max_chars),
                state: entry.status.into(),
            };
        }

        if self.busy {
            let text = if !self.phase_note.trim().is_empty() {
                Self::truncate_inline(&self.phase_note, max_chars.max(12))
            } else {
                self.phase.to_string()
            };
            return LiveStepSummary {
                text,
                state: match self.phase {
                    Phase::Error => ProgressState::Error,
                    _ => ProgressState::Active,
                },
            };
        }

        if !self.phase_note.trim().is_empty() && self.phase_note != "ready" {
            return LiveStepSummary {
                text: Self::truncate_inline(&self.phase_note, max_chars.max(12)),
                state: if matches!(self.phase, Phase::Error) {
                    ProgressState::Error
                } else {
                    ProgressState::Pending
                },
            };
        }

        LiveStepSummary {
            text: if self.connected {
                "waiting for input".to_string()
            } else {
                "offline".to_string()
            },
            state: if self.connected {
                ProgressState::Pending
            } else {
                ProgressState::Error
            },
        }
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
            .rag
            .is_some_and(|rect| Self::rect_contains(rect, column, row))
        {
            Some(PanelFocus::Rag)
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

        if let Some(message_idx) = self
            .ui_hitboxes
            .thinking_headers
            .iter()
            .find(|(_, rect)| Self::rect_contains(*rect, column, row))
            .map(|(message_idx, _)| *message_idx)
        {
            self.focused_panel = PanelFocus::Messages;
            self.toggle_thinking_message(message_idx);
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
            PanelFocus::Rag if self.ui_hitboxes.rag_viewport_h > 0 => {
                self.scroll_aux_panel(PanelFocus::Rag, amount, up);
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
            PanelFocus::Rag => (
                &mut self.rag_scroll,
                self.ui_hitboxes.rag_content_h,
                self.ui_hitboxes.rag_viewport_h,
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

    pub fn expand_all_thinking_messages(&mut self) {
        let mut changed = false;
        for message in self
            .messages
            .iter_mut()
            .filter(|message| message.role == Role::Thinking && !message.is_streaming)
        {
            if message.collapsed {
                message.set_collapsed(false);
                changed = true;
            }
        }

        if changed {
            self.rebuild_cached_transcript();
            if self.at_bottom {
                self.scroll_to_bottom();
            } else {
                self.clamp_scroll();
            }
        }
    }

    fn toggle_thinking_message(&mut self, message_idx: usize) {
        if let Some(message) = self.messages.get_mut(message_idx) {
            message.toggle_collapsed();
            self.rebuild_cached_transcript();
            if self.at_bottom {
                self.scroll_to_bottom();
            } else {
                self.clamp_scroll();
            }
        }
    }

    pub fn stop_live_message(&mut self) {
        self.flush_live_message(None);
    }

    fn last_transcript_message_matches(&self, role: Role, text: &str) -> bool {
        self.messages
            .iter()
            .rev()
            .find(|message| message.role != Role::Thinking)
            .map(|message| message.role == role && message.text == text)
            .unwrap_or(false)
    }

    fn flush_live_message(&mut self, fallback_text: Option<String>) {
        let mut live = match self.live.take() {
            Some(m) => m,
            None => {
                if let Some(text) = fallback_text {
                    let text = self.sanitize_assistant_text_with_context(&text);
                    if text.len() > 2
                        && text.chars().any(|c| c.is_alphanumeric())
                        && !self.last_transcript_message_matches(Role::Assistant, &text)
                    {
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
            live.text = self.sanitize_assistant_text_with_context(&live.text);
        } else if live.role == Role::Thinking {
            live.text = Self::sanitize_thinking_text(&live.text);
        }
        if live.text.is_empty() {
            return;
        }

        if live.role == Role::Assistant
            && self.last_transcript_message_matches(Role::Assistant, &live.text)
        {
            self.raw_buf.clear();
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
                let visible = self.sanitize_assistant_text_with_context(&self.raw_buf);
                match &mut self.live {
                    Some(lm) if lm.role == Role::Assistant => lm.replace_stream_text(visible),
                    Some(_) => {
                        self.stop_live_message();
                        let mut lm = Message::new_streaming(Role::Assistant);
                        lm.replace_stream_text(visible);
                        self.live = Some(lm);
                    }
                    None => {
                        let mut lm = Message::new_streaming(Role::Assistant);
                        lm.replace_stream_text(visible);
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
                self.busy = !matches!(self.phase, Phase::Ready | Phase::Error);
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
                let activity_args = args.clone().unwrap_or_default();
                self.current_turn_tool_args.push(activity_args.clone());
                self.begin_tool_activity(call_number, &name, &activity_args);
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
                result_output,
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
                self.complete_tool_activity(
                    call_number,
                    &name,
                    &status,
                    duration_ms,
                    cache_hit,
                    error.as_deref(),
                    result_preview.as_deref(),
                    args.as_ref()
                        .unwrap_or(&Value::Object(serde_json::Map::new())),
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
                if let Some((path, diff)) = Self::extract_file_diff_from_result_output(
                    &name,
                    args.as_ref()
                        .unwrap_or(&Value::Object(serde_json::Map::new())),
                    result_output.as_deref(),
                ) {
                    self.remember_changed_file(name.clone(), path, diff);
                }
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

            BridgeEvent::Lifecycle { subsystem, payload } => {
                self.stop_live_message();
                let state = payload
                    .get("state")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                self.trace(format!(
                    "lifecycle subsystem={} state={} detail={} ",
                    subsystem,
                    state,
                    Self::format_lifecycle_event_text(&subsystem, &payload)
                ));
            }

            BridgeEvent::Compaction { payload } => {
                self.stop_live_message();
                self.trace(format!(
                    "compaction event detail={}",
                    Self::format_compaction_event_text(&payload)
                ));
            }

            BridgeEvent::Summary { payload } => {
                self.stop_live_message();
                let kind = payload
                    .get("kind")
                    .and_then(Value::as_str)
                    .unwrap_or("summary");
                self.trace(format!(
                    "summary event kind={} detail={}",
                    kind,
                    Self::format_summary_event_text(&payload)
                ));
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
                let visible_assistant_text = self.sanitize_assistant_text_with_context(&assistant_text);
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
        if self.phase != Phase::Error {
            self.phase = Phase::Ready;
            self.phase_note = "ready".into();
        }
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
    pub fn handle_local_slash(&mut self, cmd: &ParsedSlashCommand) -> bool {
        let lower = cmd.command();

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
            let target = cmd.arg("target").unwrap_or("all");
            let mut closed_any = false;
            if matches!(target, "all" | "image") && self.has_image_preview() {
                self.clear_image_preview();
                self.add_system_message("Image side panel closed.");
                closed_any = true;
            }
            if matches!(target, "all" | "todo" | "tasks") && self.todo_on {
                self.todo_on = false;
                self.add_system_message("Task side panel closed.");
                closed_any = true;
            }
            if matches!(target, "all" | "context") && self.context_on {
                self.context_on = false;
                self.add_system_message("Context panel closed.");
                closed_any = true;
            }
            if matches!(target, "all" | "rag") && self.rag_on {
                self.rag_on = false;
                self.add_system_message("RAG panel closed.");
                closed_any = true;
            }
            if matches!(target, "all" | "inspector") && self.tool_output_on {
                self.tool_output_on = false;
                self.add_system_message("Inspector closed.");
                closed_any = true;
            }

            if !closed_any {
                self.add_system_message("No side panels are open to close.");
            }
            return true;
        }

        if lower == "/trace" {
            let arg = cmd.arg("mode").unwrap_or("");
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
            lines.push(format!("- `{}`  {}", item.usage, description));
            for arg in &item.args {
                let choice_suffix = if arg.choices.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", arg.choices.join("|"))
                };
                let cardinality = if arg.repeats {
                    "repeatable"
                } else if arg.required {
                    "required"
                } else {
                    "optional"
                };
                lines.push(format!(
                    "  `{}`  {}{} · {}",
                    arg.name, arg.help, choice_suffix, cardinality
                ));
            }
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
        let mcp_state = s.lifecycle.as_ref()
            .map_or("unknown".to_string(), |lifecycle| Self::bridge_lifecycle_state(lifecycle, "mcp"));
        let plugin_state = s.lifecycle.as_ref()
            .map_or("unknown".to_string(), |lifecycle| Self::bridge_lifecycle_state(lifecycle, "plugin"));
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
            "active: {}\nsession: {}\nmessages: {}\nagents: {}\nmcp: {} ({})\nplugin: {}\npipeline: {}\nrapidfuzz: {}\ntiktoken: {}\ntrace: {}\nactive_repos: {}\nretrievals: {}\nrepo_library: {}\nmounts: {}\nrag_docs: {}\nrag_chunks: {}\nrecent_rag_queries: {}\nskills: {}\ncontext_tokens≈{}",
            s.active, session, s.msg_count, agents, mcps, mcp_state, plugin_state, pipeline,
            if s.rapidfuzz { "enabled" } else { "disabled" },
            if s.tiktoken { "enabled" } else { "disabled" },
            if self.trace_on { "on" } else { "off" },
            s.active_repos.len(),
            s.retrieval_insights.len(),
            s.repo_library.len(),
            s.mounted_paths.len(),
            s.rag_docs.len(),
            s.rag_inventory.repo_chunks,
            s.recent_rag_queries.len(),
            Self::summarize_name_list(&s.loaded_skills, 8),
            self.context_token_estimate_total(),
        ));
    }

    fn apply_bridge_state(&mut self, v: &Value) {
        let previous_todo_count = self.bridge_state.todo.len();
        self.bridge_state = BridgeState::from_value(v);
        if previous_todo_count == 0 && !self.bridge_state.todo.is_empty() {
            self.todo_on = true;
        }
    }

    // ── Spinner tick ──────────────────────────────────────────────────────────

    pub fn tick(&mut self) {
        if self.busy {
            self.spinner_tick = (self.spinner_tick + 1) % 10;
            if let Some(live) = &mut self.live {
                live.advance_thinking_label();
            }
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

    pub fn toggle_rag_panel(&mut self) {
        self.rag_on = !self.rag_on;
        if self.rag_on {
            self.focused_panel = PanelFocus::Rag;
        }
        self.trace(format!("rag_panel={}", self.rag_on));
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
                            usage: spec.usage.clone(),
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
        let live_step = self.live_step_summary(72);
        let tools_chip = if self.total_tool_errors > 0 {
            format!(
                "{} ({}x err)",
                self.bridge_state.tool_count, self.total_tool_errors
            )
        } else {
            self.bridge_state.tool_count.to_string()
        };
        format!(
            "{}{} · {} · agent:{} · msgs:{} · tools:{} · skills:{} · last_turn_tools:{} · route:{} · image:{} · trace:{} · raw:{} · rag:{} · Ctrl+O trace · Ctrl+P raw · Ctrl+T tasks · Ctrl+G rag · Ctrl+R inspector",
            spin,
            self.phase,
            live_step.text,
            self.bridge_state.active,
            self.bridge_state.msg_count,
            tools_chip,
            self.active_skill_ids.len(),
            self.last_turn_tool_names.len(),
            self.route_label(),
            if self.has_image_preview() { "on" } else { "off" },
            if self.trace_on { "on" } else { "off" },
            if self.raw_on { "on" } else { "off" },
            if self.rag_on { "on" } else { "off" },
        )
    }

    pub fn last_turn_tool_count(&self) -> usize {
        self.last_turn_tool_names.len()
    }

    pub fn progress_entries(&self) -> Vec<ProgressEntry> {
        let mut entries = Vec::new();

        let live_step = self.live_step_summary(72);
        entries.push(ProgressEntry {
            label: "step".to_string(),
            meta: Some(match live_step.state {
                ProgressState::Active => "live".to_string(),
                ProgressState::Complete => "done".to_string(),
                ProgressState::Error => "error".to_string(),
                ProgressState::Pending => {
                    if self.connected {
                        "idle".to_string()
                    } else {
                        "offline".to_string()
                    }
                }
            }),
            detail: Some(live_step.text),
            state: live_step.state,
            is_last: false,
        });

        if !self.active_skill_ids.is_empty() {
            let detail = if self.active_selected_tools.is_empty() {
                Some(Self::summarize_name_list(&self.active_skill_ids, 3))
            } else {
                Some(format!(
                    "{} → {}",
                    Self::summarize_name_list(&self.active_skill_ids, 2),
                    Self::summarize_name_list(&self.active_selected_tools, 3)
                ))
            };
            entries.push(ProgressEntry {
                label: "route".to_string(),
                meta: Some(format!("{} skill(s)", self.active_skill_ids.len())),
                detail,
                state: if self.busy {
                    ProgressState::Active
                } else {
                    ProgressState::Complete
                },
                is_last: false,
            });
        }

        let live_role = self.live.as_ref().map(|message| message.role);
        if matches!(self.phase, Phase::Streaming) || matches!(live_role, Some(Role::Assistant)) {
            entries.push(ProgressEntry {
                label: "response".to_string(),
                meta: Some("streaming".to_string()),
                detail: Some(Self::truncate_inline(&self.phase_note, 56)),
                state: ProgressState::Active,
                is_last: false,
            });
        }

        if let Some(error) = self.last_tool_error.as_deref() {
            if self.busy || self.current_turn_tool_errors > 0 {
                entries.push(ProgressEntry {
                    label: "issue".to_string(),
                    meta: Some("tool error".to_string()),
                    detail: Some(Self::truncate_inline(error, 64)),
                    state: ProgressState::Error,
                    is_last: false,
                });
            }
        }

        if !self.busy && !self.last_turn_tool_names.is_empty() {
            entries.push(ProgressEntry {
                label: "last turn".to_string(),
                meta: Some(format!("{} tool(s)", self.last_turn_tool_names.len())),
                detail: Some(Self::summarize_name_list(&self.last_turn_tool_names, 4)),
                state: if self.total_tool_errors > 0 {
                    ProgressState::Error
                } else {
                    ProgressState::Complete
                },
                is_last: false,
            });
        }

        if let Some(last) = entries.last_mut() {
            last.is_last = true;
        }

        entries
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
                KeyCode::Char('g') => {
                    return KeyAction::ToggleRagPanel;
                }
                KeyCode::Char('r') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    return KeyAction::ToggleToolOutput;
                }
                KeyCode::Char('y') => {
                    return KeyAction::ExpandThinking;
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
                if submit_text.starts_with('/') {
                    match self.parse_slash_command(&submit_text) {
                        Ok(parsed) => {
                            if self.handle_local_slash(&parsed) {
                                return KeyAction::None;
                            }
                        }
                        Err(message) => {
                            self.add_system_message(message);
                            return KeyAction::None;
                        }
                    }
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
            KeyCode::Char('g') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                return KeyAction::ToggleRagPanel;
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

    #[test]
    fn assistant_tokens_do_not_continue_thinking_message() {
        let mut app = App::new();

        app.handle_bridge_event(BridgeEvent::ThinkingToken(
            "I should inspect the renderer first.".to_string(),
        ));
        app.handle_bridge_event(BridgeEvent::Token(
            "Here is the actual answer.".to_string(),
        ));

        assert_eq!(app.messages.len(), 1);
        assert_eq!(app.messages[0].role, Role::Thinking);
        assert_eq!(app.messages[0].text, "I should inspect the renderer first.");

        let live = app.live.as_ref().expect("assistant message should be live");
        assert_eq!(live.role, Role::Assistant);
        assert_eq!(live.text, "Here is the actual answer.");
    }

    #[test]
    fn phase_event_sets_busy_state_and_updates_phase_note() {
        let mut app = App::new();

        app.handle_bridge_event(BridgeEvent::Phase {
            state: "thinking".to_string(),
            note: "planning next step".to_string(),
        });

        assert!(app.busy);
        assert_eq!(app.phase, Phase::Thinking);
        assert_eq!(app.phase_note, "planning next step");
    }

    #[test]
    fn error_phase_survives_set_idle_after_failed_result() {
        let mut app = App::new();

        app.handle_chat_result(Err(anyhow::anyhow!("network failure")));

        assert!(!app.busy);
        assert_eq!(app.phase, Phase::Error);
        assert_eq!(app.phase_note, "chat failed");
    }

    #[test]
    fn finalized_thinking_is_collapsed_and_toggleable() {
        let mut app = App::new();

        app.handle_bridge_event(BridgeEvent::ThinkingToken(
            "First line of reasoning\nSecond line of reasoning".to_string(),
        ));
        app.stop_live_message();

        assert_eq!(app.messages.len(), 1);
        assert_eq!(app.messages[0].role, Role::Thinking);
        assert!(app.messages[0].collapsed);

        let (collapsed_lines, headers) = app.visible_transcript(8);
        assert_eq!(headers.len(), 1);
        assert!(collapsed_lines.iter().any(|line| line
            .spans
            .iter()
            .any(|span| span.content.contains("hidden line") || span.content.contains("hidden lines"))));

        app.toggle_thinking_message(0);

        assert!(!app.messages[0].collapsed);
        let (expanded_lines, _) = app.visible_transcript(8);
        assert!(expanded_lines.iter().any(|line| line
            .spans
            .iter()
            .any(|span| span.content.contains("Second line of reasoning"))));
    }

    #[test]
    fn ctrl_y_action_expands_all_finalized_thinking_messages() {
        let mut app = App::new();

        app.add_message(Role::Thinking, "First reasoning\nfirst detail");
        app.add_message(Role::Assistant, "Visible answer");
        app.add_message(Role::Thinking, "Second reasoning\nsecond detail");

        assert!(app.messages[0].collapsed);
        assert!(app.messages[2].collapsed);

        app.expand_all_thinking_messages();

        assert!(!app.messages[0].collapsed);
        assert!(!app.messages[2].collapsed);

        let (expanded_lines, _) = app.visible_transcript(16);
        assert!(expanded_lines.iter().any(|line| line
            .spans
            .iter()
            .any(|span| span.content.contains("first detail"))));
        assert!(expanded_lines.iter().any(|line| line
            .spans
            .iter()
            .any(|span| span.content.contains("second detail"))));
    }

    #[test]
    fn thinking_verb_label_has_no_numeric_prefix() {
        let label = DefaultRenderer::thinking_category_label("some reasoning", Some(0));

        assert_eq!(label, "Accomplishing");
        assert!(!label.chars().next().is_some_and(|ch| ch.is_ascii_digit()));
    }

    #[test]
    fn collapsed_thinking_preview_has_no_numeric_prefix() {
        let lines = DefaultRenderer::collapsed_thinking_body("some reasoning", Some(0));
        let spans = &lines[0].spans;

        assert_eq!(spans[1].content.as_ref(), "1 hidden line");
        assert_eq!(spans[3].content.as_ref(), "Accomplishing");
        assert!(!spans[3]
            .content
            .chars()
            .next()
            .is_some_and(|ch| ch.is_ascii_digit()));
    }

    #[test]
    fn tool_end_write_file_result_output_populates_changed_files() {
        let mut app = App::new();

        app.handle_bridge_event(BridgeEvent::ToolEnd {
            name: "write_file".to_string(),
            sequence: 1,
            status: "ok".to_string(),
            duration_ms: 12,
            cache_hit: false,
            error: None,
            result_preview: Some("wrote file".to_string()),
            result_output: Some(
                json!({
                    "status": "ok",
                    "path": "src/demo.py",
                    "diff": "--- src/demo.py\n+++ src/demo.py\n@@ -0,0 +1 @@\n+print(\"hello\")\n"
                })
                .to_string(),
            ),
            args: Some(json!({ "path": "src/demo.py" })),
        });

        assert_eq!(app.changed_files().len(), 1);
        assert_eq!(app.changed_files()[0].tool, "write_file");
        assert_eq!(app.changed_files()[0].path, "src/demo.py");
        assert!(app.changed_files()[0].diff.contains("+print(\"hello\")"));
    }

    #[test]
    fn runtime_events_stay_in_trace_not_transcript() {
        let mut app = App::new();

        app.handle_bridge_event(BridgeEvent::Lifecycle {
            subsystem: "mcp".to_string(),
            payload: json!({
                "subsystem": "mcp",
                "state": "ready",
                "total": 1,
                "healthy": 1,
                "degraded": 0,
                "failed": 0,
                "names": ["context7"],
            }),
        });
        app.handle_bridge_event(BridgeEvent::Compaction {
            payload: json!({
                "kept_messages": 6,
                "removed_messages": 12,
                "summary_chars": 480,
            }),
        });
        app.handle_bridge_event(BridgeEvent::Summary {
            payload: json!({
                "kind": "turn_outcome",
                "summary": "turn finished cleanly",
            }),
        });

        assert!(app.messages.is_empty());
        assert_eq!(app.trace_entries().len(), 3);
        assert!(app.trace_entries()[0].contains("lifecycle subsystem=mcp state=ready"));
        assert!(app.trace_entries()[1].contains("compaction event detail=session compaction recorded"));
        assert!(app.trace_entries()[2].contains("summary event kind=turn_outcome"));
    }

    #[test]
    fn final_chat_result_does_not_duplicate_streamed_assistant_message() {
        let mut app = App::new();
        let text = "Hello! I'm Logician, your engineering and analysis agent. How can I help you today?";

        app.handle_bridge_event(BridgeEvent::Token(text.to_string()));
        app.stop_live_message();

        assert_eq!(app.messages.len(), 1);
        assert_eq!(app.messages[0].role, Role::Assistant);
        assert_eq!(app.messages[0].text, text);
        app.handle_chat_result(Ok(json!({
            "final_response": text,
            "tool_calls": [],
            "iterations": 1,
            "tool_errors": 0,
        })));

        assert_eq!(app.messages.len(), 1);
        assert_eq!(app.messages[0].role, Role::Assistant);
        assert_eq!(app.messages[0].text, text);
    }

    #[test]
    fn assistant_live_stream_drops_prefix_matching_previous_thinking_message() {
        let mut app = App::new();

        app.handle_bridge_event(BridgeEvent::ThinkingToken(
            "The user said hi.".to_string(),
        ));
        app.stop_live_message();

        app.handle_bridge_event(BridgeEvent::Token(
            "The user said hi.Hello!".to_string(),
        ));

        let live = app.live.as_ref().expect("assistant message should be live");
        assert_eq!(live.role, Role::Assistant);
        assert_eq!(live.text, "Hello!");

        app.stop_live_message();

        assert_eq!(app.messages.len(), 2);
        assert_eq!(app.messages[1].role, Role::Assistant);
        assert_eq!(app.messages[1].text, "Hello!");
    }

    #[test]
    fn assistant_stream_drops_prefix_matching_previous_thinking_message() {
        let mut app = App::new();

        app.handle_bridge_event(BridgeEvent::ThinkingToken(
            "The user has sent a simple greeting (\"hi\"). This is a social interaction.".to_string(),
        ));
        app.stop_live_message();

        app.handle_bridge_event(BridgeEvent::Token(
            "The user has sent a simple greeting (\"hi\"). This is a social interaction.Hello!"
                .to_string(),
        ));

        let live = app.live.as_ref().expect("assistant message should be live");
        assert_eq!(live.text, "Hello!");
    }

    #[test]
    fn thinking_message_repairs_previous_assistant_prefix() {
        let mut app = App::new();

        app.handle_bridge_event(BridgeEvent::Token(
            "The user has sent a simple greeting (\"hi\"). This is a social interaction.Hello!"
                .to_string(),
        ));
        app.stop_live_message();

        app.handle_bridge_event(BridgeEvent::ThinkingToken(
            "The user has sent a simple greeting (\"hi\"). This is a social interaction."
                .to_string(),
        ));
        app.stop_live_message();

        assert_eq!(app.messages.len(), 2);
        assert_eq!(app.messages[0].role, Role::Assistant);
        assert_eq!(app.messages[0].text, "Hello!");
        assert_eq!(app.messages[1].role, Role::Thinking);
    }

    #[test]
    fn final_chat_result_dedupes_assistant_even_with_trailing_thinking_message() {
        let mut app = App::new();

        app.add_message(Role::Assistant, "Hello!");
        app.add_message(Role::Thinking, "Reasoning");

        app.handle_chat_result(Ok(json!({
            "final_response": "Hello!",
            "tool_calls": [],
            "iterations": 1,
            "tool_errors": 0,
        })));

        assert_eq!(app.messages.len(), 2);
        assert_eq!(app.messages[0].role, Role::Assistant);
        assert_eq!(app.messages[0].text, "Hello!");
        assert_eq!(app.messages[1].role, Role::Thinking);
    }

    #[test]
    fn tool_events_are_grouped_into_single_transcript_summary() {
        let mut app = App::new();

        app.handle_bridge_event(BridgeEvent::ToolStart {
            name: "read_file".to_string(),
            args: Some(json!({"path": "src/main.rs"})),
            sequence: 1,
        });
        app.handle_bridge_event(BridgeEvent::ToolEnd {
            name: "read_file".to_string(),
            sequence: 1,
            status: "ok".to_string(),
            duration_ms: 12,
            cache_hit: false,
            error: None,
            result_preview: Some("path=src/main.rs · lines=120".to_string()),
            result_output: None,
            args: Some(json!({"path": "src/main.rs"})),
        });

        assert!(app.messages.is_empty());

        app.handle_chat_result(Ok(json!({
            "final_response": "",
            "tool_calls": [{"name": "read_file"}],
            "iterations": 1,
            "tool_errors": 0,
        })));

        assert_eq!(app.messages.len(), 1);
        assert_eq!(app.messages[0].role, Role::Tool);
        assert!(app.messages[0].text.contains("step receipt"));
        assert!(app.messages[0].text.contains("ok: #1 read_file"));
        assert_eq!(app.activity_entries().len(), 1);
    }

    #[test]
    fn live_step_summary_matches_current_activity_wording() {
        let mut app = App::new();

        app.handle_bridge_event(BridgeEvent::ToolStart {
            name: "read_file".to_string(),
            args: Some(json!({"path": "src/main.rs"})),
            sequence: 1,
        });

        let summary = app.live_step_summary(80);
        assert_eq!(summary.state, ProgressState::Active);
        assert!(summary.text.contains("#1 read_file"));
        assert!(summary.text.contains("src/main.rs"));
    }

    #[test]
    fn transcript_receipt_is_error_first_and_compact() {
        let mut app = App::new();

        app.handle_bridge_event(BridgeEvent::ToolStart {
            name: "read_file".to_string(),
            args: Some(json!({"path": "src/main.rs"})),
            sequence: 1,
        });
        app.handle_bridge_event(BridgeEvent::ToolEnd {
            name: "read_file".to_string(),
            sequence: 1,
            status: "ok".to_string(),
            duration_ms: 12,
            cache_hit: false,
            error: None,
            result_preview: Some("path=src/main.rs · lines=120".to_string()),
            result_output: None,
            args: Some(json!({"path": "src/main.rs"})),
        });
        app.handle_bridge_event(BridgeEvent::ToolStart {
            name: "write_file".to_string(),
            args: Some(json!({"path": "src/main.rs"})),
            sequence: 2,
        });
        app.handle_bridge_event(BridgeEvent::ToolEnd {
            name: "write_file".to_string(),
            sequence: 2,
            status: "error".to_string(),
            duration_ms: 33,
            cache_hit: false,
            error: Some("permission denied".to_string()),
            result_preview: None,
            result_output: None,
            args: Some(json!({"path": "src/main.rs"})),
        });

        app.handle_chat_result(Ok(json!({
            "final_response": "",
            "tool_calls": [{"name": "read_file"}, {"name": "write_file"}],
            "iterations": 1,
            "tool_errors": 1,
        })));

        let receipt = &app.messages[0].text;
        assert!(receipt.contains("step receipt · 2 steps · 1 error"));
        let error_idx = receipt.find("error:").expect("receipt should include error line");
        let ok_idx = receipt.find("ok:").expect("receipt should include ok line");
        assert!(error_idx < ok_idx);
        assert!(!receipt.contains("activity stack"));
        assert!(!receipt.contains("├─"));
    }
}
