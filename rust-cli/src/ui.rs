use crate::app::{
    ActivityEntry, ActivityState, App, PanelFocus, ProgressEntry, ShortcutHint,
    SlashPopupEntry,
};
use crate::markdown::render_diff;
use ratatui::{
    layout::{Alignment, Constraint, Layout, Position, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};
use serde_json::Value;
use std::{env, path::Path};

// ── Layout constants ──────────────────────────────────────────────────────────

const POPUP_BORDER_H: u16 = 2;
const INPUT_BORDER_H: u16 = 2;
const INPUT_MIN_H: u16 = INPUT_BORDER_H + 2;
const STATUS_BORDER_H: u16 = 1;
const STATUS_TEXT_H: u16 = 4;
const STATUS_TOTAL_H: u16 = STATUS_BORDER_H + STATUS_TEXT_H;

// ── Color palette (One Dark-inspired) ────────────────────────────────────────
const C_ACCENT: Color = Color::Rgb(80, 200, 255);  // bright aqua – focus, active state
const C_PURPLE: Color = Color::Rgb(165, 125, 255); // vivid purple – skills, highlights
const C_GREEN:  Color = Color::Rgb(124, 255, 190); // mint green – success, retrieval
const C_GOLD:   Color = Color::Rgb(255, 210, 130); // warm gold – prompts, warnings
const C_ORANGE: Color = Color::Rgb(255, 156, 98);  // vibrant orange – numbers, repos
const C_TEAL:   Color = Color::Rgb(88, 204, 220);  // aqua – auxiliary accents
const C_RED:    Color = Color::Rgb(245, 110, 133); // coral red – errors, blocked
const C_MUTED:  Color = Color::Rgb(144, 156, 170); // softer gray – labels, secondary
const C_SUBTLE: Color = Color::Rgb(84, 94, 114);    // dark muted – separators
const C_BORDER: Color = Color::Rgb(46, 54, 66);    // near-black – panel border unfocused
const C_BG: Color = Color::Reset;                    // terminal background
const C_PANEL_BG: Color = Color::Reset;              // panel surface (transparent)
const C_PANEL_FOCUS_BG: Color = Color::Reset;        // focused panel surface (transparent)
const C_PANEL_TITLE: Color = Color::Rgb(155, 190, 255); // panel title accent
const C_HINT: Color = Color::Rgb(112, 129, 159);    // hint label

struct PanelSpec {
    title: String,
    focus: PanelFocus,
    min_width: u16,
    min_height: u16,
    borders: Borders,
}

impl PanelSpec {
    fn new(title: impl Into<String>, focus: PanelFocus) -> Self {
        Self {
            title: title.into(),
            focus,
            min_width: 4,
            min_height: 4,
            borders: Borders::ALL,
        }
    }
}

fn begin_panel(f: &mut Frame, app: &App, area: Rect, spec: PanelSpec) -> Option<Rect> {
    if area.width < spec.min_width || area.height < spec.min_height {
        return None;
    }

    let block = Block::default()
        .title(spec.title)
        .borders(spec.borders)
        .border_type(BorderType::Rounded)
        .style(panel_background_style(app, spec.focus))
        .border_style(panel_border_style(app, spec.focus));
    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.width == 0 || inner.height == 0 {
        return None;
    }

    Some(inner)
}

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn draw(f: &mut Frame, app: &mut App) {
    let area = f.area();
    app.begin_frame();

    let popup_items = app.slash_popup_entries(8);
    let show_popup = !app.busy && app.slash_popup_visible() && !popup_items.is_empty();
    let requested_input_h = (app.input_line_count() as u16)
        .saturating_add(1)
        .saturating_add(INPUT_BORDER_H)
        .max(INPUT_MIN_H);

    let [msgs_area, popup_area, input_area, status_area] =
        compute_layout(area, show_popup, popup_items.len(), requested_input_h);
    let (chat_area, panels_area) = split_chat_and_panels(
        msgs_area,
        app.has_image_preview(),
        app.todo_on,
        app.tool_output_on,
        app.context_on,
        app.rag_on,
    );
    let dashboard_area = split_panels(
        panels_area,
        app.has_image_preview(),
        app.todo_on,
        app.tool_output_on,
        app.context_on,
        app.rag_on,
    );

    // Side-effect kept here but isolated — feeds scroll logic for this frame.
    app.update_area_h(chat_area.height);

    let show_splash = app.messages.iter().all(|message| message.role == crate::app::Role::System);
    if show_splash {
        draw_startup_splash(f, app, chat_area);
    } else {
        draw_messages(f, app, chat_area);
    }
    if let Some(dashboard_area) = dashboard_area {
        draw_dashboard_panel(f, app, dashboard_area);
    }
    if show_popup {
        draw_slash_popup(f, app, popup_area, &popup_items);
    }
    draw_input(f, app, input_area);
    draw_status(f, app, status_area);
}

// ── Layout computation (pure, testable) ───────────────────────────────────────

fn compute_layout(
    area: Rect,
    show_popup: bool,
    popup_item_count: usize,
    requested_input_h: u16,
) -> [Rect; 4] {
    let popup_h: u16 = if show_popup {
        (popup_item_count as u16 + POPUP_BORDER_H).min(10)
    } else {
        0
    };

    let max_input_h_by_screen = area.height.saturating_mul(2) / 5;
    let min_input_h = INPUT_MIN_H.min(area.height.saturating_sub(popup_h + STATUS_TOTAL_H));
    let max_input_h = area
        .height
        .saturating_sub(popup_h + STATUS_TOTAL_H + 1)
        .max(min_input_h)
        .min(max_input_h_by_screen.max(min_input_h));
    let input_h = requested_input_h.clamp(min_input_h, max_input_h);

    let fixed_h = popup_h + input_h + STATUS_TOTAL_H;
    let msgs_h = area.height.saturating_sub(fixed_h);

    // Always allocate four rows; zero-height areas are safe throughout.
    let chunks = Layout::vertical([
        Constraint::Length(msgs_h),
        Constraint::Length(popup_h),
        Constraint::Length(input_h),
        Constraint::Length(STATUS_TOTAL_H),
    ])
    .split(area);

    [chunks[0], chunks[1], chunks[2], chunks[3]]
}

fn split_chat_and_panels(
    area: Rect,
    show_image: bool,
    show_todo: bool,
    show_tool_output: bool,
    show_context: bool,
    show_rag: bool,
) -> (Rect, Option<Rect>) {
    let panel_count = [show_image, show_todo, show_tool_output, show_context, show_rag]
        .into_iter()
        .filter(|enabled| *enabled)
        .count();

    if panel_count == 0 || area.height < 6 || area.width < 50 {
        return (area, None);
    }

    let panel_pct = match panel_count {
        1 => 35,
        2 => 38,
        3 => 42,
        _ => 45,
    };

    if area.width < 100 {
        let chunks = Layout::vertical([
            Constraint::Percentage(100 - panel_pct),
            Constraint::Percentage(panel_pct),
        ])
        .split(area);
        return (chunks[0], Some(chunks[1]));
    }

    let chunks = Layout::horizontal([
        Constraint::Percentage(100 - panel_pct),
        Constraint::Percentage(panel_pct),
    ])
    .split(area);
    (chunks[0], Some(chunks[1]))
}

fn split_panels(
    area: Option<Rect>,
    show_image: bool,
    show_todo: bool,
    show_tool_output: bool,
    show_context: bool,
    show_rag: bool,
) -> Option<Rect> {
    let Some(area) = area else {
        return None;
    };
    if !(show_image || show_todo || show_tool_output || show_context || show_rag) {
        return None;
    }
    Some(area)
}

fn draw_dashboard_panel(f: &mut Frame, app: &mut App, area: Rect) {
    let Some(inner) = begin_panel(
        f,
        app,
        area,
        PanelSpec::new(" Dashboard ", PanelFocus::ToolOutput),
    ) else {
        return;
    };

    let tabs = app.dashboard_tabs();
    if tabs.is_empty() {
        let paragraph = Paragraph::new("No dashboard panels enabled.")
            .style(Style::default().fg(C_MUTED));
        f.render_widget(paragraph, inner);
        return;
    }

    let active = app.active_dashboard_tab().unwrap_or(tabs[0]);

    let tab_spans: Vec<Span> = tabs
        .into_iter()
        .map(|tab| {
            let label = match tab {
                PanelFocus::Image => "Image",
                PanelFocus::Todo => "Todo",
                PanelFocus::ToolOutput => "Tool Output",
                PanelFocus::Context => "Context",
                PanelFocus::Rag => "RAG",
                _ => "Other",
            };
            let style = if tab == active {
                Style::default().fg(C_ACCENT).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(C_HINT)
            };
            Span::styled(format!(" {label} "), style)
        })
        .collect();

    let header_area = Rect {
        x: inner.x,
        y: inner.y,
        width: inner.width,
        height: 1,
    };
    f.render_widget(
        Paragraph::new(Line::from(tab_spans)).wrap(Wrap { trim: false }),
        header_area,
    );

    let content_area = Rect {
        x: inner.x,
        y: inner.y + 1,
        width: inner.width,
        height: inner.height.saturating_sub(1),
    };

    match active {
        PanelFocus::Image => {
            if app.has_image_preview() {
                draw_image_panel(f, app, content_area);
            } else {
                render_empty_dashboard(f, content_area, "Image preview unavailable.");
            }
        }
        PanelFocus::Todo => draw_todo_panel(f, app, content_area),
        PanelFocus::ToolOutput => draw_tool_output_panel(f, app, content_area),
        PanelFocus::Context => draw_context_panel(f, app, content_area),
        PanelFocus::Rag => draw_rag_panel(f, app, content_area),
        _ => render_empty_dashboard(f, content_area, "No dashboard tab selected."),
    }
}

fn render_empty_dashboard(f: &mut Frame, area: Rect, message: &str) {
    let paragraph = Paragraph::new(message)
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(C_MUTED));
    f.render_widget(paragraph, area);
}

fn draw_startup_splash(f: &mut Frame, app: &App, area: Rect) {
    if area.height == 0 {
        return;
    }

    let cwd = env::current_dir()
        .ok()
        .and_then(|p| p.to_str().map(|s| s.to_owned()))
        .unwrap_or_else(|| "-".to_string());
    let phase_label = if app.phase_note.trim().is_empty() || app.phase_note == "ready" {
        app.phase.to_string()
    } else {
        format!("{} - {}", app.phase, app.phase_note)
    };

    let mut lines = vec![
        Line::raw("")
        .into(),
        Line::from(Span::styled(
            "        .-.",
            Style::default().fg(C_ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled("        .-''''-.", Style::default().fg(C_ACCENT))),
        Line::from(Span::styled("     .-'        '-.", Style::default().fg(C_ACCENT))),
        Line::from(Span::styled("   .'              '.", Style::default().fg(C_ACCENT))),
        Line::from(Span::styled("  /                  \\", Style::default().fg(C_ACCENT))),
        Line::from(Span::styled(" |                    |", Style::default().fg(C_ACCENT))),
        Line::from(Span::styled(" |     .--------.     |", Style::default().fg(C_ACCENT))),
        Line::from(Span::styled(" |    /  ~~~ ~~  \\    |", Style::default().fg(C_ACCENT))),
        Line::from(Span::styled(" |    \\__~~~~~~__/    |", Style::default().fg(C_ACCENT))),
        Line::from(Span::styled(" |         |          |", Style::default().fg(C_ACCENT))),
        Line::from(Span::styled(" |         |          |", Style::default().fg(C_ACCENT))),
        Line::from(Span::styled("  \\        |         /", Style::default().fg(C_ACCENT))),
        Line::from(Span::styled("   '.      |       .'", Style::default().fg(C_ACCENT))),
        Line::from(Span::styled("     '-.   |    .-'", Style::default().fg(C_ACCENT))),
        Line::from(Span::styled("         '-|.-'", Style::default().fg(C_ACCENT))),
        Line::from(Span::styled("           ||", Style::default().fg(C_ACCENT))),
        Line::from(Span::styled("         __||__", Style::default().fg(C_ACCENT))),
        Line::from(Span::styled("        |______|", Style::default().fg(C_ACCENT))),
        Line::from(Span::styled("         \\____/", Style::default().fg(C_ACCENT))),
        Line::raw(""),
        Line::from(Span::styled(
            "Logician CLI",
            Style::default().fg(C_ACCENT).add_modifier(Modifier::BOLD),
        )),
        Line::raw(""),
        Line::from(Span::styled(
            "A calm Rust-powered agent UI for repo exploration and graph ingestion.",
            Style::default().fg(C_PURPLE),
        )),
        Line::raw(""),
        Line::from(Span::styled(format!("path: {cwd}"), Style::default().fg(C_MUTED))),
        Line::from(Span::styled(format!("status: {phase_label}"), Style::default().fg(C_GREEN))),
    ];

    if let Some(error) = app.last_tool_error.as_deref() {
        if !error.trim().is_empty() {
            lines.push(Line::raw(""));
            lines.push(Line::from(Span::styled(
                "startup error:",
                Style::default().fg(C_RED).add_modifier(Modifier::BOLD),
            )));
            lines.push(Line::from(Span::styled(error, Style::default().fg(C_RED))));
        }
    }

    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "Use /help or Tab to open commands, Ctrl+R to toggle the dashboard.",
        Style::default().fg(C_SUBTLE),
    )));
    lines.push(Line::from(Span::styled(
        "Run ./graph-cli build . after repo changes to refresh the graph.",
        Style::default().fg(C_GOLD),
    )));

    let paragraph = Paragraph::new(lines)
        .alignment(Alignment::Center)
        .wrap(Wrap { trim: true })
        .style(Style::default().fg(Color::White));
    f.render_widget(paragraph, area);
}
// ── Messages panel ────────────────────────────────────────────────────────────

fn draw_messages(f: &mut Frame, app: &mut App, area: Rect) {
    if area.height == 0 {
        return;
    }
    let (lines, thinking_headers) = app.visible_transcript(area.height as u16);
    let header_rects = thinking_headers
        .into_iter()
        .map(|(message_idx, row)| {
            (
                message_idx,
                Rect {
                    x: area.x,
                    y: area.y.saturating_add(row),
                    width: area.width,
                    height: 1,
                },
            )
        })
        .collect::<Vec<_>>();
    app.register_messages_area(area, header_rects);
    let para = Paragraph::new(lines)
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(Color::White));
    f.render_widget(para, area);
}

fn render_activity_stack(entries: &[ActivityEntry]) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    if entries.is_empty() {
        return lines;
    }

    lines.push(Line::from(Span::styled(
        "activity stack",
        Style::default()
            .fg(C_ACCENT)
            .add_modifier(Modifier::BOLD),
    )));

    for (idx, entry) in entries.iter().enumerate() {
        let branch = if idx + 1 == entries.len() { "└─ " } else { "├─ " };
        let mut spans = vec![
            Span::styled(
                branch.to_string(),
                Style::default().fg(C_SUBTLE),
            ),
            Span::styled(
                format!("#{} {}", entry.sequence, entry.name),
                Style::default()
                    .fg(entry.status.color())
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" · {}", match entry.status {
                    ActivityState::Running => "running",
                    ActivityState::Complete => "ok",
                    ActivityState::Error => "error",
                }),
                Style::default().fg(Color::White),
            ),
        ];
        if let Some(duration_ms) = entry.duration_ms {
            spans.push(Span::styled(
                format!(" · {}ms", duration_ms),
                Style::default().fg(Color::Gray),
            ));
        }
        if entry.cache_hit {
            spans.push(Span::styled(
                " · cached",
                Style::default().fg(Color::Yellow),
            ));
        }
        lines.push(Line::from(spans));

        if let Some(summary) = entry.summary.as_deref() {
            lines.push(Line::from(vec![
                Span::styled("   ", Style::default()),
                Span::styled(summary.to_string(), Style::default().fg(Color::Gray)),
            ]));
        }
        if let Some(detail) = entry.detail.as_deref() {
            lines.push(Line::from(vec![
                Span::styled("   ", Style::default()),
                Span::styled(
                    detail.to_string(),
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::DIM),
                ),
            ]));
        }
    }

    lines.push(Line::raw(""));
    lines
}

fn draw_image_panel(f: &mut Frame, app: &mut App, area: Rect) {
    app.register_image_area(area);

    let path = app.image_preview_path().unwrap_or("").to_string();
    let file_name = Path::new(&path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("image");
    let Some(inner) = begin_panel(
        f,
        app,
        area,
        PanelSpec::new(format!(" Image: {file_name} "), PanelFocus::Image),
    ) else {
        return;
    };

    if let Err(err) = app.render_image_preview(f, inner) {
        let fallback = if path.is_empty() {
            format!("image unavailable\n{err}")
        } else {
            format!("{path}\n\n{err}")
        };
        f.render_widget(
            Paragraph::new(fallback)
                .wrap(Wrap { trim: false })
                .style(Style::default().fg(Color::Gray)),
            inner,
        );
    }
}

fn draw_todo_panel(f: &mut Frame, app: &mut App, area: Rect) {
    let Some(inner) = begin_panel(
        f,
        app,
        area,
        PanelSpec::new(" Task Checklist ", PanelFocus::Todo),
    ) else {
        return;
    };

    let mut lines: Vec<Line<'static>> = Vec::new();
    if app.bridge_state.todo.is_empty() {
        lines.push(Line::from(vec![Span::styled(
            "  No active tasks.",
            Style::default()
                .fg(C_SUBTLE)
                .add_modifier(Modifier::ITALIC),
        )]));
    } else {
        for item in &app.bridge_state.todo {
            let symbol = match item.status.as_str() {
                "completed" | "done" => " [x] ",
                "in-progress" | "in_progress" | "doing" => " [/] ",
                "blocked" => " [!] ",
                _ => " [ ] ",
            };
            let color = match item.status.as_str() {
                "completed" | "done" => C_GREEN,
                "in-progress" | "in_progress" | "doing" => C_GOLD,
                "blocked" => C_RED,
                _ => C_MUTED,
            };

            lines.push(Line::from(vec![
                Span::styled(
                    symbol,
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(item.title.clone(), Style::default().fg(Color::White)),
            ]));
            if !item.note.is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("     ", Style::default()),
                    Span::styled(
                        item.note.clone(),
                        Style::default()
                            .fg(Color::DarkGray)
                            .add_modifier(Modifier::DIM),
                    ),
                ]));
            }
        }
    }

    app.register_todo_area(
        area,
        inner.height,
        lines.len().min(u16::MAX as usize) as u16,
    );
    f.render_widget(
        Paragraph::new(lines)
            .scroll((app.todo_scroll(), 0))
            .wrap(Wrap { trim: false }),
        inner,
    );
}

fn draw_tool_output_panel(f: &mut Frame, app: &mut App, area: Rect) {
    let title = if app.has_change_records() {
        format!(
            " Inspector (changes: {}) (Ctrl+R) ",
            app.changed_files().len()
        )
    } else {
        " Inspector (Ctrl+R) ".to_string()
    };
    let Some(inner) = begin_panel(
        f,
        app,
        area,
        PanelSpec::new(title, PanelFocus::ToolOutput),
    ) else {
        return;
    };

    if app.has_change_records() {
        let (lines, change_headers) = render_changes_panel(app, inner);
        app.register_tool_output_area(
            area,
            inner.height,
            lines.len().min(u16::MAX as usize) as u16,
            change_headers,
        );
        f.render_widget(
            Paragraph::new(lines)
                .scroll((app.tool_output_scroll(), 0))
                .wrap(Wrap { trim: false })
                .style(Style::default()),
            inner,
        );
        return;
    }

    let mut styled_lines: Vec<Line<'static>> = Vec::new();

    let activity_entries = app.activity_entries();
    if !activity_entries.is_empty() {
        styled_lines.extend(render_activity_stack(activity_entries));
    }

    if !app.active_skill_ids().is_empty() {
        styled_lines.push(Line::from(Span::styled(
            "request skills",
            Style::default()
                .fg(C_PURPLE)
                .add_modifier(Modifier::BOLD),
        )));
        styled_lines.push(Line::from(vec![
            Span::styled("  active ", Style::default().fg(C_SUBTLE)),
            Span::styled(
                app.active_skill_ids().join(", "),
                Style::default().fg(C_PURPLE),
            ),
        ]));
        if !app.active_selected_tools().is_empty() {
            styled_lines.push(Line::from(vec![
                Span::styled("  tools  ", Style::default().fg(C_SUBTLE)),
                Span::styled(
                    app.active_selected_tools().join(", "),
                    Style::default().fg(C_TEAL),
                ),
            ]));
        }
        styled_lines.push(Line::raw(""));
    }

    if app.trace_on && !app.trace_entries().is_empty() {
        styled_lines.push(Line::from(Span::styled(
            "recent trace",
            Style::default()
                .fg(C_GOLD)
                .add_modifier(Modifier::BOLD),
        )));
        let trace_entries = app.trace_entries();
        let start = trace_entries.len().saturating_sub(24);
        for entry in &trace_entries[start..] {
            styled_lines.push(Line::from(vec![
                Span::styled("  ", Style::default()),
                Span::styled(
                    entry.clone(),
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::DIM),
                ),
            ]));
        }
        styled_lines.push(Line::raw(""));
    }

    let buffer = app.tool_output_buffer();
    if !buffer.is_empty() {
        styled_lines.push(Line::from(Span::styled(
            "tool details",
            Style::default()
                .fg(C_ACCENT)
                .add_modifier(Modifier::BOLD),
        )));
        styled_lines.push(Line::raw(""));
        let full_text = buffer.join("\n\n");
        styled_lines.extend(parse_tool_output_blocks(&full_text, area.width as usize));
    }

    if styled_lines.is_empty() {
        let fallback =
            "  Inspector is empty.\n\n  Ctrl+O shows trace here.\n  Ctrl+R reopens this panel.\n  ";
        app.register_tool_output_area(area, inner.height, 4, Vec::new());
        f.render_widget(
            Paragraph::new(fallback)
                .wrap(Wrap { trim: false })
                .style(Style::default().fg(C_SUBTLE)),
            inner,
        );
        return;
    }

    app.register_tool_output_area(
        area,
        inner.height,
        styled_lines.len().min(u16::MAX as usize) as u16,
        Vec::new(),
    );

    let text = Paragraph::new(styled_lines)
        .scroll((app.tool_output_scroll(), 0))
        .wrap(Wrap { trim: false })
        .style(Style::default());
    f.render_widget(text, inner);
}

fn draw_context_panel(f: &mut Frame, app: &mut App, area: Rect) {
    let title = format!(
        " Context Explorer (Ctrl+E) · {} tool(s) · {} skill(s) · {} active repo(s) · {} retrieval(s) · {} repo(s) · {} mount(s) · {} doc(s) ",
        app.bridge_state.loaded_tools.len(),
        app.bridge_state.loaded_skills.len(),
        app.bridge_state.active_repos.len(),
        app.bridge_state.retrieval_insights.len(),
        app.bridge_state.repo_library.len(),
        app.bridge_state.mounted_paths.len(),
        app.bridge_state.rag_docs.len()
    );
    let Some(inner) = begin_panel(
        f,
        app,
        area,
        PanelSpec::new(title, PanelFocus::Context),
    ) else {
        return;
    };

    let mut lines: Vec<Line<'static>> = Vec::new();
    lines.push(Line::from(vec![
        Span::styled(
            "session ",
            Style::default()
                .fg(C_SUBTLE)
                .add_modifier(Modifier::DIM),
        ),
        Span::styled(
            app.bridge_state.session.clone(),
            Style::default().fg(C_TEAL),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled(
            "messages ",
            Style::default()
                .fg(C_SUBTLE)
                .add_modifier(Modifier::DIM),
        ),
        Span::styled(
            app.bridge_state.msg_count.to_string(),
            Style::default().fg(Color::White),
        ),
        Span::styled(
            "  total tokens≈ ",
            Style::default()
                .fg(C_SUBTLE)
                .add_modifier(Modifier::DIM),
        ),
        Span::styled(
            app.context_token_estimate_total().to_string(),
            Style::default().fg(C_GOLD),
        ),
    ]));
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "loaded tools",
        Style::default()
            .fg(C_TEAL)
            .add_modifier(Modifier::BOLD),
    )));
    if app.bridge_state.loaded_tools.is_empty() {
        lines.push(Line::from(Span::styled(
            "  none",
            Style::default()
                .fg(C_SUBTLE)
                .add_modifier(Modifier::ITALIC),
        )));
    } else {
        for tool in &app.bridge_state.loaded_tools {
            lines.push(Line::from(vec![
                Span::styled("└─ ", Style::default().fg(C_SUBTLE)),
                Span::styled(tool.clone(), Style::default().fg(C_TEAL)),
            ]));
        }
    }

    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "loaded skills",
        Style::default()
            .fg(C_PURPLE)
            .add_modifier(Modifier::BOLD),
    )));
    if app.bridge_state.loaded_skills.is_empty() {
        lines.push(Line::from(Span::styled(
            "  none",
            Style::default()
                .fg(C_SUBTLE)
                .add_modifier(Modifier::ITALIC),
        )));
    } else {
        for skill in &app.bridge_state.loaded_skills {
            lines.push(Line::from(vec![
                Span::styled("└─ ", Style::default().fg(C_SUBTLE)),
                Span::styled(skill.clone(), Style::default().fg(C_PURPLE)),
            ]));
        }
    }

    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "active repos",
        Style::default()
            .fg(C_GOLD)
            .add_modifier(Modifier::BOLD),
    )));
    if app.bridge_state.active_repos.is_empty() {
        lines.push(Line::from(Span::styled(
            "  none",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        )));
    } else {
        for repo in &app.bridge_state.active_repos {
            let label = if repo.name.trim().is_empty() {
                repo.id.clone()
            } else {
                format!("{} ({})", repo.name, repo.id)
            };
            lines.push(Line::from(vec![
                Span::styled("└─ ", Style::default().fg(Color::DarkGray)),
                Span::styled(label, Style::default().fg(Color::Yellow)),
            ]));
            lines.push(Line::from(vec![
                Span::styled("   ", Style::default()),
                Span::styled(repo.path.clone(), Style::default().fg(Color::White)),
            ]));
            let mut meta_spans = vec![Span::styled("   ", Style::default())];
            if repo.files_processed > 0 {
                meta_spans.push(Span::styled(
                    format!("files {}", repo.files_processed),
                    Style::default().fg(Color::Cyan),
                ));
                meta_spans.push(Span::styled("  ", Style::default()));
            }
            if repo.chunks_added > 0 {
                meta_spans.push(Span::styled(
                    format!("chunks {}", repo.chunks_added),
                    Style::default().fg(Color::LightGreen),
                ));
                meta_spans.push(Span::styled("  ", Style::default()));
            }
            if repo.graph_nodes > 0 {
                meta_spans.push(Span::styled(
                    format!("nodes {}", repo.graph_nodes),
                    Style::default().fg(Color::Magenta),
                ));
                meta_spans.push(Span::styled("  ", Style::default()));
            }
            if repo.graph_edges > 0 {
                meta_spans.push(Span::styled(
                    format!("edges {}", repo.graph_edges),
                    Style::default().fg(Color::LightMagenta),
                ));
                meta_spans.push(Span::styled("  ", Style::default()));
            }
            if repo.graph_symbols > 0 {
                meta_spans.push(Span::styled(
                    format!("symbols {}", repo.graph_symbols),
                    Style::default().fg(Color::Magenta),
                ));
                meta_spans.push(Span::styled("  ", Style::default()));
            }
            if !repo.branch.trim().is_empty() {
                meta_spans.push(Span::styled(
                    repo.branch.clone(),
                    Style::default().fg(Color::Gray),
                ));
                if !repo.commit.trim().is_empty() {
                    meta_spans.push(Span::styled("  ", Style::default()));
                }
            }
            if !repo.commit.trim().is_empty() {
                meta_spans.push(Span::styled(
                    repo.commit.clone(),
                    Style::default().fg(Color::DarkGray),
                ));
            }
            if !repo.last_ingested_at.trim().is_empty() {
                meta_spans.push(Span::styled("  ", Style::default()));
                meta_spans.push(Span::styled(
                    repo.last_ingested_at.clone(),
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::DIM),
                ));
            }
            if !repo.last_graph_built_at.trim().is_empty() {
                meta_spans.push(Span::styled("  ", Style::default()));
                meta_spans.push(Span::styled(
                    repo.last_graph_built_at.clone(),
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::DIM),
                ));
            }
            if meta_spans.len() > 1 {
                lines.push(Line::from(meta_spans));
            }
        }
    }

    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "repo library",
        Style::default()
            .fg(C_ORANGE)
            .add_modifier(Modifier::BOLD),
    )));
    if app.bridge_state.repo_library.is_empty() {
        lines.push(Line::from(Span::styled(
            "  none",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        )));
    } else {
        for repo in &app.bridge_state.repo_library {
            let is_active = app
                .bridge_state
                .active_repos
                .iter()
                .any(|item| item.id == repo.id);
            let label = if repo.name.trim().is_empty() {
                repo.id.clone()
            } else {
                format!("{} ({})", repo.name, repo.id)
            };
            lines.push(Line::from(vec![
                Span::styled("└─ ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    label,
                    Style::default().fg(if is_active {
                        Color::Yellow
                    } else {
                        Color::White
                    }),
                ),
                Span::styled(
                    if is_active { "  active" } else { "" },
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::DIM),
                ),
            ]));
            let mut info_spans = vec![Span::styled("   ", Style::default())];
            if repo.files_processed > 0 {
                info_spans.push(Span::styled(
                    format!("files {}", repo.files_processed),
                    Style::default().fg(Color::Cyan),
                ));
                info_spans.push(Span::styled("  ", Style::default()));
            }
            if repo.chunks_added > 0 {
                info_spans.push(Span::styled(
                    format!("chunks {}", repo.chunks_added),
                    Style::default().fg(Color::LightGreen),
                ));
                info_spans.push(Span::styled("  ", Style::default()));
            }
            if repo.graph_nodes > 0 {
                info_spans.push(Span::styled(
                    format!("nodes {}", repo.graph_nodes),
                    Style::default().fg(Color::Magenta),
                ));
                info_spans.push(Span::styled("  ", Style::default()));
            }
            if repo.graph_edges > 0 {
                info_spans.push(Span::styled(
                    format!("edges {}", repo.graph_edges),
                    Style::default().fg(Color::LightMagenta),
                ));
                info_spans.push(Span::styled("  ", Style::default()));
            }
            if repo.graph_symbols > 0 {
                info_spans.push(Span::styled(
                    format!("symbols {}", repo.graph_symbols),
                    Style::default().fg(Color::Magenta),
                ));
                info_spans.push(Span::styled("  ", Style::default()));
            }
            if !repo.path.trim().is_empty() {
                info_spans.push(Span::styled(
                    repo.path.clone(),
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::DIM),
                ));
            }
            if info_spans.len() > 1 {
                lines.push(Line::from(info_spans));
            }
            if !repo.last_ingested_at.trim().is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("   ", Style::default()),
                    Span::styled(
                        format!("last ingested {}", repo.last_ingested_at),
                        Style::default()
                            .fg(Color::DarkGray)
                            .add_modifier(Modifier::DIM),
                    ),
                ]));
            }
            if !repo.last_graph_built_at.trim().is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("   ", Style::default()),
                    Span::styled(
                        format!("last graph {}", repo.last_graph_built_at),
                        Style::default()
                            .fg(Color::DarkGray)
                            .add_modifier(Modifier::DIM),
                    ),
                ]));
            }
        }
    }

    lines.push(Line::raw(""));

    lines.push(Line::from(Span::styled(
        "mounted paths",
        Style::default()
            .fg(C_ACCENT)
            .add_modifier(Modifier::BOLD),
    )));
    if app.bridge_state.mounted_paths.is_empty() {
        lines.push(Line::from(Span::styled(
            "  none",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        )));
    } else {
        for item in &app.bridge_state.mounted_paths {
            lines.push(Line::from(vec![
                Span::styled("└─ ", Style::default().fg(Color::DarkGray)),
                Span::styled(item.path.clone(), Style::default().fg(Color::White)),
            ]));
            lines.push(Line::from(vec![
                Span::styled("   ", Style::default()),
                Span::styled(
                    format!("files {}", item.file_count),
                    Style::default().fg(Color::Cyan),
                ),
                Span::styled("  ", Style::default()),
                Span::styled(
                    format!("tokens≈ {}", item.token_count),
                    Style::default().fg(Color::Yellow),
                ),
                Span::styled("  ", Style::default()),
                Span::styled(
                    format!("depth {}", item.map_depth),
                    Style::default().fg(Color::Gray),
                ),
            ]));
            if !item.glob.trim().is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("   ", Style::default()),
                    Span::styled(
                        format!("glob {}", item.glob),
                        Style::default()
                            .fg(Color::DarkGray)
                            .add_modifier(Modifier::DIM),
                    ),
                ]));
            }
        }
    }

    app.register_context_area(
        area,
        inner.height,
        lines.len().min(u16::MAX as usize) as u16,
    );
    f.render_widget(
        Paragraph::new(lines)
            .scroll((app.context_scroll(), 0))
            .wrap(Wrap { trim: false }),
        inner,
    );
}

fn draw_rag_panel(f: &mut Frame, app: &mut App, area: Rect) {
    let inventory = &app.bridge_state.rag_inventory;
    let title = format!(
        " RAG Explorer (Ctrl+G) · {} chunk(s) · {} repo(s) · {} retrieval(s) · {} search(es) ",
        inventory.repo_chunks.max(inventory.active_doc_chunks),
        inventory.repo_count.max(inventory.active_repo_count),
        inventory.retrieval_count,
        app.bridge_state.recent_rag_queries.len()
    );
    let Some(inner) = begin_panel(f, app, area, PanelSpec::new(title, PanelFocus::Rag)) else {
        return;
    };

    let mut lines: Vec<Line<'static>> = Vec::new();
    lines.push(Line::from(vec![
        Span::styled(
            "store ",
            Style::default()
                .fg(C_SUBTLE)
                .add_modifier(Modifier::DIM),
        ),
        Span::styled(
            if inventory.vector_backend.trim().is_empty() {
                "unknown".to_string()
            } else {
                inventory.vector_backend.clone()
            },
            Style::default().fg(C_GREEN).add_modifier(Modifier::BOLD),
        ),
        Span::styled("  ", Style::default()),
        Span::styled(
            inventory.vector_path.clone(),
            Style::default().fg(C_MUTED),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled(
            "chunks ",
            Style::default()
                .fg(C_SUBTLE)
                .add_modifier(Modifier::DIM),
        ),
        Span::styled(inventory.repo_chunks.to_string(), Style::default().fg(C_GOLD)),
        Span::styled("  active docs ", Style::default().fg(C_SUBTLE)),
        Span::styled(
            format!("{} / {} chunks", inventory.active_doc_count, inventory.active_doc_chunks),
            Style::default().fg(C_TEAL),
        ),
        Span::styled("  retrievals ", Style::default().fg(C_SUBTLE)),
        Span::styled(inventory.retrieval_count.to_string(), Style::default().fg(C_ACCENT)),
    ]));
    lines.push(Line::from(vec![
        Span::styled(
            "hint ",
            Style::default()
                .fg(C_SUBTLE)
                .add_modifier(Modifier::DIM),
        ),
        Span::styled(
            "/rag list · /rag search <query> [top_k] · /rag clear",
            Style::default().fg(C_ORANGE),
        ),
    ]));

    if !inventory.legacy_paths.is_empty() {
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled(
            "legacy paths",
            Style::default().fg(C_RED).add_modifier(Modifier::BOLD),
        )));
        for path in &inventory.legacy_paths {
            lines.push(Line::from(vec![
                Span::styled("└─ ", Style::default().fg(C_SUBTLE)),
                Span::styled(path.clone(), Style::default().fg(C_RED)),
            ]));
        }
    }

    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "recent searches",
        Style::default().fg(C_GOLD).add_modifier(Modifier::BOLD),
    )));
    if app.bridge_state.recent_rag_queries.is_empty() {
        lines.push(Line::from(Span::styled(
            "  none yet",
            Style::default().fg(C_SUBTLE).add_modifier(Modifier::ITALIC),
        )));
    } else {
        for query in &app.bridge_state.recent_rag_queries {
            lines.push(Line::from(vec![
                Span::styled("└─ ", Style::default().fg(C_SUBTLE)),
                Span::styled(query.query.clone(), Style::default().fg(Color::White)),
                Span::styled("  ", Style::default()),
                Span::styled(
                    format!("{} hit(s)", query.count),
                    Style::default().fg(C_GREEN),
                ),
                Span::styled("  ", Style::default()),
                Span::styled(
                    format!("top_k {}", query.top_k),
                    Style::default().fg(C_TEAL),
                ),
            ]));
            if !query.repo_filter.is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("   ", Style::default()),
                    Span::styled("repos ", Style::default().fg(C_SUBTLE)),
                    Span::styled(query.repo_filter.join(", "), Style::default().fg(C_ORANGE)),
                ]));
            }
            if !query.hits.is_empty() {
                let hit_preview = query
                    .hits
                    .iter()
                    .take(3)
                    .map(|item| {
                        let repo = if item.repo_name.trim().is_empty() {
                            item.repo_id.clone()
                        } else {
                            item.repo_name.clone()
                        };
                        let path = if item.path.trim().is_empty() {
                            "chunk".to_string()
                        } else {
                            item.path.clone()
                        };
                        if item.distance.trim().is_empty() {
                            format!("{} · {}", repo, path)
                        } else {
                            format!("{} · {} · d={}", repo, path, item.distance)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("  |  ");
                lines.push(Line::from(vec![
                    Span::styled("   ", Style::default()),
                    Span::styled(hit_preview, Style::default().fg(C_MUTED)),
                ]));
            }
        }
    }

    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "recent retrieval",
        Style::default().fg(C_TEAL).add_modifier(Modifier::BOLD),
    )));
    if app.bridge_state.retrieval_insights.is_empty() {
        lines.push(Line::from(Span::styled(
            "  none",
            Style::default().fg(C_SUBTLE).add_modifier(Modifier::ITALIC),
        )));
    } else {
        for insight in &app.bridge_state.retrieval_insights {
            let label = if !insight.repo_name.trim().is_empty() {
                insight.repo_name.clone()
            } else if !insight.repo_id.trim().is_empty() {
                insight.repo_id.clone()
            } else {
                "retrieval".to_string()
            };
            lines.push(Line::from(vec![
                Span::styled("└─ ", Style::default().fg(C_SUBTLE)),
                Span::styled(label, Style::default().fg(C_TEAL)),
            ]));
            if !insight.query.trim().is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("   ", Style::default()),
                    Span::styled("query ", Style::default().fg(C_SUBTLE)),
                    Span::styled(insight.query.clone(), Style::default().fg(Color::White)),
                ]));
            }
            if !insight.retrieved_paths.is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("   ", Style::default()),
                    Span::styled("hits  ", Style::default().fg(C_SUBTLE)),
                    Span::styled(
                        insight.retrieved_paths.join(", "),
                        Style::default().fg(C_ACCENT),
                    ),
                ]));
            }
        }
    }

    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "top indexed repos",
        Style::default().fg(C_ORANGE).add_modifier(Modifier::BOLD),
    )));
    if inventory.top_repos.is_empty() {
        lines.push(Line::from(Span::styled(
            "  none",
            Style::default().fg(C_SUBTLE).add_modifier(Modifier::ITALIC),
        )));
    } else {
        for repo in &inventory.top_repos {
            let label = if repo.repo_name.trim().is_empty() {
                repo.repo_id.clone()
            } else if repo.repo_id.trim().is_empty() || repo.repo_name == repo.repo_id {
                repo.repo_name.clone()
            } else {
                format!("{} ({})", repo.repo_name, repo.repo_id)
            };
            lines.push(Line::from(vec![
                Span::styled("└─ ", Style::default().fg(C_SUBTLE)),
                Span::styled(label, Style::default().fg(C_ORANGE)),
                Span::styled("  ", Style::default()),
                Span::styled(format!("{} chunks", repo.chunks), Style::default().fg(C_GOLD)),
                Span::styled("  ", Style::default()),
                Span::styled(format!("{} files", repo.files), Style::default().fg(C_TEAL)),
            ]));
            if !repo.last_ingested_at.trim().is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("   ", Style::default()),
                    Span::styled(
                        format!("last ingested {}", repo.last_ingested_at),
                        Style::default().fg(C_MUTED).add_modifier(Modifier::DIM),
                    ),
                ]));
            }
        }
    }

    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "active rag docs",
        Style::default().fg(C_GREEN).add_modifier(Modifier::BOLD),
    )));
    if app.bridge_state.rag_docs.is_empty() {
        lines.push(Line::from(Span::styled(
            "  none",
            Style::default().fg(C_SUBTLE).add_modifier(Modifier::ITALIC),
        )));
    } else {
        for item in &app.bridge_state.rag_docs {
            lines.push(Line::from(vec![
                Span::styled("└─ ", Style::default().fg(C_SUBTLE)),
                Span::styled(item.path.clone(), Style::default().fg(Color::White)),
            ]));
            lines.push(Line::from(vec![
                Span::styled("   ", Style::default()),
                Span::styled(
                    if item.kind.trim().is_empty() {
                        "doc".to_string()
                    } else {
                        item.kind.clone()
                    },
                    Style::default().fg(C_GREEN),
                ),
                Span::styled("  ", Style::default()),
                Span::styled(format!("chunks {}", item.chunks), Style::default().fg(C_TEAL)),
                Span::styled("  ", Style::default()),
                Span::styled(
                    format!("tokens≈ {}", item.token_count),
                    Style::default().fg(C_GOLD),
                ),
            ]));
            if !item.label.trim().is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("   ", Style::default()),
                    Span::styled(item.label.clone(), Style::default().fg(C_MUTED)),
                ]));
            }
        }
    }

    app.register_rag_area(area, inner.height, lines.len().min(u16::MAX as usize) as u16);
    f.render_widget(
        Paragraph::new(lines)
            .scroll((app.rag_scroll(), 0))
            .wrap(Wrap { trim: false }),
        inner,
    );
}

fn render_changes_panel(app: &App, inner: Rect) -> (Vec<Line<'static>>, Vec<(usize, Rect)>) {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut change_headers = Vec::new();

    lines.push(Line::from(vec![
        Span::styled(
            "recent files",
            Style::default()
                .fg(C_GOLD)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(" ({})", app.changed_files().len()),
            Style::default().fg(C_SUBTLE),
        ),
    ]));

    if let Some(current) = app.current_changed_file() {
        for idx in (0..app.changed_files().len()).rev() {
            let item = &app.changed_files()[idx];
            let active = item.path == current.path;
            let file_name = Path::new(&item.path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(&item.path);
            let row = lines.len().min(u16::MAX as usize) as u16;
            let scroll = app.tool_output_scroll();
            if row >= scroll && row < scroll.saturating_add(inner.height) {
                change_headers.push((
                    idx,
                    Rect {
                        x: inner.x,
                        y: inner.y.saturating_add(row.saturating_sub(scroll)),
                        width: inner.width,
                        height: 1,
                    },
                ));
            }
            lines.push(Line::from(vec![
                Span::styled(
                    if item.expanded { "▼ " } else { "▶ " },
                    Style::default().fg(if active { C_GOLD } else { C_SUBTLE }),
                ),
                Span::styled(
                    file_name.to_string(),
                    Style::default().fg(if active { C_ACCENT } else { Color::White }),
                ),
                Span::styled("  ", Style::default()),
                Span::styled(
                    item.tool.to_string(),
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::DIM),
                ),
            ]));
            if item.expanded {
                lines.push(Line::from(vec![
                    Span::styled("   path ", Style::default().fg(Color::DarkGray)),
                    Span::styled(item.path.clone(), Style::default().fg(Color::Cyan)),
                ]));
                lines.extend(render_diff(&item.path, &item.diff).into_iter().map(|line| {
                    let mut spans = Vec::with_capacity(line.spans.len() + 1);
                    spans.push(Span::raw("   "));
                    spans.extend(line.spans);
                    Line::from(spans)
                }));
                lines.push(Line::raw(""));
            }
        }
    }

    (lines, change_headers)
}

// ── Input composer ────────────────────────────────────────────────────────────

fn draw_input(f: &mut Frame, app: &mut App, area: Rect) {
    if area.height == 0 || area.width == 0 {
        return;
    }

    app.register_input_area(area);

    let border_color = if app.busy {
        C_SUBTLE
    } else if app.is_panel_focused(PanelFocus::Input) {
        C_ACCENT
    } else {
        C_GOLD
    };
    let block = Block::default()
        .borders(Borders::TOP | Borders::BOTTOM)
        .border_type(BorderType::Rounded)
        .style(Style::default().bg(C_PANEL_BG))
        .border_style(Style::default().fg(border_color));
    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height == 0 || inner.width == 0 {
        return;
    }

    let [body_area, footer_area] = if inner.height > 1 {
        let chunks = Layout::vertical([
            Constraint::Length(inner.height.saturating_sub(1)),
            Constraint::Length(1),
        ])
        .split(inner);
        [chunks[0], chunks[1]]
    } else {
        [inner, Rect::default()]
    };

    let raw_lines: Vec<&str> = if app.input.is_empty() {
        vec![""]
    } else {
        app.input.split('\n').collect()
    };
    let (cursor_row, cursor_col) = app.input_cursor_line_col();
    let viewport_h = body_area.height.max(1) as usize;
    let viewport_start = cursor_row.saturating_sub(viewport_h.saturating_sub(1));
    let visible_lines: Vec<Line<'static>> = raw_lines
        .iter()
        .enumerate()
        .skip(viewport_start)
        .take(viewport_h)
        .map(|(row, line)| composer_line(app, row, line, cursor_row, cursor_col))
        .collect();

    f.render_widget(Paragraph::new(visible_lines).style(Style::default().fg(Color::White)), body_area);

    if footer_area.height > 0 && footer_area.width > 0 {
        f.render_widget(
            Paragraph::new(vec![composer_footer_line(app, footer_area.width as usize)])
                .style(Style::default().fg(C_MUTED)),
            footer_area,
        );
    }

    if !app.busy {
        let prefix_w = 2u16;
        let cursor_x = body_area.x + prefix_w + cursor_col as u16;
        let cursor_y = body_area.y + (cursor_row.saturating_sub(viewport_start)) as u16;
        if cursor_x < body_area.x + body_area.width && cursor_y < body_area.y + body_area.height {
            f.set_cursor_position(Position::new(cursor_x, cursor_y));
        }
    }
}

fn composer_line(
    app: &App,
    row: usize,
    line: &&str,
    cursor_row: usize,
    cursor_col: usize,
) -> Line<'static> {
    let prefix = if row == 0 { "❯ " } else { "  " };
    let prefix_style = if app.busy {
        Style::default().fg(C_SUBTLE)
    } else if row == 0 {
        Style::default()
            .fg(C_GOLD)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(C_SUBTLE)
    };
    let text_style = if app.busy {
        Style::default().fg(C_SUBTLE)
    } else {
        Style::default().fg(Color::White)
    };

    let mut spans = vec![Span::styled(prefix.to_string(), prefix_style)];

    if app.input.is_empty() && row == 0 {
        spans.push(Span::styled(
            "Type a message or paste code…".to_string(),
            Style::default()
                .fg(C_SUBTLE)
                .add_modifier(Modifier::ITALIC),
        ));
        return Line::from(spans);
    }

    if app.busy || row != cursor_row {
        spans.push(Span::styled((*line).to_string(), text_style));
        return Line::from(spans);
    }

    let chars: Vec<char> = line.chars().collect();
    let split_at = cursor_col.min(chars.len());
    let before: String = chars[..split_at].iter().collect();
    let current = chars
        .get(split_at)
        .map(|ch| ch.to_string())
        .unwrap_or_else(|| " ".to_string());
    let after: String = if split_at < chars.len() {
        chars[split_at + 1..].iter().collect()
    } else {
        String::new()
    };

    spans.push(Span::styled(before, text_style));
    spans.push(Span::styled(
        current,
        Style::default().fg(Color::Black).bg(Color::White),
    ));
    spans.push(Span::styled(after, text_style));
    Line::from(spans)
}

fn composer_footer_line(app: &App, max_width: usize) -> Line<'static> {
    let mut left: Vec<Span<'static>> = Vec::new();
    let hints = if app.busy {
        [
            ("Esc", "interrupt"),
            ("Ctrl+O", "trace"),
            ("Ctrl+R", "inspector"),
        ]
    } else {
        [
            ("Enter", "send"),
            ("Shift+Enter", "newline"),
            ("Ctrl+O", "trace"),
        ]
    };

    for (idx, (chord, label)) in hints.iter().enumerate() {
        if idx > 0 {
            push_sep(&mut left);
        }
        left.push(Span::styled(
            (*chord).to_string(),
            Style::default().fg(C_MUTED),
        ));
        left.push(Span::raw(" "));
        left.push(Span::styled(
            (*label).to_string(),
            Style::default().fg(C_SUBTLE),
        ));
    }

    let live_step = app.live_step_summary(max_width / 2);
    let mut right = Vec::new();
    push_chip(&mut right, "step", &live_step.text, live_step.state.color());

    if spans_width(&left) + 2 + spans_width(&right) <= max_width {
        let gap = max_width.saturating_sub(spans_width(&left) + spans_width(&right));
        left.push(Span::raw(" ".repeat(gap.max(2))));
        left.extend(right);
        return Line::from(left);
    }

    let compact = format!(
        "{}  [{}]",
        if app.busy {
            "Esc interrupt · Ctrl+O trace"
        } else {
            "Enter send · Shift+Enter newline"
        },
        live_step.text
    );
    Line::from(Span::styled(
        clip_to_width(&compact, max_width.saturating_sub(1)),
        Style::default().fg(Color::Gray),
    ))
}

fn draw_slash_popup(f: &mut Frame, app: &mut App, area: Rect, items: &[SlashPopupEntry]) {
    let Some(inner) = begin_panel(
        f,
        app,
        area,
        PanelSpec::new(" Commands ", PanelFocus::SlashPopup),
    ) else {
        return;
    };

    let selected = app.slash_popup_selected().min(items.len().saturating_sub(1));
    let item_rects = (0..items.len())
        .take(inner.height as usize)
        .map(|idx| Rect {
            x: inner.x,
            y: inner.y + idx as u16,
            width: inner.width,
            height: 1,
        })
        .collect::<Vec<_>>();
    app.register_slash_popup(area, item_rects);

    let list_items: Vec<ListItem> = items
        .iter()
        .enumerate()
        .map(|(idx, item)| {
            let active = idx == selected;
            let marker = if active { "▶ " } else { "  " };
            let marker_style = Style::default().fg(C_GOLD).add_modifier(if active {
                Modifier::BOLD
            } else {
                Modifier::empty()
            });
            let cmd_style = Style::default().fg(C_TEAL).add_modifier(if active {
                Modifier::BOLD
            } else {
                Modifier::empty()
            });
            let category_style = Style::default().fg(C_PURPLE).add_modifier(Modifier::ITALIC);
            let desc_style = Style::default().fg(if active { Color::White } else { C_MUTED });
            let usage_style = Style::default()
                .fg(if active { C_GOLD } else { C_SUBTLE })
                .add_modifier(Modifier::DIM);

            let category = slash_popup_category(&item.command);
            let header_line = Line::from(vec![
                Span::styled(marker.to_string(), marker_style),
                Span::styled(item.command.clone(), cmd_style),
                Span::raw(" "),
                Span::styled(format!("[{category}]"), category_style),
            ]);

            let body_line = Line::from(vec![
                Span::styled(item.usage.clone(), usage_style),
                Span::raw("  "),
                Span::styled(item.description.clone(), desc_style),
            ]);

            ListItem::new(vec![header_line, body_line])
        })
        .collect();

    let mut list_state = ListState::default();
    list_state.select(Some(selected));
    f.render_stateful_widget(
        List::new(list_items)
            .highlight_style(
                Style::default()
                    .bg(C_PANEL_FOCUS_BG)
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )
            .highlight_symbol(" "),
        inner,
        &mut list_state,
    );
}

fn slash_popup_category(command: &str) -> &'static str {
    let key = command.trim_start_matches('/').split_whitespace().next().unwrap_or("");
    match key {
        "repo" => "repo",
        "rag" => "rag",
        "tool" => "tool",
        "context" => "context",
        "agent" => "agent",
        "skill" => "skill",
        "wiki" => "wiki",
        "help" | "?" => "help",
        "status" | "trace" | "clear" | "quit" | "exit" | "q" => "meta",
        "load" | "save" | "export" | "import" | "mount" | "upload" => "session",
        "new" | "reload" | "reset" | "compact" => "workflow",
        _ => "misc",
    }
}

fn panel_border_style(app: &App, panel: PanelFocus) -> Style {
    if app.is_panel_focused(panel) {
        Style::default()
            .fg(C_ACCENT)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(C_BORDER)
    }
}

fn panel_background_style(app: &App, panel: PanelFocus) -> Style {
    let mut style = Style::default().bg(C_PANEL_BG).fg(Color::White);
    if app.is_panel_focused(panel) {
        style = style.bg(C_PANEL_FOCUS_BG).add_modifier(Modifier::BOLD);
    }
    style
}

// ── Status bar ────────────────────────────────────────────────────────────────

fn draw_status(f: &mut Frame, app: &App, area: Rect) {
    if area.height == 0 {
        return;
    }

    // Top border row
    let [border_area, text_area] = {
        let c = Layout::vertical([
            Constraint::Length(STATUS_BORDER_H),
            Constraint::Length(STATUS_TEXT_H),
        ])
        .split(area);
        [c[0], c[1]]
    };

    f.render_widget(
        Paragraph::new(status_border_line(area.width, app.phase.color()))
            .style(Style::default().bg(C_PANEL_BG)),
        border_area,
    );

    let lines = status_lines(app, area.width as usize, text_area.height as usize);
    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .style(Style::default().bg(C_PANEL_BG)),
        text_area,
    );
}

fn status_border_line(width: u16, _phase_color: Color) -> Line<'static> {
    let w = width as usize;
    if w == 0 {
        return Line::raw("");
    }
    let label = " ⧉ LOGICIAN " ;
    if w <= label.len() {
        return Line::from(Span::styled(
            "─".repeat(w),
            Style::default().fg(C_BORDER),
        ));
    }
    let dash_fill = "─".repeat(w - label.len());
    Line::from(vec![
        Span::styled(label.to_string(), Style::default().fg(C_PANEL_TITLE).add_modifier(Modifier::BOLD)),
        Span::styled(dash_fill, Style::default().fg(C_SUBTLE)),
    ])
}

fn status_line(app: &App, max_width: usize) -> Line<'static> {
    let live_step = app.live_step_summary((max_width / 3).max(18));
    if max_width < 52 {
        let compact = format!(
            "{} {} · {} · {}",
            if app.connected { "online" } else { "offline" },
            app.phase,
            live_step.text,
            app.bridge_state.active
        );
        return Line::from(Span::styled(compact, Style::default().fg(Color::Gray)));
    }

    let mut spans: Vec<Span<'static>> = Vec::new();
    push_chip(
        &mut spans,
        "net",
        if app.connected { "online" } else { "offline" },
        if app.connected {
            Color::LightGreen
        } else {
            Color::LightRed
        },
    );
    push_sep(&mut spans);

    let phase_label = if app.busy {
        format!("{} {}", spinner_char(app.spinner_tick), app.phase)
    } else {
        app.phase.to_string()
    };
    push_chip(&mut spans, "phase", &phase_label, app.phase.color());
    push_sep(&mut spans);
    push_chip(&mut spans, "step", &live_step.text, live_step.state.color());
    push_gap(&mut spans);

    push_kv(
        &mut spans,
        "agent",
        &app.bridge_state.active,
        Color::LightBlue,
    );
    push_sep(&mut spans);
    push_kv(
        &mut spans,
        "msgs",
        &app.bridge_state.msg_count.to_string(),
        Color::Yellow,
    );
    push_sep(&mut spans);
    push_kv(
        &mut spans,
        "tools",
        &app.bridge_state.tool_count.to_string(),
        Color::Cyan,
    );
    push_sep(&mut spans);
    push_kv(
        &mut spans,
        "skills",
        &app.active_skill_ids().len().to_string(),
        Color::Magenta,
    );
    push_sep(&mut spans);
    push_kv(
        &mut spans,
        "last",
        &app.last_turn_tool_count().to_string(),
        Color::Gray,
    );
    push_sep(&mut spans);
    push_kv(
        &mut spans,
        "calls",
        &app.bridge_state.tool_call_count.to_string(),
        Color::DarkGray,
    );
    if app.bridge_state.plan_mode {
        push_sep(&mut spans);
        push_chip(&mut spans, "mode", "plan", Color::Yellow);
    }
    let route_label = app.route_label();
    if route_label != "-" {
        push_sep(&mut spans);
        push_kv(&mut spans, "route", &route_label, Color::LightBlue);
    }
    push_gap(&mut spans);

    push_toggle(&mut spans, "trace", app.trace_on);
    push_sep(&mut spans);
    push_toggle(&mut spans, "image", app.has_image_preview());
    push_sep(&mut spans);
    push_toggle(&mut spans, "task", app.todo_on);
    push_sep(&mut spans);
    push_toggle(&mut spans, "context", app.context_on);
    push_sep(&mut spans);
    push_toggle(&mut spans, "rag", app.rag_on);
    push_sep(&mut spans);
    push_toggle(&mut spans, "panel", app.tool_output_on);

    let mut shortcut_spans = vec![Span::raw("  ")];
    for item in app.shortcut_hints().iter().take(6) {
        push_shortcut_hint(&mut shortcut_spans, item);
    }
    if spans_width(&spans) + spans_width(&shortcut_spans) <= max_width {
        spans.extend(shortcut_spans);
    }

    if spans_width(&spans) > max_width {
        let text = app.status_text();
        let clipped = clip_to_width(&text, max_width.saturating_sub(1));
        return Line::from(Span::styled(clipped, Style::default().fg(Color::Gray)));
    }
    Line::from(spans)
}

fn status_lines(app: &App, max_width: usize, max_lines: usize) -> Vec<Line<'static>> {
    if max_lines == 0 {
        return Vec::new();
    }

    let mut lines = vec![status_line(app, max_width)];
    if max_lines == 1 {
        return lines;
    }

    let progress = app.progress_entries();
    for entry in progress.into_iter().take(max_lines.saturating_sub(1)) {
        lines.push(progress_entry_line(&entry, max_width));
    }

    while lines.len() < max_lines {
        lines.push(Line::raw(""));
    }

    lines
}

fn progress_entry_line(entry: &ProgressEntry, max_width: usize) -> Line<'static> {
    if max_width < 24 {
        let prefix = if entry.is_last { "└─" } else { "├─" };
        let text = clip_to_width(
            &format!("{prefix} {}", entry.label),
            max_width.saturating_sub(1),
        );
        return Line::from(Span::styled(text, Style::default().fg(Color::Gray)));
    }

    let mut spans = Vec::new();
    spans.push(Span::styled(
        if entry.is_last { "└─ " } else { "├─ " },
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::DIM),
    ));
    spans.push(Span::styled(
        entry.label.clone(),
        Style::default()
            .fg(entry.state.color())
            .add_modifier(Modifier::BOLD),
    ));
    if let Some(meta) = entry.meta.as_deref() {
        spans.push(Span::styled(
            " · ",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM),
        ));
        spans.push(Span::styled(
            meta.to_string(),
            Style::default().fg(Color::White),
        ));
    }
    if let Some(detail) = entry.detail.as_deref() {
        spans.push(Span::styled(
            "  ",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM),
        ));
        spans.push(Span::styled(
            detail.to_string(),
            Style::default()
                .fg(Color::Gray)
                .add_modifier(Modifier::DIM),
        ));
    }

    if spans_width(&spans) > max_width {
        let mut compact = format!(
            "{}{}",
            if entry.is_last { "└─ " } else { "├─ " },
            entry.label
        );
        if let Some(meta) = entry.meta.as_deref() {
            compact.push_str(&format!(" · {meta}"));
        }
        if let Some(detail) = entry.detail.as_deref() {
            compact.push_str(&format!("  {detail}"));
        }
        return Line::from(Span::styled(
            clip_to_width(&compact, max_width.saturating_sub(1)),
            Style::default().fg(Color::Gray),
        ));
    }

    Line::from(spans)
}

fn spinner_char(tick: usize) -> &'static str {
    const SPINNERS: [&str; 10] = ["|", "/", "-", "\\", "|", "/", "-", "\\", "|", "/"];
    SPINNERS[tick % SPINNERS.len()]
}

fn push_chip(spans: &mut Vec<Span<'static>>, label: &str, value: &str, color: Color) {
    spans.push(Span::styled("[", Style::default().fg(C_SUBTLE)));
    spans.push(Span::styled(label.to_string(), Style::default().fg(C_MUTED)));
    spans.push(Span::styled(":", Style::default().fg(C_SUBTLE)));
    spans.push(Span::styled(
        value.to_string(),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    ));
    spans.push(Span::styled("]", Style::default().fg(C_SUBTLE)));
}

fn push_kv(spans: &mut Vec<Span<'static>>, key: &str, value: &str, color: Color) {
    spans.push(Span::styled(
        format!("{}:", key),
        Style::default().fg(C_MUTED),
    ));
    spans.push(Span::styled(
        value.to_string(),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    ));
}

fn push_toggle(spans: &mut Vec<Span<'static>>, key: &str, enabled: bool) {
    let color = if enabled { C_GREEN } else { C_SUBTLE };
    let value = if enabled { "on" } else { "off" };
    push_kv(spans, key, value, color);
}

fn push_gap(spans: &mut Vec<Span<'static>>) {
    spans.push(Span::raw("  "));
}

fn push_sep(spans: &mut Vec<Span<'static>>) {
    spans.push(Span::styled(
        " · ",
        Style::default().fg(C_HINT),
    ));
}

fn push_shortcut_hint(spans: &mut Vec<Span<'static>>, item: &ShortcutHint) {
    if spans.len() > 1 {
        push_sep(spans);
    }
    spans.push(Span::styled(
        item.chord.to_string(),
        Style::default().fg(C_MUTED),
    ));
    spans.push(Span::raw(" "));
    spans.push(Span::styled(
        item.description.to_string(),
        Style::default().fg(C_SUBTLE),
    ));
}

fn spans_width(spans: &[Span<'_>]) -> usize {
    spans.iter().map(|s| s.content.chars().count()).sum()
}

fn clip_to_width(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        return text.to_string();
    }
    if max_chars == 1 {
        return "…".to_string();
    }
    let mut out = String::with_capacity(max_chars);
    for ch in chars.into_iter().take(max_chars - 1) {
        out.push(ch);
    }
    out.push('…');
    out
}

// ── Tool output block parsing and rendering ──────────────────────────────────

/// Parse tool output into lines with spans for syntax highlighting
/// Returns a vector of Line objects
fn parse_tool_output_blocks(text: &str, _max_width: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    let mut blocks = text
        .split("\n\n")
        .map(str::trim)
        .filter(|block| !block.is_empty())
        .peekable();

    while let Some(block) = blocks.next() {
        if let Ok(value) = serde_json::from_str::<Value>(block) {
            lines.extend(render_json_block(&value));
        } else {
            for line in block.lines() {
                lines.push(Line::from(Span::styled(
                    line.to_string(),
                    Style::default().fg(Color::White),
                )));
            }
        }

        if blocks.peek().is_some() {
            lines.push(Line::raw(""));
            lines.push(Line::from(Span::styled(
                "─".repeat(24),
                Style::default().fg(C_SUBTLE),
            )));
            lines.push(Line::raw(""));
        }
    }

    lines
}

fn render_json_block(value: &Value) -> Vec<Line<'static>> {
    serde_json::to_string_pretty(value)
        .unwrap_or_else(|_| value.to_string())
        .lines()
        .map(render_json_line)
        .collect()
}

fn render_json_line(line: &str) -> Line<'static> {
    let indent_len = line.chars().take_while(|ch| ch.is_whitespace()).count();
    let (indent, rest) = line.split_at(indent_len);
    let mut spans = Vec::new();

    if !indent.is_empty() {
        spans.push(Span::styled(
            indent.to_string(),
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM),
        ));
    }

    if let Some(key_end) = rest.find("\":") {
        let (key, tail_with_colon) = rest.split_at(key_end + 1);
        spans.push(Span::styled(
            key.to_string(),
            Style::default()
                .fg(C_GOLD)
                .add_modifier(Modifier::BOLD),
        ));
        spans.extend(tokenize_json_tail(tail_with_colon));
    } else {
        spans.extend(tokenize_json_tail(rest));
    }

    Line::from(spans)
}

fn tokenize_json_tail(text: &str) -> Vec<Span<'static>> {
    let chars: Vec<char> = text.chars().collect();
    let mut spans = Vec::new();
    let mut idx = 0;

    while idx < chars.len() {
        let ch = chars[idx];

        if ch.is_whitespace() {
            let start = idx;
            while idx < chars.len() && chars[idx].is_whitespace() {
                idx += 1;
            }
            spans.push(Span::raw(chars[start..idx].iter().collect::<String>()));
            continue;
        }

        if matches!(ch, '{' | '}' | '[' | ']' | ':' | ',') {
            spans.push(Span::styled(
                ch.to_string(),
                Style::default().fg(C_SUBTLE),
            ));
            idx += 1;
            continue;
        }

        if ch == '"' {
            let start = idx;
            idx += 1;
            let mut escaped = false;
            while idx < chars.len() {
                let current = chars[idx];
                idx += 1;
                if current == '"' && !escaped {
                    break;
                }
                escaped = current == '\\' && !escaped;
                if current != '\\' {
                    escaped = false;
                }
            }
            spans.push(Span::styled(
                chars[start..idx].iter().collect::<String>(),
                Style::default().fg(C_GREEN),
            ));
            continue;
        }

        if ch.is_ascii_digit() || ch == '-' {
            let start = idx;
            idx += 1;
            while idx < chars.len() && matches!(chars[idx], '0'..='9' | '.' | 'e' | 'E' | '+' | '-')
            {
                idx += 1;
            }
            spans.push(Span::styled(
                chars[start..idx].iter().collect::<String>(),
                Style::default().fg(C_ORANGE),
            ));
            continue;
        }

        if ch.is_ascii_alphabetic() {
            let start = idx;
            idx += 1;
            while idx < chars.len() && (chars[idx].is_ascii_alphanumeric() || chars[idx] == '_') {
                idx += 1;
            }
            let token = chars[start..idx].iter().collect::<String>();
            let style = match token.as_str() {
                "true" | "false" => Style::default().fg(C_PURPLE),
                "null" => Style::default()
                    .fg(C_SUBTLE)
                    .add_modifier(Modifier::ITALIC),
                _ => Style::default().fg(Color::White),
            };
            spans.push(Span::styled(token, style));
            continue;
        }

        spans.push(Span::styled(
            ch.to_string(),
            Style::default().fg(Color::White),
        ));
        idx += 1;
    }

    spans
}
