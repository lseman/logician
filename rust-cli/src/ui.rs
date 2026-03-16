use crate::app::{App, SlashPopupEntry};
use ratatui::{
    layout::{Constraint, Layout, Position, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};
use std::path::Path;

// ── Layout constants ──────────────────────────────────────────────────────────

const POPUP_BORDER_H: u16 = 2;
const INPUT_H: u16 = 1;
const STATUS_BORDER_H: u16 = 1;
const STATUS_TEXT_H: u16 = 1;
const STATUS_TOTAL_H: u16 = STATUS_BORDER_H + STATUS_TEXT_H;

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn draw(f: &mut Frame, app: &mut App) {
    let area = f.area();

    let popup_items = app.slash_popup_entries(8);
    let show_popup = !app.busy && app.slash_popup_visible() && !popup_items.is_empty();

    let [msgs_area, popup_area, input_area, status_area] =
        compute_layout(area, show_popup, popup_items.len());
    let (chat_area, panels_area) = split_chat_and_panels(msgs_area, app.has_image_preview(), app.todo_on);
    let (image_area, todo_area) = split_panels(panels_area, app.has_image_preview(), app.todo_on);

    // Side-effect kept here but isolated — feeds scroll logic for this frame.
    app.update_area_h(chat_area.height);

    draw_messages(f, app, chat_area);
    if let Some(img_area) = image_area {
        draw_image_panel(f, app, img_area);
    }
    if let Some(td_area) = todo_area {
        draw_todo_panel(f, app, td_area);
    }
    if show_popup {
        draw_slash_popup(f, app, popup_area, &popup_items);
    }
    draw_input(f, app, input_area);
    draw_status(f, app, status_area);
}

// ── Layout computation (pure, testable) ───────────────────────────────────────

fn compute_layout(area: Rect, show_popup: bool, popup_item_count: usize) -> [Rect; 4] {
    let popup_h: u16 = if show_popup {
        (popup_item_count as u16 + POPUP_BORDER_H).min(10)
    } else {
        0
    };

    let fixed_h = popup_h + INPUT_H + STATUS_TOTAL_H;
    let msgs_h = area.height.saturating_sub(fixed_h);

    // Always allocate four rows; zero-height areas are safe throughout.
    let chunks = Layout::vertical([
        Constraint::Length(msgs_h),
        Constraint::Length(popup_h),
        Constraint::Length(INPUT_H),
        Constraint::Length(STATUS_TOTAL_H),
    ])
    .split(area);

    [chunks[0], chunks[1], chunks[2], chunks[3]]
}

fn split_chat_and_panels(area: Rect, show_image: bool, show_todo: bool) -> (Rect, Option<Rect>) {
    if (!show_image && !show_todo) || area.height < 6 || area.width < 50 {
        return (area, None);
    }

    let panel_pct = if show_image && show_todo { 45 } else { 38 };

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

fn split_panels(area: Option<Rect>, show_image: bool, show_todo: bool) -> (Option<Rect>, Option<Rect>) {
    let Some(area) = area else {
        return (None, None);
    };

    if show_image && show_todo {
        // Prefer vertical split for panels if horizontal space is tight, 
        // but here we are already in the "side" area.
        let chunks = if area.width > area.height * 2 {
            Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).split(area)
        } else {
            Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)]).split(area)
        };
        return (Some(chunks[0]), Some(chunks[1]));
    }

    if show_image {
        (Some(area), None)
    } else {
        (None, Some(area))
    }
}

// ── Messages panel ────────────────────────────────────────────────────────────

fn draw_messages(f: &mut Frame, app: &App, area: Rect) {
    if area.height == 0 {
        return;
    }
    let para = Paragraph::new(app.visible_lines(area.height)).wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

fn draw_image_panel(f: &mut Frame, app: &mut App, area: Rect) {
    if area.width < 4 || area.height < 4 {
        return;
    }

    let path = app.image_preview_path().unwrap_or("").to_string();
    let file_name = Path::new(&path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("image");
    let title = format!(" Image: {file_name} ");
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.width < 2 || inner.height < 2 {
        return;
    }

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

fn draw_todo_panel(f: &mut Frame, app: &App, area: Rect) {
    if area.width < 4 || area.height < 4 {
        return;
    }

    let block = Block::default()
        .title(" Task Checklist ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut list_items = Vec::new();
    if app.bridge_state.todo.is_empty() {
        list_items.push(ListItem::new(Line::from(vec![Span::styled(
            "  No active tasks.",
            Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
        )])));
    } else {
        for item in &app.bridge_state.todo {
            let symbol = match item.status.as_str() {
                "completed" | "done" => " [x] ",
                "in-progress" | "doing" => " [/] ",
                _ => " [ ] ",
            };
            let color = match item.status.as_str() {
                "completed" | "done" => Color::Green,
                "in-progress" | "doing" => Color::Yellow,
                _ => Color::Gray,
            };

            list_items.push(ListItem::new(Line::from(vec![
                Span::styled(symbol, Style::default().fg(color).add_modifier(Modifier::BOLD)),
                Span::styled(item.title.clone(), Style::default().fg(Color::White)),
            ])));
            if !item.note.is_empty() {
                list_items.push(ListItem::new(Line::from(vec![
                    Span::styled("     ", Style::default()),
                    Span::styled(
                        item.note.clone(),
                        Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM),
                    ),
                ])));
            }
        }
    }

    f.render_widget(List::new(list_items), inner);
}

// ── Input line ────────────────────────────────────────────────────────────────

fn draw_input(f: &mut Frame, app: &App, area: Rect) {
    const PROMPT: &str = "❯ ";
    const PROMPT_VISUAL_WIDTH: u16 = 2;

    let spans = input_spans(app.busy, PROMPT, &app.input, app.input_byte_offset());
    f.render_widget(Paragraph::new(Line::from(spans)), area);

    if !app.busy {
        // Count chars (not bytes) so multi-byte characters don't misalign
        // the terminal cursor. For true wide-char (CJK/emoji) support you'd
        // want unicode-width, but that conflicts with ratatui 0.29's pinned
        // =0.2.0 — char count is correct for all BMP single-width codepoints.
        let before = &app.input[..app.input_byte_offset()];
        let visual_col = before.chars().count() as u16;
        let cursor_x = area.x + PROMPT_VISUAL_WIDTH + visual_col;
        let cursor_y = area.y;
        if cursor_x < area.x + area.width {
            f.set_cursor_position(Position::new(cursor_x, cursor_y));
        }
    }
}

/// Pure span builder — easy to unit-test independently of Frame.
fn input_spans(busy: bool, prompt: &str, input: &str, byte_off: usize) -> Vec<Span<'static>> {
    if busy {
        return vec![
            Span::styled(prompt.to_string(), Style::default().fg(Color::DarkGray)),
            Span::styled(input.to_string(), Style::default().fg(Color::DarkGray)),
        ];
    }

    let before = input[..byte_off].to_string();
    let after = &input[byte_off..];
    let cursor_char = after
        .chars()
        .next()
        .map(|c| c.to_string())
        .unwrap_or_else(|| " ".to_string());
    let rest: String = after.chars().skip(1).collect();

    vec![
        Span::styled(prompt.to_string(), Style::default().fg(Color::Yellow)),
        Span::styled(before, Style::default().fg(Color::White)),
        Span::styled(
            cursor_char,
            Style::default().fg(Color::Black).bg(Color::White),
        ),
        Span::styled(rest, Style::default().fg(Color::White)),
    ]
}

// ── Slash-command popup ───────────────────────────────────────────────────────

fn draw_slash_popup(f: &mut Frame, app: &App, area: Rect, items: &[SlashPopupEntry]) {
    if area.height < POPUP_BORDER_H || items.is_empty() {
        return;
    }

    let block = Block::default()
        .title(" Commands ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    f.render_widget(block, area);

    // Selection clamping belongs in App; we only render what we're given.
    let selected = app.slash_popup_selected();

    let list_items: Vec<ListItem> = items
        .iter()
        .enumerate()
        .map(|(idx, item)| {
            let active = idx == selected;
            let marker = if active { "▶ " } else { "  " };
            let marker_style = Style::default().fg(Color::Yellow).add_modifier(if active {
                Modifier::BOLD
            } else {
                Modifier::empty()
            });
            let cmd_style = Style::default().fg(Color::Cyan).add_modifier(if active {
                Modifier::BOLD
            } else {
                Modifier::empty()
            });
            let desc_style = Style::default().fg(if active { Color::White } else { Color::Gray });

            ListItem::new(Line::from(vec![
                Span::styled(marker.to_string(), marker_style),
                Span::styled(format!("{:<12}", item.command), cmd_style),
                Span::styled(item.description.to_string(), desc_style),
            ]))
        })
        .collect();

    let mut list_state = ListState::default();
    list_state.select(Some(selected));

    f.render_stateful_widget(List::new(list_items), inner, &mut list_state);
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
        Paragraph::new(status_border_line(area.width, app.phase.color())),
        border_area,
    );

    let line = status_line(app, area.width as usize);
    f.render_widget(Paragraph::new(line), text_area);
}

fn status_border_line(width: u16, phase_color: Color) -> Line<'static> {
    let w = width as usize;
    if w == 0 {
        return Line::raw("");
    }
    let label = " STATUS ";
    if w <= label.len() {
        return Line::from(Span::styled(
            "─".repeat(w),
            Style::default().fg(Color::DarkGray),
        ));
    }
    let mut text = String::with_capacity(w);
    text.push_str(label);
    text.push_str(&"─".repeat(w - label.len()));
    Line::from(Span::styled(
        text,
        Style::default()
            .fg(phase_color)
            .add_modifier(Modifier::DIM),
    ))
}

fn status_line(app: &App, max_width: usize) -> Line<'static> {
    if max_width < 52 {
        let compact = format!(
            "{} {} · {} · {}",
            if app.connected { "online" } else { "offline" },
            app.phase,
            app.phase_note,
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
            Color::Green
        } else {
            Color::Red
        },
    );
    push_sep(&mut spans);

    let phase_label = if app.busy {
        format!("{} {}", spinner_char(app.spinner_tick), app.phase)
    } else {
        app.phase.to_string()
    };
    push_chip(&mut spans, "phase", &phase_label, app.phase.color());
    if !app.phase_note.trim().is_empty() {
        push_sep(&mut spans);
        push_kv(&mut spans, "note", &app.phase_note, Color::Gray);
    }
    push_gap(&mut spans);

    push_kv(&mut spans, "agent", &app.bridge_state.active, Color::LightBlue);
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
        &app.bridge_state.skill_count.to_string(),
        Color::Magenta,
    );
    push_sep(&mut spans);
    push_kv(
        &mut spans,
        "last",
        &app.last_turn_tool_count().to_string(),
        Color::Gray,
    );
    push_gap(&mut spans);

    push_toggle(&mut spans, "trace", app.trace_on);
    push_sep(&mut spans);
    push_toggle(&mut spans, "image", app.has_image_preview());
    push_sep(&mut spans);
    push_toggle(&mut spans, "task", app.todo_on);

    let hint = vec![
        Span::raw("  "),
        Span::styled("Ctrl+O", Style::default().fg(Color::Gray)),
        Span::raw("/"),
        Span::styled("Ctrl+P", Style::default().fg(Color::Gray)),
        Span::raw("/"),
        Span::styled("Ctrl+T", Style::default().fg(Color::Gray)),
    ];
    if spans_width(&spans) + spans_width(&hint) <= max_width {
        spans.extend(hint);
    }

    if spans_width(&spans) > max_width {
        let text = app.status_text();
        let clipped = clip_to_width(&text, max_width.saturating_sub(1));
        return Line::from(Span::styled(clipped, Style::default().fg(Color::Gray)));
    }
    Line::from(spans)
}

fn spinner_char(tick: usize) -> &'static str {
    const SPINNERS: [&str; 10] = ["|", "/", "-", "\\", "|", "/", "-", "\\", "|", "/"];
    SPINNERS[tick % SPINNERS.len()]
}

fn push_chip(spans: &mut Vec<Span<'static>>, label: &str, value: &str, color: Color) {
    spans.push(Span::styled(
        "[",
        Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM),
    ));
    spans.push(Span::styled(
        label.to_string(),
        Style::default().fg(Color::Gray).add_modifier(Modifier::DIM),
    ));
    spans.push(Span::styled(
        ":",
        Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM),
    ));
    spans.push(Span::styled(
        value.to_string(),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    ));
    spans.push(Span::styled(
        "]",
        Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM),
    ));
}

fn push_kv(spans: &mut Vec<Span<'static>>, key: &str, value: &str, color: Color) {
    spans.push(Span::styled(
        format!("{}:", key),
        Style::default().fg(Color::DarkGray),
    ));
    spans.push(Span::styled(
        value.to_string(),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    ));
}

fn push_toggle(spans: &mut Vec<Span<'static>>, key: &str, enabled: bool) {
    let color = if enabled { Color::Green } else { Color::DarkGray };
    let value = if enabled { "on" } else { "off" };
    push_kv(spans, key, value, color);
}

fn push_gap(spans: &mut Vec<Span<'static>>) {
    spans.push(Span::raw("  "));
}

fn push_sep(spans: &mut Vec<Span<'static>>) {
    spans.push(Span::styled(
        " · ",
        Style::default().fg(Color::DarkGray).add_modifier(Modifier::DIM),
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
