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
    let (chat_area, image_area) = split_chat_and_image(msgs_area, app.has_image_preview());

    // Side-effect kept here but isolated — feeds scroll logic for this frame.
    app.update_area_h(chat_area.height);

    draw_messages(f, app, chat_area);
    if let Some(img_area) = image_area {
        draw_image_panel(f, app, img_area);
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

fn split_chat_and_image(area: Rect, show_image: bool) -> (Rect, Option<Rect>) {
    if !show_image || area.height < 6 || area.width < 40 {
        return (area, None);
    }

    if area.width < 90 {
        let chunks =
            Layout::vertical([Constraint::Percentage(64), Constraint::Percentage(36)]).split(area);
        return (chunks[0], Some(chunks[1]));
    }

    let chunks =
        Layout::horizontal([Constraint::Percentage(62), Constraint::Percentage(38)]).split(area);
    (chunks[0], Some(chunks[1]))
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
        Paragraph::new(Line::from("─".repeat(area.width as usize)))
            .style(Style::default().fg(Color::DarkGray)),
        border_area,
    );

    let text = app.status_text();
    let color = app.phase.color();
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(text, Style::default().fg(color)))),
        text_area,
    );
}
