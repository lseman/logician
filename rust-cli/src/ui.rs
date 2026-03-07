use ratatui::{
    layout::{Constraint, Layout, Position, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::app::{App, SlashPopupEntry};

pub fn draw(f: &mut Frame, app: &mut App) {
    let area = f.area();

    let popup_items = app.slash_popup_entries(8);
    let show_popup = !app.busy && app.slash_popup_visible() && !popup_items.is_empty();
    let popup_h: u16 = if show_popup {
        (popup_items.len() as u16 + 2).min(10)
    } else {
        0
    };

    // bottom section: optional popup + input (1) + status bar (1 + border = 3)
    let bottom_h: u16 = 4 + popup_h;
    let msgs_h = area.height.saturating_sub(bottom_h);

    let chunks = if show_popup {
        Layout::vertical([
            Constraint::Length(msgs_h),
            Constraint::Length(popup_h),
            Constraint::Length(1),
            Constraint::Length(3),
        ])
        .split(area)
    } else {
        Layout::vertical([
            Constraint::Length(msgs_h),
            Constraint::Length(1),
            Constraint::Length(3),
        ])
        .split(area)
    };

    app.update_area_h(msgs_h);

    if show_popup {
        draw_messages(f, app, chunks[0]);
        draw_slash_popup(f, app, chunks[1], &popup_items);
        draw_input(f, app, chunks[2]);
        draw_status(f, app, chunks[3]);
    } else {
        draw_messages(f, app, chunks[0]);
        draw_input(f, app, chunks[1]);
        draw_status(f, app, chunks[2]);
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

// ── Input line ────────────────────────────────────────────────────────────────

fn draw_input(f: &mut Frame, app: &App, area: Rect) {
    let prompt = "❯ ";
    let prompt_width = 2u16;

    // Build display text with cursor indicator
    let byte_off = app.input_byte_offset();
    let (before, after) = app.input.split_at(byte_off);

    let spans = if app.busy {
        vec![
            Span::styled(prompt, Style::default().fg(Color::DarkGray)),
            Span::styled(app.input.clone(), Style::default().fg(Color::DarkGray)),
        ]
    } else {
        let after_display = if after.is_empty() { " " } else { after };
        vec![
            Span::styled(prompt, Style::default().fg(Color::Yellow)),
            Span::styled(before.to_string(), Style::default().fg(Color::White)),
            Span::styled(
                after_display
                    .chars()
                    .next()
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| " ".to_string()),
                Style::default().fg(Color::Black).bg(Color::White),
            ),
            Span::styled(
                after.chars().skip(1).collect::<String>(),
                Style::default().fg(Color::White),
            ),
        ]
    };

    let line = Line::from(spans);
    let para = Paragraph::new(vec![line]);
    f.render_widget(para, area);

    // Position terminal cursor
    if !app.busy {
        let cursor_x = area.x + prompt_width + app.input_cursor as u16;
        let cursor_y = area.y;
        if cursor_x < area.x + area.width {
            f.set_cursor_position(Position::new(cursor_x, cursor_y));
        }
    }
}

fn draw_slash_popup(f: &mut Frame, app: &App, area: Rect, items: &[SlashPopupEntry]) {
    if area.height < 2 || items.is_empty() {
        return;
    }

    let block = Block::default()
        .title(" Commands ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let selected = app
        .slash_popup_selected()
        .min(items.len().saturating_sub(1));

    let lines: Vec<Line<'static>> = items
        .iter()
        .enumerate()
        .map(|(idx, item)| {
            let active = idx == selected;
            let marker = if active { "▶ " } else { "  " };
            let marker_style = if active {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::DarkGray)
            };
            let cmd_style = if active {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Cyan)
            };
            let desc_style = if active {
                Style::default().fg(Color::White)
            } else {
                Style::default().fg(Color::Gray)
            };
            Line::from(vec![
                Span::styled(marker.to_string(), marker_style),
                Span::styled(format!("{:<12}", item.command), cmd_style),
                Span::styled(item.description.to_string(), desc_style),
            ])
        })
        .collect();

    let para = Paragraph::new(lines).wrap(Wrap { trim: true });
    f.render_widget(para, inner);
}

// ── Status bar ────────────────────────────────────────────────────────────────

fn draw_status(f: &mut Frame, app: &App, area: Rect) {
    let text = app.status_text();
    let color = app.phase.color();

    let block = Block::default()
        .borders(Borders::TOP)
        .border_style(Style::default().fg(Color::DarkGray));

    let inner = block.inner(area);
    f.render_widget(block, area);

    let para = Paragraph::new(Line::from(vec![Span::styled(
        text,
        Style::default().fg(color),
    )]));
    f.render_widget(para, inner);
}
