use crate::app::{App, PanelFocus, ShortcutHint, SlashPopupEntry};
use crate::markdown::render_diff;
use ratatui::{
    layout::{Constraint, Layout, Position, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
    Frame,
};
use serde_json::Value;
use std::path::Path;

// ── Layout constants ──────────────────────────────────────────────────────────

const POPUP_BORDER_H: u16 = 2;
const INPUT_BORDER_H: u16 = 2;
const INPUT_MIN_H: u16 = INPUT_BORDER_H + 1;
const STATUS_BORDER_H: u16 = 1;
const STATUS_TEXT_H: u16 = 1;
const STATUS_TOTAL_H: u16 = STATUS_BORDER_H + STATUS_TEXT_H;

// ── Entry point ───────────────────────────────────────────────────────────────

pub fn draw(f: &mut Frame, app: &mut App) {
    let area = f.area();
    app.begin_frame();

    let popup_items = app.slash_popup_entries(8);
    let show_popup = !app.busy && app.slash_popup_visible() && !popup_items.is_empty();
    let requested_input_h = (app.input_line_count() as u16)
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
    );
    let (image_area, todo_area, tool_output_area, context_area) = split_panels(
        panels_area,
        app.has_image_preview(),
        app.todo_on,
        app.tool_output_on,
        app.context_on,
    );

    // Side-effect kept here but isolated — feeds scroll logic for this frame.
    app.update_area_h(chat_area.height);

    draw_messages(f, app, chat_area);
    if let Some(img_area) = image_area {
        draw_image_panel(f, app, img_area);
    }
    if let Some(td_area) = todo_area {
        draw_todo_panel(f, app, td_area);
    }
    if let Some(to_area) = tool_output_area {
        draw_tool_output_panel(f, app, to_area);
    }
    if let Some(cx_area) = context_area {
        draw_context_panel(f, app, cx_area);
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
) -> (Rect, Option<Rect>) {
    let panel_count = [show_image, show_todo, show_tool_output, show_context]
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
) -> (Option<Rect>, Option<Rect>, Option<Rect>, Option<Rect>) {
    let Some(area) = area else {
        return (None, None, None, None);
    };

    #[derive(Clone, Copy)]
    enum PanelKind {
        Image,
        Todo,
        ToolOutput,
        Context,
    }

    let mut kinds = Vec::new();
    if show_image {
        kinds.push(PanelKind::Image);
    }
    if show_todo {
        kinds.push(PanelKind::Todo);
    }
    if show_tool_output {
        kinds.push(PanelKind::ToolOutput);
    }
    if show_context {
        kinds.push(PanelKind::Context);
    }

    if kinds.is_empty() {
        return (None, None, None, None);
    }

    let count = kinds.len() as u32;
    let constraints = (0..kinds.len())
        .map(|_| Constraint::Ratio(1, count))
        .collect::<Vec<_>>();
    let chunks = if area.width > area.height * 2 {
        Layout::horizontal(constraints).split(area)
    } else {
        Layout::vertical(constraints).split(area)
    };

    let mut image_area = None;
    let mut todo_area = None;
    let mut tool_output_area = None;
    let mut context_area = None;

    for (idx, kind) in kinds.into_iter().enumerate() {
        match kind {
            PanelKind::Image => image_area = Some(chunks[idx]),
            PanelKind::Todo => todo_area = Some(chunks[idx]),
            PanelKind::ToolOutput => tool_output_area = Some(chunks[idx]),
            PanelKind::Context => context_area = Some(chunks[idx]),
        }
    }

    (image_area, todo_area, tool_output_area, context_area)
}

// ── Messages panel ────────────────────────────────────────────────────────────

fn draw_messages(f: &mut Frame, app: &mut App, area: Rect) {
    if area.height == 0 {
        return;
    }
    app.register_messages_area(area);
    let para = Paragraph::new(app.visible_lines(area.height)).wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

fn draw_image_panel(f: &mut Frame, app: &mut App, area: Rect) {
    if area.width < 4 || area.height < 4 {
        return;
    }

    app.register_image_area(area);

    let path = app.image_preview_path().unwrap_or("").to_string();
    let file_name = Path::new(&path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("image");
    let title = format!(" Image: {file_name} ");
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(panel_border_style(app, PanelFocus::Image));
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

fn draw_todo_panel(f: &mut Frame, app: &mut App, area: Rect) {
    if area.width < 4 || area.height < 4 {
        return;
    }

    let block = Block::default()
        .title(" Task Checklist ")
        .borders(Borders::ALL)
        .border_style(panel_border_style(app, PanelFocus::Todo));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines: Vec<Line<'static>> = Vec::new();
    if app.bridge_state.todo.is_empty() {
        lines.push(Line::from(vec![Span::styled(
            "  No active tasks.",
            Style::default()
                .fg(Color::DarkGray)
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
                "completed" | "done" => Color::Green,
                "in-progress" | "in_progress" | "doing" => Color::Yellow,
                "blocked" => Color::Red,
                _ => Color::Gray,
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
    if area.width < 4 || area.height < 4 {
        return;
    }

    let title = if app.has_change_records() {
        format!(
            " Inspector (changes: {}) (Ctrl+R) ",
            app.changed_files().len()
        )
    } else {
        " Inspector (Ctrl+R) ".to_string()
    };
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(panel_border_style(app, PanelFocus::ToolOutput));
    let inner = block.inner(area);
    f.render_widget(block, area);

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

    if !app.active_skill_ids().is_empty() {
        styled_lines.push(Line::from(Span::styled(
            "request skills",
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        )));
        styled_lines.push(Line::from(vec![
            Span::styled("  active ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                app.active_skill_ids().join(", "),
                Style::default().fg(Color::Magenta),
            ),
        ]));
        if !app.active_selected_tools().is_empty() {
            styled_lines.push(Line::from(vec![
                Span::styled("  tools  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    app.active_selected_tools().join(", "),
                    Style::default().fg(Color::Cyan),
                ),
            ]));
        }
        styled_lines.push(Line::raw(""));
    }

    if app.trace_on && !app.trace_entries().is_empty() {
        styled_lines.push(Line::from(Span::styled(
            "recent trace",
            Style::default()
                .fg(Color::Yellow)
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
                .fg(Color::Cyan)
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
                .style(Style::default().fg(Color::DarkGray)),
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
    if area.width < 4 || area.height < 4 {
        return;
    }

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
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(panel_border_style(app, PanelFocus::Context));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut lines: Vec<Line<'static>> = Vec::new();
    lines.push(Line::from(vec![
        Span::styled(
            "session ",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM),
        ),
        Span::styled(
            app.bridge_state.session.clone(),
            Style::default().fg(Color::Cyan),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled(
            "messages ",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM),
        ),
        Span::styled(
            app.bridge_state.msg_count.to_string(),
            Style::default().fg(Color::White),
        ),
        Span::styled(
            "  total tokens≈ ",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::DIM),
        ),
        Span::styled(
            app.context_token_estimate_total().to_string(),
            Style::default().fg(Color::Yellow),
        ),
    ]));
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "loaded tools",
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    )));
    if app.bridge_state.loaded_tools.is_empty() {
        lines.push(Line::from(Span::styled(
            "  none",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        )));
    } else {
        for tool in &app.bridge_state.loaded_tools {
            lines.push(Line::from(vec![
                Span::styled("└─ ", Style::default().fg(Color::DarkGray)),
                Span::styled(tool.clone(), Style::default().fg(Color::Cyan)),
            ]));
        }
    }

    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "loaded skills",
        Style::default()
            .fg(Color::Magenta)
            .add_modifier(Modifier::BOLD),
    )));
    if app.bridge_state.loaded_skills.is_empty() {
        lines.push(Line::from(Span::styled(
            "  none",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        )));
    } else {
        for skill in &app.bridge_state.loaded_skills {
            lines.push(Line::from(vec![
                Span::styled("└─ ", Style::default().fg(Color::DarkGray)),
                Span::styled(skill.clone(), Style::default().fg(Color::Magenta)),
            ]));
        }
    }

    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "active repos",
        Style::default()
            .fg(Color::Yellow)
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
        "recent retrieval",
        Style::default()
            .fg(Color::LightCyan)
            .add_modifier(Modifier::BOLD),
    )));
    if app.bridge_state.retrieval_insights.is_empty() {
        lines.push(Line::from(Span::styled(
            "  none",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        )));
    } else {
        for insight in &app.bridge_state.retrieval_insights {
            let label = if !insight.repo_name.trim().is_empty() {
                if insight.repo_id.trim().is_empty() || insight.repo_name == insight.repo_id {
                    insight.repo_name.clone()
                } else {
                    format!("{} ({})", insight.repo_name, insight.repo_id)
                }
            } else if !insight.repo_id.trim().is_empty() {
                insight.repo_id.clone()
            } else {
                "retrieval".to_string()
            };
            lines.push(Line::from(vec![
                Span::styled("└─ ", Style::default().fg(Color::DarkGray)),
                Span::styled(label, Style::default().fg(Color::LightCyan)),
            ]));
            if !insight.query.trim().is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("   ", Style::default()),
                    Span::styled("query ", Style::default().fg(Color::DarkGray)),
                    Span::styled(insight.query.clone(), Style::default().fg(Color::White)),
                ]));
            }
            if !insight.seed_paths.is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("   ", Style::default()),
                    Span::styled("seeds ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        insight.seed_paths.join(", "),
                        Style::default().fg(Color::Yellow),
                    ),
                ]));
            }
            if !insight.retrieved_paths.is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("   ", Style::default()),
                    Span::styled("hits  ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        insight.retrieved_paths.join(", "),
                        Style::default().fg(Color::Cyan),
                    ),
                ]));
            }
            if !insight.related_files.is_empty() {
                let related = insight
                    .related_files
                    .iter()
                    .map(|item| {
                        if item.score > 0 {
                            format!("{} ({})", item.rel_path, item.score)
                        } else {
                            item.rel_path.clone()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                lines.push(Line::from(vec![
                    Span::styled("   ", Style::default()),
                    Span::styled("graph ", Style::default().fg(Color::DarkGray)),
                    Span::styled(related, Style::default().fg(Color::LightGreen)),
                ]));
            }
            if !insight.related_symbols.is_empty() {
                let symbols = insight
                    .related_symbols
                    .iter()
                    .map(|item| {
                        let mut label = item.name.clone();
                        if !item.rel_path.trim().is_empty() {
                            label.push('@');
                            label.push_str(&item.rel_path);
                        }
                        if item.line > 0 {
                            label.push(':');
                            label.push_str(&item.line.to_string());
                        }
                        if !item.symbol_kind.trim().is_empty() {
                            label.push_str(" [");
                            label.push_str(&item.symbol_kind);
                            label.push(']');
                        }
                        label
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                lines.push(Line::from(vec![
                    Span::styled("   ", Style::default()),
                    Span::styled("why   ", Style::default().fg(Color::DarkGray)),
                    Span::styled(symbols, Style::default().fg(Color::Magenta)),
                ]));
            }
        }
    }

    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "repo library",
        Style::default()
            .fg(Color::LightYellow)
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
            .fg(Color::LightBlue)
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

    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "active rag docs",
        Style::default()
            .fg(Color::LightGreen)
            .add_modifier(Modifier::BOLD),
    )));
    if app.bridge_state.rag_docs.is_empty() {
        lines.push(Line::from(Span::styled(
            "  none",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        )));
    } else {
        for item in &app.bridge_state.rag_docs {
            lines.push(Line::from(vec![
                Span::styled("└─ ", Style::default().fg(Color::DarkGray)),
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
                    Style::default().fg(Color::Green),
                ),
                Span::styled("  ", Style::default()),
                Span::styled(
                    format!("chunks {}", item.chunks),
                    Style::default().fg(Color::Cyan),
                ),
                Span::styled("  ", Style::default()),
                Span::styled(
                    format!("tokens≈ {}", item.token_count),
                    Style::default().fg(Color::Yellow),
                ),
            ]));
            if !item.label.trim().is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("   ", Style::default()),
                    Span::styled(
                        item.label.clone(),
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

fn render_changes_panel(app: &App, inner: Rect) -> (Vec<Line<'static>>, Vec<(usize, Rect)>) {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let mut change_headers = Vec::new();

    lines.push(Line::from(vec![
        Span::styled(
            "recent files",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(" ({})", app.changed_files().len()),
            Style::default().fg(Color::DarkGray),
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
                    Style::default().fg(if active {
                        Color::Yellow
                    } else {
                        Color::DarkGray
                    }),
                ),
                Span::styled(
                    file_name.to_string(),
                    Style::default().fg(if active { Color::Cyan } else { Color::White }),
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
        Color::DarkGray
    } else if app.is_panel_focused(PanelFocus::Input) {
        Color::Cyan
    } else {
        Color::Yellow
    };
    let block = Block::default()
        .title(" Compose ")
        .title_bottom(Line::from(vec![
            Span::styled(" Enter ", Style::default().fg(Color::Yellow)),
            Span::styled("send", Style::default().fg(Color::DarkGray)),
            Span::styled("  ", Style::default()),
            Span::styled("Shift+Enter", Style::default().fg(Color::Yellow)),
            Span::styled(" newline", Style::default().fg(Color::DarkGray)),
        ]))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color));
    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height == 0 || inner.width == 0 {
        return;
    }

    let raw_lines: Vec<&str> = if app.input.is_empty() {
        vec![""]
    } else {
        app.input.split('\n').collect()
    };
    let (cursor_row, cursor_col) = app.input_cursor_line_col();
    let viewport_h = inner.height as usize;
    let viewport_start = cursor_row.saturating_sub(viewport_h.saturating_sub(1));
    let visible_lines: Vec<Line<'static>> = raw_lines
        .iter()
        .enumerate()
        .skip(viewport_start)
        .take(viewport_h)
        .map(|(row, line)| composer_line(app, row, line, cursor_row, cursor_col))
        .collect();

    f.render_widget(Paragraph::new(visible_lines), inner);

    if !app.busy {
        let prefix_w = 2u16;
        let cursor_x = inner.x + prefix_w + cursor_col as u16;
        let cursor_y = inner.y + (cursor_row.saturating_sub(viewport_start)) as u16;
        if cursor_x < inner.x + inner.width && cursor_y < inner.y + inner.height {
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
        Style::default().fg(Color::DarkGray)
    } else if row == 0 {
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let text_style = if app.busy {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default().fg(Color::White)
    };

    let mut spans = vec![Span::styled(prefix.to_string(), prefix_style)];

    if app.input.is_empty() && row == 0 {
        spans.push(Span::styled(
            "Type a message or paste code…".to_string(),
            Style::default()
                .fg(Color::DarkGray)
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

// ── Slash-command popup ───────────────────────────────────────────────────────

fn draw_slash_popup(f: &mut Frame, app: &mut App, area: Rect, items: &[SlashPopupEntry]) {
    if area.height < POPUP_BORDER_H || items.is_empty() {
        return;
    }

    let block = Block::default()
        .title(" Commands ")
        .borders(Borders::ALL)
        .border_style(panel_border_style(app, PanelFocus::SlashPopup));
    let inner = block.inner(area);
    f.render_widget(block, area);
    let item_rects = (0..items.len())
        .map(|idx| Rect {
            x: inner.x,
            y: inner.y.saturating_add(idx as u16),
            width: inner.width,
            height: 1,
        })
        .collect::<Vec<_>>();
    app.register_slash_popup(area, item_rects);

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

fn panel_border_style(app: &App, panel: PanelFocus) -> Style {
    if app.is_panel_focused(panel) {
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::DarkGray)
    }
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
        Style::default().fg(phase_color).add_modifier(Modifier::DIM),
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

fn spinner_char(tick: usize) -> &'static str {
    const SPINNERS: [&str; 10] = ["|", "/", "-", "\\", "|", "/", "-", "\\", "|", "/"];
    SPINNERS[tick % SPINNERS.len()]
}

fn push_chip(spans: &mut Vec<Span<'static>>, label: &str, value: &str, color: Color) {
    spans.push(Span::styled(
        "[",
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::DIM),
    ));
    spans.push(Span::styled(
        label.to_string(),
        Style::default().fg(Color::Gray).add_modifier(Modifier::DIM),
    ));
    spans.push(Span::styled(
        ":",
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::DIM),
    ));
    spans.push(Span::styled(
        value.to_string(),
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    ));
    spans.push(Span::styled(
        "]",
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::DIM),
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
    let color = if enabled {
        Color::Green
    } else {
        Color::DarkGray
    };
    let value = if enabled { "on" } else { "off" };
    push_kv(spans, key, value, color);
}

fn push_gap(spans: &mut Vec<Span<'static>>) {
    spans.push(Span::raw("  "));
}

fn push_sep(spans: &mut Vec<Span<'static>>) {
    spans.push(Span::styled(
        " · ",
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::DIM),
    ));
}

fn push_shortcut_hint(spans: &mut Vec<Span<'static>>, item: &ShortcutHint) {
    if spans.len() > 1 {
        push_sep(spans);
    }
    spans.push(Span::styled(
        item.chord.to_string(),
        Style::default().fg(Color::Gray),
    ));
    spans.push(Span::raw(" "));
    spans.push(Span::styled(
        item.description.to_string(),
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::DIM),
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
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::DIM),
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
                .fg(Color::Yellow)
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
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::DIM),
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
                Style::default().fg(Color::Green),
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
                Style::default().fg(Color::Cyan),
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
                "true" | "false" => Style::default().fg(Color::Magenta),
                "null" => Style::default()
                    .fg(Color::DarkGray)
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
