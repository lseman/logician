use pulldown_cmark::{CodeBlockKind, Event as MdEvent, Options, Parser, Tag, TagEnd};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

// ── think-tag pre-processing ──────────────────────────────────────────────────

#[derive(Debug)]
struct Segment {
    text: String,
    in_think: bool,
}

fn split_think(text: &str) -> Vec<Segment> {
    let mut segs: Vec<Segment> = Vec::new();
    let mut depth = 0usize;
    let mut last = 0usize;
    let bytes = text.as_bytes();
    let len = text.len();
    let mut i = 0;

    while i < len {
        // fast-scan to next '<'
        if bytes[i] != b'<' {
            i += 1;
            continue;
        }

        // try </think>
        if text[i..].starts_with("</think>") || text[i..].starts_with("</Think>") {
            if i > last {
                segs.push(Segment {
                    text: text[last..i].to_string(),
                    in_think: depth > 0,
                });
            }
            depth = depth.saturating_sub(1);
            last = i + 8;
            i = last;
            continue;
        }
        // try <think>
        if text[i..].starts_with("<think>") || text[i..].starts_with("<Think>") {
            if i > last {
                segs.push(Segment {
                    text: text[last..i].to_string(),
                    in_think: depth > 0,
                });
            }
            depth += 1;
            last = i + 7;
            i = last;
            continue;
        }
        i += 1;
    }
    if last < len {
        segs.push(Segment {
            text: text[last..].to_string(),
            in_think: depth > 0,
        });
    }
    segs
}

// ── Streaming renderer (plain text with think coloring) ───────────────────────

pub fn render_streaming(text: &str) -> Vec<Line<'static>> {
    if text.is_empty() {
        return vec![];
    }

    let mut out: Vec<Line<'static>> = Vec::new();
    let segs = split_think(text);

    for seg in segs {
        let style = if seg.in_think {
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::ITALIC)
        } else {
            Style::default().fg(Color::White)
        };

        for raw_line in seg.text.split('\n') {
            let span = Span::styled(raw_line.to_string(), style);
            out.push(Line::from(vec![span]));
        }
    }

    out
}

// ── Full markdown renderer ────────────────────────────────────────────────────

struct MdRenderer {
    out: Vec<Line<'static>>,
    // current line being built
    spans: Vec<Span<'static>>,
    // style stack
    bold: u32,
    italic: u32,
    strike: u32,
    link_href: Option<String>,
    // code block accumulator
    in_code: bool,
    code_lang: String,
    code_buf: Vec<String>,
    // list tracking
    list_stack: Vec<ListState>,
    // table accumulator
    in_table: bool,
    table_header: Vec<String>,
    table_rows: Vec<Vec<String>>,
    cur_row: Vec<String>,
    cell_buf: String,
    in_thead: bool,
    // blockquote
    bq_depth: u32,
    // heading
    in_heading: bool,
    heading_lvl: u32,
}

#[derive(Clone)]
struct ListState {
    ordered: bool,
    item_idx: usize,
}

impl MdRenderer {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            spans: Vec::new(),
            bold: 0,
            italic: 0,
            strike: 0,
            link_href: None,
            in_code: false,
            code_lang: String::new(),
            code_buf: Vec::new(),
            list_stack: Vec::new(),
            in_table: false,
            table_header: Vec::new(),
            table_rows: Vec::new(),
            cur_row: Vec::new(),
            cell_buf: String::new(),
            in_thead: false,
            bq_depth: 0,
            in_heading: false,
            heading_lvl: 1,
        }
    }

    fn current_style(&self) -> Style {
        let mut s = Style::default().fg(Color::White);
        if self.bold > 0 {
            s = s.add_modifier(Modifier::BOLD);
        }
        if self.italic > 0 {
            s = s.add_modifier(Modifier::ITALIC);
        }
        if self.strike > 0 {
            s = s.add_modifier(Modifier::CROSSED_OUT);
        }
        if self.link_href.is_some() {
            s = s.fg(Color::Blue).add_modifier(Modifier::UNDERLINED);
        }
        s
    }

    fn push_span(&mut self, text: impl Into<String>) {
        let t = text.into();
        if t.is_empty() {
            return;
        }
        let style = self.current_style();
        self.spans.push(Span::styled(t, style));
    }

    fn push_span_styled(&mut self, text: impl Into<String>, style: Style) {
        let t = text.into();
        if t.is_empty() {
            return;
        }
        self.spans.push(Span::styled(t, style));
    }

    fn flush_line(&mut self) {
        let spans = std::mem::take(&mut self.spans);
        let indent = self.indent_prefix();
        if indent.is_empty() {
            self.out.push(Line::from(spans));
        } else {
            let mut all = vec![Span::raw(indent)];
            all.extend(spans);
            self.out.push(Line::from(all));
        }
    }

    fn flush_line_no_indent(&mut self) {
        let spans = std::mem::take(&mut self.spans);
        self.out.push(Line::from(spans));
    }

    fn blank_line(&mut self) {
        self.out.push(Line::raw(""));
    }

    fn indent_prefix(&self) -> String {
        let list_depth = self.list_stack.len();
        let bq_depth = self.bq_depth as usize;
        let bq_part = "│ ".repeat(bq_depth);
        let li_part = "  ".repeat(list_depth.saturating_sub(1));
        format!("{}{}", bq_part, li_part)
    }

    fn push_text_lines(&mut self, raw: &str) {
        // Multi-line text: flush a line for each \n
        let mut parts = raw.split('\n');
        if let Some(first) = parts.next() {
            self.push_span(first);
        }
        for part in parts {
            self.flush_line();
            if !part.is_empty() {
                self.push_span(part);
            }
        }
    }

    fn render_code_block(&mut self) {
        let lang = std::mem::take(&mut self.code_lang);
        let lines: Vec<String> = std::mem::take(&mut self.code_buf);
        // Remove trailing empty lines from the last "\n" in fence content
        let content_lines: Vec<&str> = {
            let all: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
            let mut end = all.len();
            while end > 0 && all[end - 1].trim().is_empty() {
                end -= 1;
            }
            all[..end].to_vec()
        };

        if !lang.is_empty() {
            self.out.push(Line::from(vec![Span::styled(
                format!("  ╭─ {} ", lang),
                Style::default().fg(Color::DarkGray),
            )]));
        } else {
            self.out.push(Line::from(vec![Span::styled(
                "  ╭──────────────────",
                Style::default().fg(Color::DarkGray),
            )]));
        }

        for cl in &content_lines {
            self.out.push(Line::from(vec![
                Span::styled("  │ ", Style::default().fg(Color::DarkGray)),
                Span::styled(cl.to_string(), Style::default().fg(Color::Cyan)),
            ]));
        }
        self.out.push(Line::from(vec![Span::styled(
            "  ╰──────────────────",
            Style::default().fg(Color::DarkGray),
        )]));
    }

    fn render_table(&mut self) {
        let header = std::mem::take(&mut self.table_header);
        let rows = std::mem::take(&mut self.table_rows);
        if header.is_empty() && rows.is_empty() {
            return;
        }

        let cols = header
            .len()
            .max(rows.iter().map(|r| r.len()).max().unwrap_or(0));
        let normalize = |v: &[String]| -> Vec<String> {
            (0..cols)
                .map(|i| v.get(i).cloned().unwrap_or_default())
                .collect()
        };

        let header_n = normalize(&header);
        let rows_n: Vec<Vec<String>> = rows.iter().map(|r| normalize(r)).collect();
        let all_rows: Vec<&Vec<String>> = std::iter::once(&header_n).chain(rows_n.iter()).collect();

        let col_widths: Vec<usize> = (0..cols)
            .map(|c| {
                all_rows
                    .iter()
                    .map(|r| r[c].len())
                    .max()
                    .unwrap_or(3)
                    .max(3)
            })
            .collect();

        let top = format!(
            "┌{}┐",
            col_widths
                .iter()
                .map(|w| "─".repeat(w + 2))
                .collect::<Vec<_>>()
                .join("┬")
        );
        let mid = format!(
            "├{}┤",
            col_widths
                .iter()
                .map(|w| "─".repeat(w + 2))
                .collect::<Vec<_>>()
                .join("┼")
        );
        let bot = format!(
            "└{}┘",
            col_widths
                .iter()
                .map(|w| "─".repeat(w + 2))
                .collect::<Vec<_>>()
                .join("┴")
        );

        let row_str = |cells: &[String]| -> String {
            let inner: Vec<String> = cells
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let w = col_widths.get(i).copied().unwrap_or(3);
                    let s = if c.len() > w { &c[..w] } else { c };
                    format!(" {:<width$} ", s, width = w)
                })
                .collect();
            format!("│{}│", inner.join("│"))
        };

        self.out.push(Line::from(Span::styled(
            top,
            Style::default().fg(Color::DarkGray),
        )));
        self.out.push(Line::from(vec![Span::styled(
            row_str(&header_n),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )]));
        self.out.push(Line::from(Span::styled(
            mid,
            Style::default().fg(Color::DarkGray),
        )));
        for row in &rows_n {
            self.out.push(Line::from(Span::styled(
                row_str(row),
                Style::default().fg(Color::White),
            )));
        }
        self.out.push(Line::from(Span::styled(
            bot,
            Style::default().fg(Color::DarkGray),
        )));
    }

    fn process(&mut self, text: &str) {
        let opts = Options::ENABLE_TABLES
            | Options::ENABLE_STRIKETHROUGH
            | Options::ENABLE_TASKLISTS
            | Options::ENABLE_SMART_PUNCTUATION;
        let parser = Parser::new_ext(text, opts);

        for event in parser {
            if self.in_code {
                match &event {
                    MdEvent::Text(t) => {
                        self.code_buf.push(t.to_string());
                    }
                    MdEvent::End(TagEnd::CodeBlock) => {
                        self.in_code = false;
                        self.render_code_block();
                    }
                    _ => {}
                }
                continue;
            }
            if self.in_table {
                match &event {
                    MdEvent::Start(Tag::TableHead) => {
                        self.in_thead = true;
                    }
                    MdEvent::End(TagEnd::TableHead) => {
                        self.in_thead = false;
                    }
                    MdEvent::Start(Tag::TableRow) => {
                        self.cur_row = Vec::new();
                    }
                    MdEvent::End(TagEnd::TableRow) => {
                        if self.in_thead && self.table_header.is_empty() {
                            self.table_header = std::mem::take(&mut self.cur_row);
                        } else {
                            self.table_rows.push(std::mem::take(&mut self.cur_row));
                        }
                    }
                    MdEvent::Start(Tag::TableCell) => {
                        self.cell_buf.clear();
                    }
                    MdEvent::End(TagEnd::TableCell) => {
                        self.cur_row.push(std::mem::take(&mut self.cell_buf));
                    }
                    MdEvent::Text(t) | MdEvent::Code(t) => {
                        self.cell_buf.push_str(t);
                    }
                    MdEvent::End(TagEnd::Table) => {
                        self.in_table = false;
                        self.render_table();
                    }
                    _ => {}
                }
                continue;
            }

            match event {
                // ── Block starts ────────────────────────────────────────────
                MdEvent::Start(Tag::Heading { level, .. }) => {
                    self.in_heading = true;
                    self.heading_lvl = match level {
                        pulldown_cmark::HeadingLevel::H1 => 1,
                        pulldown_cmark::HeadingLevel::H2 => 2,
                        pulldown_cmark::HeadingLevel::H3 => 3,
                        pulldown_cmark::HeadingLevel::H4 => 4,
                        pulldown_cmark::HeadingLevel::H5 => 5,
                        pulldown_cmark::HeadingLevel::H6 => 6,
                    };
                }
                MdEvent::End(TagEnd::Heading(_)) => {
                    self.in_heading = false;
                    // flush with heading color
                    let color = match self.heading_lvl {
                        1 => Color::LightYellow,
                        2 => Color::LightGreen,
                        3 => Color::LightCyan,
                        4 => Color::LightBlue,
                        5 => Color::LightMagenta,
                        _ => Color::Gray,
                    };
                    // Restyle current spans with heading color
                    for sp in &mut self.spans {
                        sp.style = sp.style.fg(color).add_modifier(Modifier::BOLD);
                    }
                    self.flush_line_no_indent();
                    self.blank_line();
                }

                MdEvent::Start(Tag::Paragraph) => {}
                MdEvent::End(TagEnd::Paragraph) => {
                    if !self.spans.is_empty() {
                        self.flush_line();
                    }
                    self.blank_line();
                }

                MdEvent::Start(Tag::CodeBlock(kind)) => {
                    self.in_code = true;
                    self.code_lang = match kind {
                        CodeBlockKind::Fenced(lang) => lang.to_string(),
                        CodeBlockKind::Indented => String::new(),
                    };
                    self.code_buf.clear();
                }

                MdEvent::Start(Tag::BlockQuote(_)) => {
                    self.bq_depth += 1;
                }
                MdEvent::End(TagEnd::BlockQuote(_)) => {
                    self.bq_depth = self.bq_depth.saturating_sub(1);
                    self.blank_line();
                }

                MdEvent::Start(Tag::List(ordered)) => {
                    self.list_stack.push(ListState {
                        ordered: ordered.is_some(),
                        item_idx: ordered.map(|s| s as usize).unwrap_or(1),
                    });
                }
                MdEvent::End(TagEnd::List(_)) => {
                    self.list_stack.pop();
                    if self.list_stack.is_empty() {
                        self.blank_line();
                    }
                }
                MdEvent::Start(Tag::Item) => {
                    if let Some(ls) = self.list_stack.last_mut() {
                        let marker = if ls.ordered {
                            let m = format!("{}. ", ls.item_idx);
                            ls.item_idx += 1;
                            m
                        } else {
                            "• ".to_string()
                        };
                        let indent = "  ".repeat(self.list_stack.len().saturating_sub(1));
                        let bq = "│ ".repeat(self.bq_depth as usize);
                        let prefix = format!("{}{}{}", bq, indent, marker);
                        self.spans
                            .push(Span::styled(prefix, Style::default().fg(Color::Yellow)));
                    }
                }
                MdEvent::End(TagEnd::Item) => {
                    if !self.spans.is_empty() {
                        self.flush_line_no_indent();
                    }
                }

                MdEvent::Start(Tag::Table(_aligns)) => {
                    self.in_table = true;
                    self.table_header.clear();
                    self.table_rows.clear();
                }

                // ── Inline formatting ────────────────────────────────────────
                MdEvent::Start(Tag::Strong) => {
                    self.bold += 1;
                }
                MdEvent::End(TagEnd::Strong) => {
                    self.bold = self.bold.saturating_sub(1);
                }
                MdEvent::Start(Tag::Emphasis) => {
                    self.italic += 1;
                }
                MdEvent::End(TagEnd::Emphasis) => {
                    self.italic = self.italic.saturating_sub(1);
                }
                MdEvent::Start(Tag::Strikethrough) => {
                    self.strike += 1;
                }
                MdEvent::End(TagEnd::Strikethrough) => {
                    self.strike = self.strike.saturating_sub(1);
                }

                MdEvent::Start(Tag::Link { dest_url, .. }) => {
                    self.link_href = Some(dest_url.to_string());
                }
                MdEvent::End(TagEnd::Link) => {
                    if let Some(href) = self.link_href.take() {
                        self.push_span_styled(
                            format!(" ({})", href),
                            Style::default().fg(Color::Blue),
                        );
                    }
                }

                // ── Text content ─────────────────────────────────────────────
                MdEvent::Text(t) => {
                    self.push_text_lines(&t);
                }

                MdEvent::Code(t) => {
                    self.push_span_styled(t.to_string(), Style::default().fg(Color::Cyan));
                }

                MdEvent::SoftBreak => {
                    self.push_span(" ");
                }
                MdEvent::HardBreak => {
                    self.flush_line();
                }

                MdEvent::Rule => {
                    self.out.push(Line::from(Span::styled(
                        "─".repeat(72),
                        Style::default().fg(Color::DarkGray),
                    )));
                }

                MdEvent::Html(html) => {
                    // silently ignore HTML (think tags etc. not handled here)
                    let _ = html;
                }

                _ => {}
            }
        }

        // flush any remaining spans
        if !self.spans.is_empty() {
            self.flush_line();
        }
    }
}

/// Render finalized markdown to styled terminal lines.
/// Think-tag content is rendered as italic magenta before the rest.
pub fn render_markdown(text: &str) -> Vec<Line<'static>> {
    if text.is_empty() {
        return vec![];
    }

    let segs = split_think(text);
    let mut all: Vec<Line<'static>> = Vec::new();

    for seg in segs {
        if seg.text.trim().is_empty() {
            continue;
        }

        if seg.in_think {
            // think sections: italic magenta, no markdown parsing
            all.push(Line::from(vec![Span::styled(
                "  ⟨thinking⟩",
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::DIM),
            )]));
            for raw_line in seg.text.trim_matches('\n').split('\n') {
                all.push(Line::from(vec![
                    Span::styled("  ", Style::default()),
                    Span::styled(
                        raw_line.to_string(),
                        Style::default()
                            .fg(Color::Magenta)
                            .add_modifier(Modifier::ITALIC),
                    ),
                ]));
            }
            all.push(Line::from(vec![Span::styled(
                "  ⟨/thinking⟩",
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::DIM),
            )]));
        } else {
            let mut r = MdRenderer::new();
            r.process(&seg.text);
            all.extend(r.out);
        }
    }

    all
}
