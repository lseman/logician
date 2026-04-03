use pulldown_cmark::{CodeBlockKind, Event as MdEvent, Options, Parser, Tag, TagEnd};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use serde_json::Value as JsonValue;
use std::{path::Path, sync::OnceLock};
use syntect::{
    easy::HighlightLines,
    highlighting::{Color as SyntectColor, FontStyle, Style as SyntectStyle, Theme, ThemeSet},
    parsing::{SyntaxReference, SyntaxSet},
};

const MAX_RENDER_CODE_LINES: usize = 160;
const MAX_RENDER_CODE_CHARS: usize = 12_000;
const DEFAULT_SYNTAX_THEME: &str = "base16-ocean.dark";
const SYNTAX_THEME_ENV: &str = "LOGICIAN_SYNTAX_THEME";

struct SyntaxAssets {
    syntax_set: SyntaxSet,
    theme_set: ThemeSet,
}

static SYNTAX_ASSETS: OnceLock<SyntaxAssets> = OnceLock::new();

fn syntax_assets() -> &'static SyntaxAssets {
    SYNTAX_ASSETS.get_or_init(|| SyntaxAssets {
        syntax_set: SyntaxSet::load_defaults_newlines(),
        theme_set: ThemeSet::load_defaults(),
    })
}

fn active_theme() -> &'static Theme {
    let assets = syntax_assets();

    if let Ok(requested) = std::env::var(SYNTAX_THEME_ENV) {
        if let Some(theme) = assets.theme_set.themes.get(&requested) {
            return theme;
        }
        if let Some((_, theme)) = assets
            .theme_set
            .themes
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(&requested))
        {
            return theme;
        }
    }

    assets
        .theme_set
        .themes
        .get(DEFAULT_SYNTAX_THEME)
        .or_else(|| assets.theme_set.themes.values().next())
        .expect("syntect theme set should not be empty")
}

fn syntect_color_to_ratatui(color: SyntectColor) -> Color {
    Color::Rgb(color.r, color.g, color.b)
}

fn ratatui_style_from_syntect(style: SyntectStyle) -> Style {
    let mut out = Style::default().fg(syntect_color_to_ratatui(style.foreground));

    if style.font_style.contains(FontStyle::BOLD) {
        out = out.add_modifier(Modifier::BOLD);
    }
    if style.font_style.contains(FontStyle::ITALIC) {
        out = out.add_modifier(Modifier::ITALIC);
    }
    if style.font_style.contains(FontStyle::UNDERLINE) {
        out = out.add_modifier(Modifier::UNDERLINED);
    }

    out
}

fn code_theme_background() -> Option<Color> {
    None
}

fn code_fence_token(lang: &str) -> &str {
    lang.trim()
        .split(|ch: char| ch.is_whitespace() || matches!(ch, ',' | '{' | '(' | '['))
        .next()
        .unwrap_or("")
        .trim_matches('.')
}

fn syntax_for_code_fence<'a>(lang: &str, assets: &'a SyntaxAssets) -> &'a SyntaxReference {
    let token = code_fence_token(lang);
    if token.is_empty() {
        return assets.syntax_set.find_syntax_plain_text();
    }

    assets
        .syntax_set
        .find_syntax_by_token(token)
        .or_else(|| assets.syntax_set.find_syntax_by_name(token))
        .unwrap_or_else(|| assets.syntax_set.find_syntax_plain_text())
}

fn syntax_for_path<'a>(path: &str, assets: &'a SyntaxAssets) -> &'a SyntaxReference {
    let path = Path::new(path);

    if let Some(file_name) = path.file_name().and_then(|name| name.to_str()) {
        if let Some(syntax) = assets.syntax_set.find_syntax_by_name(file_name) {
            return syntax;
        }
        if let Some(syntax) = assets.syntax_set.find_syntax_by_extension(file_name) {
            return syntax;
        }
        if let Some((_, ext)) = file_name.rsplit_once('.') {
            if let Some(syntax) = assets.syntax_set.find_syntax_by_extension(ext) {
                return syntax;
            }
        }
    }

    if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
        if let Some(syntax) = assets.syntax_set.find_syntax_by_extension(ext) {
            return syntax;
        }
    }

    assets.syntax_set.find_syntax_plain_text()
}

fn highlight_spans_for_line(
    highlighter: &mut HighlightLines<'_>,
    line: &str,
    line_bg: Option<Color>,
) -> Vec<Span<'static>> {
    if line.is_empty() {
        return match line_bg {
            Some(bg) => vec![Span::styled(" ".to_string(), Style::default().bg(bg))],
            None => vec![Span::raw("")],
        };
    }

    let assets = syntax_assets();
    let ranges = match highlighter.highlight_line(line, &assets.syntax_set) {
        Ok(ranges) => ranges,
        Err(_) => {
            let mut fallback = Style::default().fg(Color::White);
            if let Some(bg) = line_bg {
                fallback = fallback.bg(bg);
            }
            return vec![Span::styled(line.to_string(), fallback)];
        }
    };

    ranges
        .into_iter()
        .map(|(style, text)| {
            let mut style = ratatui_style_from_syntect(style);
            if let Some(bg) = line_bg {
                style = style.bg(bg);
            }
            Span::styled(text.to_string(), style)
        })
        .collect()
}

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

pub fn strip_think_blocks(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    for seg in split_think(text) {
        if !seg.in_think {
            out.push_str(&seg.text);
        }
    }
    out.trim().to_string()
}

fn is_tool_call_tag_line(line: &str) -> bool {
    matches!(
        line.trim().to_ascii_lowercase().as_str(),
        "<tool_call>" | "</tool_call>" | "<tool_call/>" | "<tool_call />"
    )
}

fn strip_tool_call_tag_lines(text: &str) -> String {
    text.lines()
        .filter(|line| !is_tool_call_tag_line(line))
        .collect::<Vec<_>>()
        .join("\n")
}

fn assistant_paragraph_is_tool_call_artifact(paragraph: &str) -> bool {
    let trimmed = paragraph.trim();
    if trimmed.is_empty() {
        return false;
    }

    if is_tool_call_tag_line(trimmed) || is_tool_call_payload_json(trimmed) {
        return true;
    }

    let without_tags = strip_tool_call_tag_lines(trimmed);
    let normalized = without_tags.trim();
    if normalized.is_empty() {
        return true;
    }

    if is_tool_call_payload_json(normalized) {
        return true;
    }

    let lower = trimmed.to_ascii_lowercase();
    lower.contains("<tool_call>") || lower.contains("</tool_call>")
}

pub fn sanitize_assistant_text(text: &str) -> String {
    if text.trim().is_empty() {
        return String::new();
    }

    let visible = strip_think_blocks(text);
    if visible.trim().is_empty() {
        return String::new();
    }

    let normalized = normalize_inline_tool_call_json(&visible);
    let mut kept: Vec<String> = Vec::new();

    for paragraph in normalized.split("\n\n") {
        if assistant_paragraph_is_tool_call_artifact(paragraph) {
            continue;
        }
        let cleaned = strip_tool_call_tag_lines(paragraph).trim().to_string();
        if !cleaned.is_empty() {
            kept.push(cleaned);
        }
    }

    kept.join("\n\n")
}

fn is_tool_call_payload_json(text: &str) -> bool {
    let trimmed = text.trim();
    let Some(end) = tool_call_json_prefix_end(trimmed) else {
        return false;
    };
    trimmed[end..].trim().is_empty()
}

fn thinking_paragraph_is_execution_chatter(paragraph: &str) -> bool {
    let trimmed = paragraph.trim();
    if trimmed.is_empty() {
        return false;
    }

    if is_tool_call_payload_json(trimmed) {
        return true;
    }

    let lower = trimmed.to_lowercase();

    let hard_markers = [
        "\"tool_call\"",
        "\"tool_calls\"",
        "\"arguments\"",
        "tool_call",
        "tool_calls",
        "arguments:",
        "read_file",
        "write_file",
        "edit_file",
        "run_ruff",
        "apply_patch",
        "run_in_terminal",
    ];
    if hard_markers.iter().any(|marker| lower.contains(marker)) {
        return true;
    }

    let meta_starts = [
        "the user asked me",
        "i should",
        "i need to",
        "now i need to",
        "let me verify",
        "the linter",
        "the file",
        "the write_file tool",
        "the read_file tool",
    ];
    let bookkeeping_terms = [
        " tool ",
        "verify",
        "linter",
        "ruff",
        "violations",
        "reading it back",
        "read it back",
        "file was written",
        "full content was written",
        "worked successfully",
        "successfully wrote",
        "created with ",
    ];
    meta_starts.iter().any(|prefix| lower.starts_with(prefix))
        && bookkeeping_terms
            .iter()
            .any(|term| lower.contains(term))
}

pub fn sanitize_thinking_text(text: &str) -> String {
    if text.trim().is_empty() {
        return String::new();
    }

    let normalized = normalize_inline_tool_call_json(text);
    let mut kept: Vec<String> = Vec::new();

    for paragraph in normalized.split("\n\n") {
        if thinking_paragraph_is_execution_chatter(paragraph) {
            continue;
        }
        let cleaned = paragraph.trim();
        if !cleaned.is_empty() {
            kept.push(cleaned.to_string());
        }
    }

    kept.join("\n\n")
}

fn maybe_pretty_json_payload(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if !(trimmed.starts_with('{') || trimmed.starts_with('[')) {
        return None;
    }
    let parsed: JsonValue = serde_json::from_str(&normalize_jsonish_quotes(trimmed)).ok()?;
    serde_json::to_string_pretty(&parsed).ok()
}

fn truncate_with_ellipsis(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let mut out = String::new();
    for (count, ch) in text.chars().enumerate() {
        if count >= max_chars.saturating_sub(1) {
            out.push('…');
            return out;
        }
        out.push(ch);
    }
    out
}

fn maybe_labeled_json_payload(line: &str) -> Option<(String, String)> {
    let trimmed = line.trim();
    let (label, payload) = trimmed.split_once(':')?;
    let label = label.trim();
    let payload = payload.trim();
    if label.is_empty() || payload.is_empty() {
        return None;
    }
    let pretty = maybe_pretty_json_payload(payload)?;
    Some((format!("{label}:"), pretty))
}

fn normalize_jsonish_quotes(text: &str) -> String {
    text.chars()
        .map(|ch| match ch {
            '“' | '”' => '"',
            '‘' | '’' => '\'',
            _ => ch,
        })
        .collect()
}

fn is_json_string_quote(ch: char) -> bool {
    matches!(ch, '"' | '“' | '”')
}

fn tool_call_json_prefix_end(text: &str) -> Option<usize> {
    let first = text.chars().next()?;
    let closer = match first {
        '{' => '}',
        '[' => ']',
        _ => return None,
    };

    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (idx, ch) in text.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                quote if is_json_string_quote(quote) => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            quote if is_json_string_quote(quote) => in_string = true,
            '{' if first == '{' => depth += 1,
            '[' if first == '[' => depth += 1,
            ch if ch == closer => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let end = idx + ch.len_utf8();
                    let prefix = normalize_jsonish_quotes(&text[..end]);
                    let parsed: JsonValue = serde_json::from_str(&prefix).ok()?;
                    let is_tool_call_payload = match parsed {
                        JsonValue::Object(ref obj) => {
                            obj.contains_key("tool_call") || obj.contains_key("tool_calls")
                        }
                        _ => false,
                    };
                    if is_tool_call_payload {
                        return Some(end);
                    }
                    return None;
                }
            }
            _ => {}
        }
    }

    None
}

fn normalize_inline_tool_call_json(text: &str) -> String {
    if text.trim().is_empty() {
        return text.to_string();
    }

    let mut out: Vec<String> = Vec::new();
    let mut in_code_fence = false;

    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            in_code_fence = !in_code_fence;
            out.push(line.to_string());
            continue;
        }

        if !in_code_fence {
            let indent_len = line.len().saturating_sub(trimmed.len());
            let indent = &line[..indent_len];
            if let Some(end) = tool_call_json_prefix_end(trimmed) {
                let suffix = trimmed[end..].trim_start();
                if !suffix.is_empty() {
                    out.push(format!("{indent}{}", &trimmed[..end]));
                    out.push(String::new());
                    out.push(format!("{indent}{suffix}"));
                    continue;
                }
            }
        }

        out.push(line.to_string());
    }

    let mut rendered = out.join("\n");
    if text.ends_with('\n') {
        rendered.push('\n');
    }
    rendered
}

fn normalize_raw_json_blocks(text: &str) -> String {
    if text.trim().is_empty() {
        return text.to_string();
    }

    let mut out: Vec<String> = Vec::new();
    let mut in_code_fence = false;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            in_code_fence = !in_code_fence;
            out.push(line.to_string());
            continue;
        }

        if !in_code_fence {
            if let Some(pretty) = maybe_pretty_json_payload(trimmed) {
                out.push("```json".to_string());
                out.extend(pretty.lines().map(|ln| ln.to_string()));
                out.push("```".to_string());
                continue;
            }

            if let Some((label, pretty)) = maybe_labeled_json_payload(line) {
                out.push(label);
                out.push("```json".to_string());
                out.extend(pretty.lines().map(|ln| ln.to_string()));
                out.push("```".to_string());
                continue;
            }
        }

        out.push(line.to_string());
    }

    let mut rendered = out.join("\n");
    if text.ends_with('\n') {
        rendered.push('\n');
    }
    rendered
}

fn split_backtick_fence(line: &str) -> Option<(&str, &str, &str)> {
    let trimmed = line.trim_start();
    let indent_len = line.len().saturating_sub(trimmed.len());
    let indent = &line[..indent_len];
    let tick_count = trimmed.chars().take_while(|&ch| ch == '`').count();
    if tick_count < 3 {
        return None;
    }
    let fence = &trimmed[..tick_count];
    let suffix = &trimmed[tick_count..];
    Some((indent, fence, suffix))
}

fn normalize_broken_code_fences(text: &str) -> String {
    if text.trim().is_empty() {
        return text.to_string();
    }

    let mut out: Vec<String> = Vec::new();
    let mut in_code_fence = false;

    for line in text.lines() {
        if let Some((indent, fence, suffix)) = split_backtick_fence(line) {
            if in_code_fence {
                if suffix.trim().is_empty() {
                    out.push(line.to_string());
                } else {
                    // Repair malformed closing fences like:
                    // ```The user asked...
                    // into:
                    // ```
                    // The user asked...
                    out.push(format!("{indent}{fence}"));
                    out.push(format!("{indent}{}", suffix.trim_start()));
                }
                in_code_fence = false;
                continue;
            }

            out.push(line.to_string());
            in_code_fence = true;
            continue;
        }

        out.push(line.to_string());
    }

    let mut rendered = out.join("\n");
    if text.ends_with('\n') {
        rendered.push('\n');
    }
    rendered
}

// ── Streaming renderer (plain text with think coloring) ───────────────────────

pub fn render_streaming(text: &str) -> Vec<Line<'static>> {
    if text.is_empty() {
        return vec![];
    }

    let mut out: Vec<Line<'static>> = Vec::new();
    let normalized = normalize_broken_code_fences(&normalize_raw_json_blocks(
        &normalize_inline_tool_call_json(text),
    ));
    let segs = split_think(&normalized);

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
        let assets = syntax_assets();
        let theme = active_theme();
        let syntax = syntax_for_code_fence(&lang, assets);
        let mut highlighter = HighlightLines::new(syntax, theme);
        let code_bg = code_theme_background();
        let gutter_style = match code_bg {
            Some(bg) => Style::default().fg(Color::DarkGray).bg(bg),
            None => Style::default().fg(Color::DarkGray),
        };
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
                gutter_style,
            )]));
        } else {
            self.out.push(Line::from(vec![Span::styled(
                "  ╭──────────────────",
                gutter_style,
            )]));
        }

        let mut visible_lines: Vec<String> = Vec::new();
        let mut total_chars = 0usize;
        let mut truncated = false;
        for (idx, cl) in content_lines.iter().enumerate() {
            if idx >= MAX_RENDER_CODE_LINES {
                truncated = true;
                break;
            }
            let line_chars = cl.chars().count();
            if total_chars + line_chars > MAX_RENDER_CODE_CHARS {
                let remaining = MAX_RENDER_CODE_CHARS.saturating_sub(total_chars);
                if remaining > 1 {
                    visible_lines.push(truncate_with_ellipsis(cl, remaining));
                }
                truncated = true;
                break;
            }
            visible_lines.push((*cl).to_string());
            total_chars += line_chars + 1;
        }
        if truncated {
            visible_lines.push("… [expand on demand]".to_string());
        }

        for cl in &visible_lines {
            let mut spans = vec![Span::styled("  │ ", gutter_style)];
            if cl == "… [expand on demand]" {
                let style = match code_bg {
                    Some(bg) => Style::default().fg(Color::Yellow).bg(bg),
                    None => Style::default().fg(Color::Yellow),
                };
                spans.push(Span::styled(cl.to_string(), style));
            } else {
                spans.extend(highlight_spans_for_line(&mut highlighter, cl, code_bg));
            }
            self.out.push(Line::from(spans));
        }
        self.out.push(Line::from(vec![Span::styled(
            "  ╰──────────────────",
            gutter_style,
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
                        // pulldown_cmark delivers the entire code block body as
                        // a single Text event with embedded '\n' characters.
                        // Split here so render_code_block sees one entry per line.
                        for line in t.split('\n') {
                            self.code_buf.push(line.to_string());
                        }
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

fn diff_content_background(raw_line: &str) -> Option<Color> {
    if raw_line.starts_with('+') {
        Some(Color::Rgb(18, 48, 28))
    } else if raw_line.starts_with('-') {
        Some(Color::Rgb(56, 24, 24))
    } else {
        code_theme_background()
    }
}

/// Render a unified diff with colored additions, deletions, and hunk headers.
pub fn render_diff(path: &str, diff_text: &str) -> Vec<Line<'static>> {
    use ratatui::style::Style;

    let mut out: Vec<Line<'static>> = Vec::new();
    let assets = syntax_assets();
    let theme = active_theme();
    let syntax = syntax_for_path(path, assets);
    let mut highlighter = HighlightLines::new(syntax, theme);

    // Header
    out.push(Line::from(vec![
        Span::styled("diff: ", Style::default().fg(Color::White)),
        Span::styled(path.to_string(), Style::default().fg(Color::Cyan)),
    ]));
    out.push(Line::from(Span::styled(
        "─".repeat(60),
        Style::default().fg(Color::DarkGray),
    )));

    if diff_text.trim().is_empty() {
        out.push(Line::from(Span::styled(
            "No textual diff available.",
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::ITALIC),
        )));
        out.push(Line::from(Span::styled(
            "─".repeat(60),
            Style::default().fg(Color::DarkGray),
        )));
        return out;
    }

    let all_lines: Vec<&str> = diff_text.lines().collect();

    // Render lines (max 80 lines to avoid overcrowding)
    const MAX_DIFF_LINES: usize = 80;
    let total_lines = all_lines.len();
    let render_limit = total_lines.min(MAX_DIFF_LINES);

    for (idx, raw_line) in all_lines.into_iter().enumerate() {
        if idx >= render_limit && idx < total_lines {
            out.push(Line::from(vec![
                Span::styled("  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "… [diff truncated in panel]",
                    Style::default().fg(Color::Yellow),
                ),
            ]));
            break;
        }

        if raw_line.starts_with("@@") {
            out.push(Line::from(Span::styled(
                raw_line.to_string(),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )));
            continue;
        }

        if raw_line.starts_with("diff --git")
            || raw_line.starts_with("index ")
            || raw_line.starts_with("--- ")
            || raw_line.starts_with("+++ ")
        {
            out.push(Line::from(Span::styled(
                raw_line.to_string(),
                Style::default().fg(Color::DarkGray),
            )));
            continue;
        }

        if raw_line.starts_with("\\ No newline at end of file") {
            out.push(Line::from(Span::styled(
                raw_line.to_string(),
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            )));
            continue;
        }

        let mut line_spans: Vec<Span<'static>> =
            vec![Span::styled("  ", Style::default().fg(Color::DarkGray))];
        let content_bg = diff_content_background(raw_line);

        if let Some(content) = raw_line.strip_prefix('+') {
            let mut style = Style::default().fg(Color::Green);
            if let Some(bg) = content_bg {
                style = style.bg(bg);
            }
            line_spans.push(Span::styled("+", style));
            line_spans.extend(highlight_spans_for_line(
                &mut highlighter,
                content,
                content_bg,
            ));
        } else if let Some(content) = raw_line.strip_prefix('-') {
            let mut style = Style::default().fg(Color::Red);
            if let Some(bg) = content_bg {
                style = style.bg(bg);
            }
            line_spans.push(Span::styled("-", style));
            line_spans.extend(highlight_spans_for_line(
                &mut highlighter,
                content,
                content_bg,
            ));
        } else {
            let mut style = Style::default().fg(Color::DarkGray);
            if let Some(bg) = content_bg {
                style = style.bg(bg);
            }
            line_spans.push(Span::styled(" ", style));
            line_spans.extend(highlight_spans_for_line(
                &mut highlighter,
                raw_line,
                content_bg,
            ));
        }

        out.push(Line::from(line_spans));
    }

    // Footer
    out.push(Line::from(Span::styled(
        "─".repeat(60),
        Style::default().fg(Color::DarkGray),
    )));

    out
}

/// Render finalized markdown to styled terminal lines.
/// Embedded think-tag content is omitted here and should be surfaced through the
/// dedicated thinking stream instead.
pub fn render_markdown(text: &str) -> Vec<Line<'static>> {
    let visible_text = strip_think_blocks(text);
    if visible_text.is_empty() {
        return vec![];
    }

    if let Some(pretty) = maybe_pretty_json_payload(&visible_text) {
        let mut r = MdRenderer::new();
        r.process(&format!("```json\n{pretty}\n```"));
        return r.out;
    }

    let mut all: Vec<Line<'static>> = Vec::new();
    let normalized = normalize_broken_code_fences(&normalize_raw_json_blocks(
        &normalize_inline_tool_call_json(&visible_text),
    ));
    let mut r = MdRenderer::new();
    r.process(&normalized);
    all.extend(r.out);

    all
}

#[cfg(test)]
mod tests {
    use super::{
        code_fence_token, normalize_broken_code_fences, normalize_inline_tool_call_json,
        sanitize_assistant_text, sanitize_thinking_text, strip_think_blocks,
    };

    #[test]
    fn repairs_trailing_prose_after_closing_fence() {
        let input = "```\nHello\n```The user asked me to summarize.\n";
        let expected = "```\nHello\n```\nThe user asked me to summarize.\n";
        assert_eq!(normalize_broken_code_fences(input), expected);
    }

    #[test]
    fn preserves_opening_fence_info_string() {
        let input = "```python\nprint('hi')\n```\n";
        assert_eq!(normalize_broken_code_fences(input), input);
    }

    #[test]
    fn splits_inline_tool_call_json_from_following_prose() {
        let input = "{\"tool_call\": {\"name\": \"read_file\", \"arguments\": {\"path\": \"/tmp/test.py\"}}}The linter ran clean.";
        let expected = "{\"tool_call\": {\"name\": \"read_file\", \"arguments\": {\"path\": \"/tmp/test.py\"}}}\n\nThe linter ran clean.";
        assert_eq!(normalize_inline_tool_call_json(input), expected);
    }

    #[test]
    fn leaves_non_tool_call_json_inline_text_unchanged() {
        let input = "{\"message\": \"hello\"}still inline";
        assert_eq!(normalize_inline_tool_call_json(input), input);
    }

    #[test]
    fn preserves_normal_planning_text_in_thinking() {
        let input = "I am narrowing the issue to the streaming renderer and status composition.";
        assert_eq!(sanitize_thinking_text(input), input);
    }

    #[test]
    fn strips_wrapped_tool_call_block_from_assistant_text() {
        let input = "Before\n\n<tool_call>\n{\"tool_call\": {\"name\": \"read_file\", \"arguments\": {\"path\": \"/tmp/test.py\"}}}\n</tool_call>\n\nAfter";
        assert_eq!(sanitize_assistant_text(input), "Before\n\nAfter");
    }

    #[test]
    fn strips_bare_tool_call_json_from_assistant_text() {
        let input = "{\"tool_call\": {\"name\": \"read_file\", \"arguments\": {\"path\": \"/tmp/test.py\"}}}";
        assert_eq!(sanitize_assistant_text(input), "");
    }

    #[test]
    fn strips_smart_quoted_multiline_tool_call_json_from_assistant_text() {
        let input = "{“tool_call”: {\n  “name”: “run_python”,\n  “arguments”: {“code”: “print(1)”}\n}}";
        assert_eq!(sanitize_assistant_text(input), "");
    }

    #[test]
    fn extracts_primary_fence_language_token() {
        assert_eq!(code_fence_token("rust,no_run"), "rust");
        assert_eq!(code_fence_token("python title=demo"), "python");
        assert_eq!(code_fence_token(""), "");
    }

    #[test]
    fn strips_think_blocks_from_visible_markdown() {
        let input = "<think>plan\nhere</think>\n\nFinal answer.";
        assert_eq!(strip_think_blocks(input), "Final answer.");
    }
}
