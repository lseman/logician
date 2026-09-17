//! Compact diff preview for model-visible result display.

use crate::diff_string::generate::DiffPrefix;

use super::types::{CompactDiffOptions, CompactDiffPreview, PREVIEW_ELISION_MARKER, PREVIEW_GAP_ROW};

// ---------------------------------------------------------------------------
// Preview helpers
// ---------------------------------------------------------------------------

fn is_preview_separator(line: &str) -> bool {
	line == PREVIEW_ELISION_MARKER || line == PREVIEW_GAP_ROW
}

fn append_preview_line(output: &mut Vec<String>, line: &str) {
	let normalized = if line == "..." || line == PREVIEW_ELISION_MARKER || line == "+…" {
		PREVIEW_ELISION_MARKER
	} else {
		line
	};
	if is_preview_separator(normalized)
		&& (output.is_empty() || output.last().is_some_and(|last| is_preview_separator(last)))
	{
		return;
	}
	output.push(normalized.to_owned());
}

#[derive(Debug)]
struct ParsedCompactDiffLine<'a> {
	kind: DiffPrefix,
	line_number: i64,
	content: &'a str,
}

fn parse_integer_prefix(value: &str) -> Option<i64> {
	let value = value.trim_start();
	let (negative, digits) = if let Some(rest) = value.strip_prefix('-') {
		(true, rest)
	} else if let Some(rest) = value.strip_prefix('+') {
		(false, rest)
	} else {
		(false, value)
	};
	let digit_count = digits.bytes().take_while(u8::is_ascii_digit).count();
	if digit_count == 0 {
		return None;
	}
	let magnitude = digits[..digit_count].parse::<i64>().ok()?;
	if negative {
		magnitude.checked_neg()
	} else {
		Some(magnitude)
	}
}

fn parse_compact_diff_line(line: &str) -> Option<ParsedCompactDiffLine<'_>> {
	let (kind, body) = match line.as_bytes().first().copied()? {
		b'+' => (DiffPrefix::Added, &line[1..]),
		b'-' => (DiffPrefix::Removed, &line[1..]),
		b' ' => (DiffPrefix::Context, &line[1..]),
		_ => return None,
	};
	let separator = body.find('|')?;
	let line_number = parse_integer_prefix(&body[..separator])?;
	Some(ParsedCompactDiffLine { kind, line_number, content: &body[separator + 1..] })
}

fn append_added_run(output: &mut Vec<String>, run: &[String], edge_lines: usize) {
	if run.is_empty() {
		return;
	}
	let collapse_threshold = edge_lines * 2 + 1;
	if run.len() <= collapse_threshold {
		for line in run {
			append_preview_line(output, line);
		}
		return;
	}
	for line in &run[..edge_lines] {
		append_preview_line(output, line);
	}
	append_preview_line(output, PREVIEW_ELISION_MARKER);
	for line in &run[run.len() - edge_lines..] {
		append_preview_line(output, line);
	}
}

// ---------------------------------------------------------------------------
// Public function
// ---------------------------------------------------------------------------

/// Build a compact current-file preview from numbered diff rows.
pub fn build_compact_diff_preview(diff: &str, options: &CompactDiffOptions) -> CompactDiffPreview {
	let lines = if diff.is_empty() {
		Vec::new()
	} else {
		diff.split('\n').collect::<Vec<_>>()
	};
	let added_run_context = options
		.max_added_run_context
		.or(options.max_unchanged_run)
		.unwrap_or(super::types::DEFAULT_ADDED_RUN_CONTEXT_LINES)
		.max(1);
	let mut added_lines = 0_usize;
	let mut removed_lines = 0_usize;
	let mut formatted = Vec::new();
	let mut added_run = Vec::new();

	for line in lines {
		let Some(parsed) = parse_compact_diff_line(line) else {
			append_added_run(&mut formatted, &added_run, added_run_context);
			added_run.clear();
			append_preview_line(&mut formatted, line);
			continue;
		};
		match parsed.kind {
			DiffPrefix::Added => {
				added_lines += 1;
				added_run.push(format!("{}:{}", parsed.line_number, parsed.content));
			},
			DiffPrefix::Removed => {
				append_added_run(&mut formatted, &added_run, added_run_context);
				added_run.clear();
				removed_lines += 1;
			},
			DiffPrefix::Context => {
				append_added_run(&mut formatted, &added_run, added_run_context);
				added_run.clear();
				let new_line_number = parsed.line_number
					+ i64::try_from(added_lines).unwrap_or(i64::MAX)
					- i64::try_from(removed_lines).unwrap_or(i64::MAX);
				append_preview_line(&mut formatted, &format!("{new_line_number}:{}", parsed.content));
			},
		}
	}
	append_added_run(&mut formatted, &added_run, added_run_context);
	while formatted
		.last()
		.is_some_and(|line| is_preview_separator(line))
	{
		formatted.pop();
	}
	CompactDiffPreview { preview: formatted.join("\n"), added_lines, removed_lines }
}
