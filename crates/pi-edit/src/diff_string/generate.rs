//! Diff string generation with block-context insertion.

use std::collections::BTreeSet;

use pi_ast::block::{EnclosingBoundaryOptions, LineRange, enclosing_block_boundaries};

use crate::diff_string::types::{BlockContextSource, DiffOutput};

use super::types::{DEFAULT_ADDED_RUN_CONTEXT_LINES, DIFF_GAP_ROW};

// ---------------------------------------------------------------------------
// Diff prefix / row helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DiffPrefix {
	Added,
	Removed,
	Context,
}

#[derive(Debug)]
struct ParsedNumberedDiffRow {
	prefix: DiffPrefix,
	line_number: u32,
}

fn format_numbered_diff_line(prefix: char, line_number: u32, content: &str) -> String {
	format!("{prefix}{line_number}|{content}")
}

fn parse_numbered_diff_row(row: &str) -> Option<ParsedNumberedDiffRow> {
	let (prefix, body) = match row.as_bytes().first().copied()? {
		b'+' => (DiffPrefix::Added, &row[1..]),
		b'-' => (DiffPrefix::Removed, &row[1..]),
		b' ' => (DiffPrefix::Context, &row[1..]),
		_ => return None,
	};
	let separator = body.find('|')?;
	let line_number = body[..separator].parse().ok()?;
	Some(ParsedNumberedDiffRow { prefix, line_number })
}

fn is_diff_change_row(row: Option<&String>) -> bool {
	row.is_some_and(|value| value.starts_with('+') || value.starts_with('-'))
}

fn parse_source_row_line_number(row: &str) -> Option<u32> {
	let parsed = parse_numbered_diff_row(row)?;
	(parsed.prefix != DiffPrefix::Added).then_some(parsed.line_number)
}

fn normalize_diff_gap_rows(rows: &mut Vec<String>) {
	let mut kept = Vec::with_capacity(rows.len());
	for (index, row) in rows.iter().enumerate() {
		if row != DIFF_GAP_ROW {
			kept.push(row.clone());
			continue;
		}
		if kept.is_empty() || kept.last().is_some_and(String::is_empty) {
			continue;
		}
		let before = kept
			.iter()
			.rev()
			.find_map(|candidate| parse_source_row_line_number(candidate));
		let after = rows[index + 1..]
			.iter()
			.filter(|candidate| !candidate.is_empty())
			.find_map(|candidate| parse_source_row_line_number(candidate));
		if matches!((before, after), (Some(left), Some(right)) if right > left + 1) {
			kept.push(String::new());
		}
	}
	*rows = kept;
}

fn adjusted_context_insert_index(rows: &[String], index: usize) -> usize {
	let mut start = index;
	while start > 0 && is_diff_change_row(rows.get(start - 1)) {
		start -= 1;
	}
	let mut end = index;
	while end < rows.len() && is_diff_change_row(rows.get(end)) {
		end += 1;
	}
	if index > start && index < end {
		end
	} else {
		index
	}
}

fn insert_bracket_context_rows(
	rows: &mut Vec<String>,
	context_lines: Vec<(u32, String)>,
	seen_rows: &mut BTreeSet<String>,
) {
	for (line_number, text) in context_lines {
		let row = format_numbered_diff_line(' ', line_number, &text);
		if seen_rows.contains(&row) {
			continue;
		}

		let mut insert_index = rows.len();
		let mut previous_source_line = None;
		let mut next_source_line = None;
		for (index, candidate) in rows.iter().enumerate() {
			let Some(parsed) = parse_numbered_diff_row(candidate) else {
				continue;
			};
			if parsed.prefix == DiffPrefix::Added {
				continue;
			}
			if parsed.line_number < line_number {
				previous_source_line = Some(parsed.line_number);
				continue;
			}
			next_source_line = Some(parsed.line_number);
			insert_index = index;
			break;
		}

		let mut chunk = Vec::with_capacity(3);
		if previous_source_line.is_some_and(|previous| line_number > previous + 1) {
			chunk.push(String::new());
		}
		chunk.push(row.clone());
		if next_source_line.is_some_and(|next| next > line_number + 1) {
			chunk.push(String::new());
		}

		let insert_index = adjusted_context_insert_index(rows, insert_index);
		rows.splice(insert_index..insert_index, chunk);
		seen_rows.insert(row);
	}
}

fn add_matching_bracket_context_rows(
	rows: &mut Vec<String>,
	old_lines: &[&str],
	new_lines: &[&str],
	source: &BlockContextSource<'_>,
) {
	let mut old_visible = Vec::new();
	let mut new_visible = Vec::new();
	let mut seen_rows = rows.iter().cloned().collect::<BTreeSet<_>>();
	let mut changes: Vec<(i64, i64)> = Vec::new();
	let mut offset = 0_i64;

	for row in rows.iter() {
		let Some(parsed) = parse_numbered_diff_row(row) else {
			continue;
		};
		match parsed.prefix {
			DiffPrefix::Removed => {
				old_visible.push(parsed.line_number);
				changes.push((i64::from(parsed.line_number) + offset, -1));
				offset -= 1;
			},
			DiffPrefix::Added => {
				new_visible.push(parsed.line_number);
				changes.push((i64::from(parsed.line_number), 1));
				offset += 1;
			},
			DiffPrefix::Context => {
				old_visible.push(parsed.line_number);
				let shifted = i64::from(parsed.line_number) + offset;
				if let Ok(line) = u32::try_from(shifted) {
					new_visible.push(line);
				}
			},
		}
	}

	let to_old_line_number = |new_line_number: u32| {
		let new_line_number = i64::from(new_line_number);
		let shift: i64 = changes
			.iter()
			.filter(|(new_position, _)| *new_position <= new_line_number)
			.map(|(_, delta)| delta)
			.sum();
		u32::try_from(new_line_number - shift).ok()
	};

	let mut context_rows = find_block_context_lines(old_lines, &old_visible, source)
		.into_iter()
		.collect::<std::collections::BTreeMap<_, _>>();
	for (line_number, text) in find_block_context_lines(new_lines, &new_visible, source) {
		if let Some(old_line_number) = to_old_line_number(line_number) {
			context_rows.entry(old_line_number).or_insert(text);
		}
	}
	insert_bracket_context_rows(rows, context_rows.into_iter().collect(), &mut seen_rows);
	normalize_diff_gap_rows(rows);
}

// ---------------------------------------------------------------------------
// Public generation functions
// ---------------------------------------------------------------------------

/// Generate a numbered diff with nearby and enclosing-block context.
pub fn generate_diff_string(
	old: &str,
	new: &str,
	context_lines: Option<usize>,
	source: &BlockContextSource<'_>,
) -> DiffOutput {
	let parts = pi_diff::line_changes_str(old, new);
	let context_lines = context_lines.unwrap_or(2);
	let mut output = Vec::new();
	let mut old_line_number = 1_u32;
	let mut new_line_number = 1_u32;
	let mut last_was_change = false;
	let mut first_changed_line = None;

	for (index, part) in parts.iter().enumerate() {
		let mut raw = part.value.split('\n').collect::<Vec<_>>();
		if raw.last() == Some(&"") {
			raw.pop();
		}

		if part.added || part.removed {
			first_changed_line.get_or_insert(new_line_number);
			for line in raw {
				if part.added {
					output.push(format_numbered_diff_line('+', new_line_number, line));
					new_line_number += 1;
				} else {
					output.push(format_numbered_diff_line('-', old_line_number, line));
					old_line_number += 1;
				}
			}
			last_was_change = true;
		} else {
			let next_part_is_change = parts
				.get(index + 1)
				.is_some_and(|next| next.added || next.removed);
			if last_was_change || next_part_is_change {
				let mut leading_skip = 0;
				let mut middle_skip = 0;
				let mut trailing_skip = 0;
				let lines_to_show;

				if last_was_change && next_part_is_change {
					if raw.len() > context_lines * 2 {
						middle_skip = raw.len() - context_lines * 2;
						lines_to_show = raw[..context_lines]
							.iter()
							.chain(&raw[raw.len() - context_lines..])
							.copied()
							.collect::<Vec<_>>();
					} else {
						lines_to_show = raw.clone();
					}
				} else if next_part_is_change {
					leading_skip = raw.len().saturating_sub(context_lines);
					lines_to_show = raw[leading_skip..].to_vec();
				} else {
					trailing_skip = raw.len().saturating_sub(context_lines);
					lines_to_show = raw[..raw.len().min(context_lines)].to_vec();
				}

				old_line_number += u32::try_from(leading_skip).unwrap_or(u32::MAX);
				new_line_number += u32::try_from(leading_skip).unwrap_or(u32::MAX);
				let first_chunk_length = if middle_skip > 0 {
					context_lines
				} else {
					lines_to_show.len()
				};
				for line in &lines_to_show[..first_chunk_length] {
					output.push(format_numbered_diff_line(' ', old_line_number, line));
					old_line_number += 1;
					new_line_number += 1;
				}
				if middle_skip > 0 {
					old_line_number += u32::try_from(middle_skip).unwrap_or(u32::MAX);
					new_line_number += u32::try_from(middle_skip).unwrap_or(u32::MAX);
					for line in &lines_to_show[first_chunk_length..] {
						output.push(format_numbered_diff_line(' ', old_line_number, line));
						old_line_number += 1;
						new_line_number += 1;
					}
				}
				old_line_number += u32::try_from(trailing_skip).unwrap_or(u32::MAX);
				new_line_number += u32::try_from(trailing_skip).unwrap_or(u32::MAX);
			} else {
				let skipped = u32::try_from(raw.len()).unwrap_or(u32::MAX);
				old_line_number += skipped;
				new_line_number += skipped;
			}
			last_was_change = false;
		}
	}

	let old_lines = old.split('\n').collect::<Vec<_>>();
	let new_lines = new.split('\n').collect::<Vec<_>>();
	add_matching_bracket_context_rows(&mut output, &old_lines, &new_lines, source);
	DiffOutput { diff: output.join("\n"), first_changed_line }
}

/// Generate numbered unified hunks without file headers.
pub fn generate_unified_diff_string(
	old: &str,
	new: &str,
	context_lines: Option<usize>,
	source: &BlockContextSource<'_>,
) -> DiffOutput {
	let old_utf16 = old.encode_utf16().collect::<Vec<_>>();
	let new_utf16 = new.encode_utf16().collect::<Vec<_>>();
	let context_lines = context_lines.unwrap_or(3);
	let hunks = pi_diff::structured_patch_hunks_u16(
		&old_utf16,
		&new_utf16,
		Some(u32::try_from(context_lines).unwrap_or(u32::MAX)),
	);
	let mut output = Vec::new();
	let mut first_changed_line = None;
	for hunk in hunks {
		output.push(format!(
			"@@ -{},{} +{},{} @@",
			hunk.old_start, hunk.old_lines, hunk.new_start, hunk.new_lines
		));
		let mut old_line = hunk.old_start;
		let mut new_line = hunk.new_start;
		for encoded in hunk.lines {
			let line = String::from_utf16(&encoded).expect("hunk text originates from valid UTF-8");
			if let Some(content) = line.strip_prefix('-') {
				first_changed_line.get_or_insert(new_line);
				output.push(format_numbered_diff_line('-', old_line, content));
				old_line += 1;
			} else if let Some(content) = line.strip_prefix('+') {
				first_changed_line.get_or_insert(new_line);
				output.push(format_numbered_diff_line('+', new_line, content));
				new_line += 1;
			} else if let Some(content) = line.strip_prefix(' ') {
				output.push(format_numbered_diff_line(' ', old_line, content));
				old_line += 1;
				new_line += 1;
			} else {
				output.push(line);
			}
		}
	}
	let old_lines = old.split('\n').collect::<Vec<_>>();
	let new_lines = new.split('\n').collect::<Vec<_>>();
	add_matching_bracket_context_rows(&mut output, &old_lines, &new_lines, source);
	DiffOutput { diff: output.join("\n"), first_changed_line }
}

// ---------------------------------------------------------------------------
// Block context helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct LineSpan {
	start_line: u32,
	end_line: u32,
}

fn visible_set_to_spans(visible: &BTreeSet<u32>) -> Vec<LineSpan> {
	let mut spans: Vec<LineSpan> = Vec::new();
	for &line in visible {
		if let Some(previous) = spans.last_mut()
			&& line <= previous.end_line.saturating_add(1)
		{
			previous.end_line = line;
		} else {
			spans.push(LineSpan { start_line: line, end_line: line });
		}
	}
	spans
}

fn native_block_context(
	full_lines: &[&str],
	visible: &BTreeSet<u32>,
	source: &BlockContextSource<'_>,
) -> Option<Vec<(u32, String)>> {
	if source.path.is_none() && source.lang.is_none() {
		return None;
	}
	let ranges = visible_set_to_spans(visible);
	if ranges.is_empty() {
		return Some(Vec::new());
	}
	let options = EnclosingBoundaryOptions {
		code: full_lines.join("\n"),
		lang: source.lang.map(str::to_owned),
		path: source.path.map(str::to_owned),
		ranges: ranges
			.into_iter()
			.map(|range| LineRange { start_line: range.start_line, end_line: range.end_line })
			.collect(),
	};
	let boundaries = enclosing_block_boundaries(options).ok()??;
	Some(
		boundaries
			.into_iter()
			.filter(|line_number| !visible.contains(line_number))
			.map(|line_number| {
				let text = usize::try_from(line_number - 1)
					.ok()
					.and_then(|index| full_lines.get(index))
					.copied()
					.unwrap_or_default()
					.to_owned();
				(line_number, text)
			})
			.collect(),
	)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScannerMode {
	Code,
	Single,
	Double,
	Template,
	BlockComment,
}

#[derive(Debug)]
struct StackEntry {
	opener: char,
	line_number: u32,
	text: String,
	visible: bool,
}

fn is_hash_comment_start(line: &str, byte_index: usize) -> bool {
	line.as_bytes().get(byte_index) == Some(&b'#')
		&& line.as_bytes()[..byte_index]
			.iter()
			.all(|byte| matches!(byte, b' ' | b'\t'))
}

fn lexical_bracket_context(full_lines: &[&str], visible: &BTreeSet<u32>) -> Vec<(u32, String)> {
	let mut context = std::collections::BTreeMap::new();
	let mut stack: Vec<StackEntry> = Vec::new();
	let mut mode = ScannerMode::Code;
	let mut escaped = false;

	for (line_index, &line) in full_lines.iter().enumerate() {
		let line_number = u32::try_from(line_index + 1).unwrap_or(u32::MAX);
		let line_visible = visible.contains(&line_number);
		let mut chars = line.char_indices().peekable();
		while let Some((byte_index, character)) = chars.next() {
			let next = chars.peek().map(|(_, value)| *value);
			if mode == ScannerMode::BlockComment {
				if character == '*' && next == Some('/') {
					mode = ScannerMode::Code;
					chars.next();
				}
				continue;
			}
			if matches!(mode, ScannerMode::Single | ScannerMode::Double | ScannerMode::Template) {
				if escaped {
					escaped = false;
					continue;
				}
				if character == '\\' {
					escaped = true;
					continue;
				}
				if (mode == ScannerMode::Single && character == '\'')
					|| (mode == ScannerMode::Double && character == '"')
					|| (mode == ScannerMode::Template && character == '`')
				{
					mode = ScannerMode::Code;
				}
				continue;
			}
			if character == '/' && next == Some('/') {
				break;
			}
			if character == '/' && next == Some('*') {
				mode = ScannerMode::BlockComment;
				chars.next();
				continue;
			}
			if character == '#' && is_hash_comment_start(line, byte_index) {
				break;
			}
			if character == '\'' {
				mode = ScannerMode::Single;
				escaped = false;
				continue;
			}
			if character == '"' {
				mode = ScannerMode::Double;
				escaped = false;
				continue;
			}
			if character == '`' {
				mode = ScannerMode::Template;
				escaped = false;
				continue;
			}
			if matches!(character, '(' | '[' | '{') {
				stack.push(StackEntry {
					opener: character,
					line_number,
					text: line.to_owned(),
					visible: line_visible,
				});
				continue;
			}
			let opener = match character {
				')' => Some('('),
				']' => Some('['),
				'}' => Some('{'),
				_ => None,
			};
			if let Some(opener) = opener
				&& let Some(match_index) = stack.iter().rposition(|entry| entry.opener == opener)
			{
				let matched = stack.remove(match_index);
				stack.truncate(match_index);
				if line_visible && !matched.visible {
					context.insert(matched.line_number, matched.text);
				}
				if matched.visible && !line_visible {
					context.insert(line_number, line.to_owned());
				}
			}
		}
		if matches!(mode, ScannerMode::Single | ScannerMode::Double) {
			mode = ScannerMode::Code;
			escaped = false;
		}
	}
	for line_number in visible {
		context.remove(line_number);
	}
	context.into_iter().collect()
}

/// Resolve off-window block-boundary lines for a visible window.
pub fn find_block_context_lines(
	full_lines: &[&str],
	visible: &[u32],
	source: &BlockContextSource<'_>,
) -> Vec<(u32, String)> {
	let visible = visible.iter().copied().collect::<BTreeSet<_>>();
	if visible.is_empty() || (!full_lines.is_empty() && visible.len() >= full_lines.len()) {
		return Vec::new();
	}
	native_block_context(full_lines, &visible, source)
		.unwrap_or_else(|| lexical_bracket_context(full_lines, &visible))
}
