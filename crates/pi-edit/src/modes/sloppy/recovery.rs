//! Sloppy echo recovery: boundary alignment, inline preparation, pattern recovery, overlap reconciliation.
//!
//! Port of `packages/coding-agent/src/edit/sloppy.ts` lines 4199–4784.

use super::apply::ApplyContext;
use super::types::{Candidate, NormalizedText, Operation, OperationRewrite, ParsedPattern, PlannedEdit, SelectionPair, markers::{GAP, SELECT_CLOSE, SELECT_DIVIDER, SELECT_OPEN}};
use super::locate::{locate, numbered_preview, collect_candidates, source_start, MatchMode};
use super::parse::has_marker_lines;
use super::pattern::{parse_pattern, normalize_text};
use crate::error::EditError;

pub(crate) fn decode_literal_markers(text: String) -> String {
	text
		.replace("\0V8LITOPEN\0", SELECT_OPEN)
		.replace("\0V8LITCLOSE\0", SELECT_CLOSE)
		.replace("\0V8LITDIV\0", SELECT_DIVIDER)
}

pub(crate) fn render_rewrite(
	rewrite: &str,
	indices: &[usize],
	captures: &[String],
	operation_number: usize,
) -> Result<String, EditError> {
	if rewrite.contains(SELECT_OPEN) || rewrite.contains(SELECT_CLOSE) {
		return Err(EditError::matched(format!(
			"Operation {operation_number} has selection markers in <SM:PUT>; <SM:FIND> is current \
			 text, <SM:PUT> is final text."
		)));
	}
	let mut rendered = String::new();
	let mut marker = 0;
	let mut index = 0;
	while index < rewrite.len() {
		if rewrite[index..].starts_with(GAP) {
			let line_start = rewrite[..index].rfind('\n').map_or(0, |at| at + 1);
			let line_end = rewrite[index + GAP.len()..]
				.find('\n')
				.map_or(rewrite.len(), |at| index + GAP.len() + at);
			let line = &rewrite[line_start..line_end];
			if marker >= indices.len() {
				if line.trim() == GAP {
					return Err(EditError::matched(format!(
						"Operation {operation_number} <SM:PUT> has a whole-line {GAP} with no <SM:FIND> \
						 gap to re-emit. <SM:PUT> is final text written verbatim: type the elided lines \
						 out, or add a matching {GAP} gap to <SM:FIND>. To write a literal {GAP} line, \
						 use the write tool."
					)));
				}
				rendered.push_str(GAP);
			} else {
				let capture = captures.get(indices[marker]).map_or("", String::as_str);
				let open_ended =
					line.trim() == GAP || rewrite[index + GAP.len()..line_end].trim().is_empty();
				if capture.contains('\n') && !open_ended {
					rendered.push_str(GAP);
				} else {
					rendered.push_str(capture);
					marker += 1;
				}
			}
			index += GAP.len();
			continue;
		}
		let character = rewrite[index..].chars().next().expect("non-empty suffix");
		rendered.push(character);
		index += character.len_utf8();
	}
	Ok(decode_literal_markers(rendered))
}

pub(crate) fn align_boundary_echoes(content: &str, candidate: &Candidate, replacement: &str) -> String {
	if replacement.is_empty()
		|| candidate.start == candidate.match_start && candidate.end == candidate.match_end
	{
		return replacement.to_owned();
	}
	let prefix = &content[candidate.match_start..candidate.start];
	let suffix = &content[candidate.end..candidate.match_end];
	let normalized_replacement = normalize_text(replacement);
	let normalized_prefix = normalize_text(prefix).text;
	let normalized_suffix = normalize_text(suffix).text;
	let prefix_echo =
		normalized_prefix.len() >= 3 && normalized_replacement.text.starts_with(&normalized_prefix);
	let suffix_echo = !normalized_suffix.is_empty()
		&& normalized_replacement.text.ends_with(&normalized_suffix)
		&& (normalized_suffix.len() >= 3 || prefix_echo);
	if !prefix_echo && !suffix_echo {
		return replacement.to_owned();
	}
	let mut from = 0;
	let mut to = replacement.len();
	if prefix_echo {
		from = replacement.strip_prefix(prefix).map_or(
			replacement.len(),
			|rest| replacement.len() - rest.len(),
		);
	}
	if suffix_echo {
		to = replacement.strip_suffix(suffix).map_or_else(
		|| {
				source_start(
					&normalized_replacement,
					normalized_replacement.text.len() - normalized_suffix.len(),
					replacement.len(),
				)
			},
			str::len,
		);
	}
	if from > to {
		return replacement.to_owned();
	}
	let mut aligned = replacement[from..to].to_owned();
	if prefix.chars().last().is_some_and(char::is_whitespace)
		&& aligned.chars().next().is_some_and(char::is_whitespace)
	{
		aligned = aligned.trim_start_matches(char::is_whitespace).to_owned();
	}
	if suffix.chars().next().is_some_and(char::is_whitespace)
		&& aligned.chars().last().is_some_and(char::is_whitespace)
	{
		aligned = aligned.trim_end_matches(char::is_whitespace).to_owned();
	}
	aligned
}

pub(crate) fn expand_full_line_deletion(content: &str, candidate: &Candidate) -> Candidate {
	if candidate.start == candidate.end {
		return candidate.clone();
	}
	let line_start = content[..candidate.start]
		.rfind('\n')
		.map_or(0, |at| at + 1);
	let newline = content[candidate.end..]
		.find('\n')
		.map(|at| candidate.end + at);
	let line_end = newline.unwrap_or(content.len());
	if !content[line_start..candidate.start]
		.bytes()
		.all(|byte| matches!(byte, b' ' | b'\t'))
		|| !content[candidate.end..line_end]
			.bytes()
			.all(|byte| matches!(byte, b' ' | b'\t'))
	{
		return candidate.clone();
	}
	let mut result = candidate.clone();
	result.start = line_start;
	result.end = newline.map_or(line_end, |at| at + 1);
	if line_start > 0 && result.end < content.len() {
		let previous_end = line_start - 1;
		let previous_start = content[..previous_end].rfind('\n').map_or(0, |at| at + 1);
		let next_end = content[result.end..]
			.find('\n')
			.map_or(content.len(), |at| result.end + at);
		let previous_blank = content[previous_start..previous_end].trim().is_empty();
		let next_blank = content[result.end..next_end].trim().is_empty();
		let previous_text = &content[previous_start..previous_end];
		if previous_blank && next_blank {
			result.end = if next_end == content.len() {
				next_end
			} else {
				next_end + 1
			};
		} else if previous_blank
			&& content[result.end..next_end]
				.trim_start()
				.starts_with([')', ']', '}'])
		{
			result.start = previous_start;
		} else if next_blank && previous_text.trim_end().ends_with(['(', '{', '[']) {
			result.end = if next_end == content.len() {
				next_end
			} else {
				next_end + 1
			};
		}
	} else if line_start > 0 && result.end >= content.len() {
		let previous_end = line_start - 1;
		let previous_start = content[..previous_end].rfind('\n').map_or(0, |at| at + 1);
		if content[previous_start..previous_end].trim().is_empty() {
			result.start = previous_start;
		}
	}
	result
}

pub(crate) fn snap_line_insertion_offset(content: &str, offset: usize, hop_blank_lines: bool) -> usize {
	let mut line_start = content[..offset].rfind('\n').map_or(0, |at| at + 1);
	if !content[line_start..offset]
		.bytes()
		.all(|byte| matches!(byte, b' ' | b'\t'))
	{
		return offset;
	}
	while hop_blank_lines && line_start > 0 {
		let previous_end = line_start - 1;
		let previous_start = content[..previous_end].rfind('\n').map_or(0, |at| at + 1);
		if !content[previous_start..previous_end]
			.bytes()
			.all(|byte| matches!(byte, b' ' | b'\t'))
		{
			break;
		}
		line_start = previous_start;
	}
	line_start
}

pub(crate) fn frame_line_insertion(
	content: &str,
	offset: usize,
	desired: &str,
	blank_separated: bool,
) -> String {
	if desired.is_empty() {
		return String::new();
	}
	if offset > 0 && offset == content.len() && content.as_bytes()[offset - 1] != b'\n' {
		return if desired.starts_with('\n') {
			desired.to_owned()
		} else {
			format!("\n{desired}")
		};
	}
	let mut framed = if desired.ends_with('\n') {
		desired.to_owned()
	} else {
		format!("{desired}\n")
	};
	if blank_separated && content.as_bytes().get(offset) != Some(&b'\n') && !framed.ends_with("\n\n")
	{
		framed.push('\n');
	}
	framed
}

pub(crate) fn rewrite_proves_whole_span(content: &str, candidate: &Candidate, rewrite: &str) -> bool {
	let normalized_rewrite = normalize_text(rewrite).text;
	let contexts = candidate
		.selection_spans
		.windows(2)
		.filter_map(|spans| {
			let context = normalize_text(&content[spans[0].1..spans[1].0]).text;
			(!context.is_empty()).then_some(context)
		})
		.collect::<Vec<_>>();
	if contexts.is_empty() {
		return false;
	}
	let mut from = 0;
	for context in contexts {
		let Some(found) = normalized_rewrite[from..].find(&context) else {
			return false;
		};
		from += found + context.len();
	}
	true
}

pub(crate) fn rewrite_selection_spans(
	content: &str,
	candidate: &Candidate,
	replacements: &[String],
) -> String {
	let mut rewritten = content[candidate.start..candidate.end].to_owned();
	let mut indexed = candidate
		.selection_spans
		.iter()
		.copied()
		.zip(replacements)
		.collect::<Vec<_>>();
	indexed.sort_by_key(|((start, _), _)| std::cmp::Reverse(*start));
	for ((start, end), replacement) in indexed {
		rewritten.replace_range(start - candidate.start..end - candidate.start, replacement);
	}
	rewritten
}

pub(crate) fn positional_rewrite_segments(
	rewrite: &str,
	count: usize,
	pattern_has_gaps: bool,
) -> Option<Vec<String>> {
	let lines = rewrite.split('\n').collect::<Vec<_>>();
	if lines.iter().any(|line| line.trim() == GAP) {
		if pattern_has_gaps {
			return None;
		}
		let mut groups = vec![Vec::new()];
		for line in lines {
			if line.trim() == GAP {
				groups.push(Vec::new());
			} else {
				groups.last_mut().expect("one group").push(line);
			}
		}
		return (groups.len() == count)
			.then(|| groups.into_iter().map(|group| group.join("\n")).collect());
	}
	if lines.len() == count {
		return Some(lines.into_iter().map(str::to_owned).collect());
	}
	if !rewrite.contains('\n') {
		let segments = rewrite.split(GAP).map(str::to_owned).collect::<Vec<_>>();
		if segments.len() == count {
			return Some(segments);
		}
	}
	None
}

pub(crate) fn prepare_inline(
	content: &str,
	located: &Candidate,
	span: (usize, usize),
	selection: &SelectionPair,
	rewrite: &str,
	operation_number: usize,
	lenient: bool,
) -> Result<(Candidate, String, Option<String>), EditError> {
	let (mut start, mut end) = span;
	if selection.gap_only {
		if let Some(newline) = content[start..].find('\n').map(|at| start + at)
			&& newline < end
		{
			end = newline;
		}
		while start < end && matches!(content.as_bytes()[start], b' ' | b'\t') {
			start += 1;
		}
		while end > start && matches!(content.as_bytes()[end - 1], b' ' | b'\t') {
			end -= 1;
		}
	}
	if selection.line_insertion && start == end && lenient {
		start = snap_line_insertion_offset(content, start, !rewrite.starts_with('\n'));
		end = start;
	}
	let mut candidate = located.clone();
	candidate.start = start;
	candidate.end = end;
	candidate.match_start = start;
	candidate.match_end = end;
	let stripped_blank = start == end
		&& rewrite.starts_with('\n')
		&& content.as_bytes().get(start.wrapping_sub(1)) == Some(&b'\n');
	let desired = if stripped_blank {
		&rewrite[1..]
	} else {
		rewrite
	};
	let blank_separated = stripped_blank && start >= 2 && content.as_bytes()[start - 2] == b'\n';
	let framed = if selection.line_insertion {
		frame_line_insertion(content, start, desired, blank_separated)
	} else {
		desired.to_owned()
	};
	let replacement =
		render_rewrite(&framed, &selection.capture_indices, &candidate.captures, operation_number)?;
	if replacement.is_empty() && start != end {
		let deleted = content[start..end].to_owned();
		candidate = expand_full_line_deletion(content, &candidate);
		return Ok((candidate, replacement, Some(deleted)));
	}
	Ok((candidate, replacement, None))
}

fn drop_selection_echoes(pattern: &str) -> Option<String> {
	let mut result = String::new();
	let mut run_start = 0;
	let mut changed = false;
	let mut index = 0;
	while index < pattern.len() {
		if pattern[index..].starts_with(GAP) {
			result.push_str(&pattern[run_start..index + GAP.len()]);
			index += GAP.len();
			run_start = index;
			continue;
		}
		if !pattern[index..].starts_with(SELECT_OPEN) {
			index += pattern[index..].chars().next().expect("suffix").len_utf8();
			continue;
		}
		let close =
			pattern[index + SELECT_OPEN.len()..].find(SELECT_CLOSE)? + index + SELECT_OPEN.len();
		let selected = &pattern[index + SELECT_OPEN.len()..close];
		let run = &pattern[run_start..index];
		let normalized_run = normalize_text(run);
		let old = normalize_text(selected).text;
		if !old.is_empty() && normalized_run.text.ends_with(&old) {
			let cut = normalized_run.starts[normalized_run.text.len() - old.len()];
			result.push_str(&pattern[run_start..run_start + cut]);
			changed = true;
		} else {
			result.push_str(run);
		}
		result.push_str(&pattern[index..close + SELECT_CLOSE.len()]);
		index = close + SELECT_CLOSE.len();
		run_start = index;
	}
	result.push_str(&pattern[run_start..]);
	changed.then_some(result)
}

fn echo_line_candidates(pattern: &str) -> Vec<String> {
	let lines = pattern.split('\n').collect::<Vec<_>>();
	let mut result = Vec::new();
	for index in 1..lines.len() {
		let previous = lines[index - 1];
		let line = lines[index];
		if !line.contains(SELECT_OPEN)
			|| previous.contains(SELECT_OPEN)
			|| previous.contains(GAP)
			|| previous.trim().is_empty()
		{
			continue;
		}
		let without_markers = line.replace(SELECT_OPEN, "").replace(SELECT_CLOSE, "");
		if normalize_text(&without_markers).text == normalize_text(previous).text {
			let mut candidate = lines.clone();
			candidate.remove(index - 1);
			result.push(candidate.join("\n"));
		}
	}
	result
}

fn trailing_selection_candidate(pattern: &str) -> Option<String> {
	let mut changed = false;
	let lines = pattern
		.split('\n')
		.map(|line| {
			let Some(open) = line.rfind(SELECT_OPEN) else {
				return line.to_owned();
			};
			let Some(close_relative) = line[open + SELECT_OPEN.len()..].find(SELECT_CLOSE) else {
				return line.to_owned();
			};
			let close = open + SELECT_OPEN.len() + close_relative;
			if !line[close + SELECT_CLOSE.len()..].trim().is_empty()
				|| line[..open].trim().is_empty()
				|| line[..open].contains(SELECT_OPEN)
				|| line[..open].contains(GAP)
			{
				return line.to_owned();
			}
			let old = &line[open + SELECT_OPEN.len()..close];
			let normalized = normalize_text(&line[..open]);
			let old_normalized = normalize_text(old).text;
			let Some(found) = normalized.text.find(&old_normalized) else {
				return line.to_owned();
			};
			if normalized.text[found + old_normalized.len()..].contains(&old_normalized) {
				return line.to_owned();
			}
			let raw_start = normalized.starts[found];
			let raw_end = normalized.ends[found + old_normalized.len() - 1];
			changed = true;
			format!(
				"{}{}{}{}{}{}",
				&line[..raw_start],
				SELECT_OPEN,
				&line[raw_start..raw_end],
				SELECT_CLOSE,
				&line[raw_end..open],
				&line[close + SELECT_CLOSE.len()..]
			)
		})
		.collect::<Vec<_>>();
	changed.then(|| lines.join("\n"))
}

fn recover_pattern_candidates(pattern: &str, inline: bool) -> Vec<String> {
	let mut result = Vec::new();
	let mut push = |candidate: Option<String>| {
		if let Some(candidate) = candidate
			&& candidate != pattern
			&& !result.contains(&candidate)
		{
			result.push(candidate);
		}
	};
	push(drop_selection_echoes(pattern));
	for candidate in echo_line_candidates(pattern) {
		push(Some(candidate));
	}
	let trailing = trailing_selection_candidate(pattern);
	push(trailing.clone());
	if !pattern.contains(GAP) {
		let lines = pattern
			.lines()
			.filter(|line| !line.trim().is_empty())
			.collect::<Vec<_>>();
		if lines.len() >= 2 && (inline && lines.iter().any(|line| line.contains(SELECT_OPEN))) {
			push(Some(lines.join(&format!("\n{GAP}\n"))));
		}
	}
	if let Some(trailing) = trailing
		&& !trailing.contains(GAP)
	{
		let lines = trailing
			.lines()
			.filter(|line| !line.trim().is_empty())
			.collect::<Vec<_>>();
		if lines.len() >= 2 && inline && lines.iter().any(|line| line.contains(SELECT_OPEN)) {
			push(Some(lines.join(&format!("\n{GAP}\n"))));
		}
	}
	result
}

fn punctuation_pair_variants(operation: &Operation) -> Vec<Operation> {
	let OperationRewrite::Inline { replacements } = &operation.rewrite else {
		return Vec::new();
	};
	let mut variants = Vec::new();
	let mut from = 0;
	let mut selection_index = 0;
	while let Some(open_relative) = operation.pattern_text[from..].find(SELECT_OPEN) {
		let open = from + open_relative;
		let selected_start = open + SELECT_OPEN.len();
		let Some(close_relative) = operation.pattern_text[selected_start..].find(SELECT_CLOSE) else {
			break;
		};
		let close = selected_start + close_relative;
		let old = &operation.pattern_text[selected_start..close];
		let next = replacements.get(selection_index).map_or("", String::as_str);
		let punctuation = |text: &str| {
			!text.is_empty()
				&& text.chars().all(|character| {
					!(character.is_alphanumeric()
						|| character.is_whitespace()
						|| matches!(character, '_' | '$'))
				})
		};
		if punctuation(old) && punctuation(next) {
			let mut characters = old.chars().chain(next.chars()).collect::<Vec<_>>();
			characters.sort_unstable();
			characters.dedup();
			if characters.len() == 2 {
				for length in [2, 1, 3] {
					for (left, right) in [(characters[0], characters[1]), (characters[1], characters[0])]
					{
						let old_run = left.to_string().repeat(length);
						let next_run = right.to_string().repeat(length);
						if old_run == old && next_run == next {
							continue;
						}
						let mut pattern_text = operation.pattern_text.clone();
						pattern_text.replace_range(selected_start..close, &old_run);
						let mut next_replacements = replacements.clone();
						next_replacements[selection_index] = next_run;
						let mut variant = operation.clone();
						variant.pattern_text = pattern_text;
						variant.rewrite = OperationRewrite::Inline { replacements: next_replacements };
						variants.push(variant);
					}
				}
			}
		}
		selection_index += 1;
		from = close + SELECT_CLOSE.len();
	}
	variants
}

pub(crate) fn locate_with_recovery(
	content: &str,
	operation: &Operation,
	number: usize,
	path: &str,
	exclusions: &[(usize, usize)],
	standalone: bool,
) -> Result<(Operation, ParsedPattern, Vec<Candidate>), EditError> {
	let pattern = parse_pattern(&operation.pattern_text, number)?;
	match locate(content, &pattern, operation, number, path, exclusions, standalone) {
		Ok(candidates) => {
			let mut resolved = operation.clone();
			if has_marker_lines(&operation.source_pattern_text) {
				let normalized = normalize_text(content);
				let raw = collect_candidates(content, &normalized, &pattern, MatchMode::Raw, false);
				let lenient = raw.candidates.is_empty()
					&& !collect_candidates(content, &normalized, &pattern, MatchMode::Normalized, false)
						.candidates
						.is_empty();
				resolved.whitespace_matched = lenient;
			}
			Ok((resolved, pattern, candidates))
		},
		Err(original) => {
			for candidate in recover_pattern_candidates(
				&operation.pattern_text,
				matches!(operation.rewrite, OperationRewrite::Inline { .. }),
			) {
				let Ok(pattern) = parse_pattern(&candidate, number) else {
					continue;
				};
				let mut recovered = operation.clone();
				recovered.pattern_text = candidate;
				if let Ok(candidates) =
					locate(content, &pattern, &recovered, number, path, exclusions, standalone)
				{
					return Ok((recovered, pattern, candidates));
				}
			}
			for mut variant in punctuation_pair_variants(operation) {
				let Ok(pattern) = parse_pattern(&variant.pattern_text, number) else {
					continue;
				};
				if let Ok(candidates) =
					locate(content, &pattern, &variant, number, path, exclusions, standalone)
				{
					let partial_run = candidates.iter().any(|candidate| {
						candidate.selection_spans.iter().any(|(start, end)| {
							if start >= end {
								return false;
							}
							let character = content[*start..]
								.chars()
								.next()
								.expect("selected character");
							content[..*start].ends_with(character)
								|| content[*end..].starts_with(character)
						})
					});
					if partial_run {
						continue;
					}
					variant.recovery_note = Some(format!(
						"Note: operation {number}'s punctuation selection was garbled by its own marker \
						 glyphs and was resolved against the file. Include a neighboring character next \
						 time (e.g. i{SELECT_OPEN}++){SELECT_DIVIDER}--){SELECT_CLOSE})."
					));
					return Ok((variant, pattern, candidates));
				}
			}
			Err(original)
		},
	}
}

fn normalized_index_at(normalized: &NormalizedText, raw_offset: usize) -> usize {
	normalized
		.starts
		.partition_point(|offset| *offset < raw_offset)
}

pub(crate) fn duplicate_collapse_span(
	content: &str,
	candidate: &Candidate,
	replacement: &str,
) -> Option<(usize, usize)> {
	const MIN_OVERLAP: usize = 8;
	let rewrite = normalize_text(replacement).text;
	if rewrite.len() < MIN_OVERLAP || rewrite.len() > 5000 {
		return None;
	}
	let normalized = normalize_text(content);
	let match_start = normalized_index_at(&normalized, candidate.start);
	let match_end = normalized_index_at(&normalized, candidate.end);
	for overlap in (MIN_OVERLAP..=rewrite.len().min(match_start)).rev() {
		if normalized.text[match_start - overlap..match_start] != rewrite[..overlap] {
			continue;
		}
		let mut start = normalized
			.starts
			.get(match_start - overlap)
			.copied()
			.unwrap_or(candidate.start);
		let line_start = content[..start].rfind('\n').map_or(0, |at| at + 1);
		if content[line_start..start]
			.bytes()
			.all(|byte| matches!(byte, b' ' | b'\t'))
		{
			start = line_start;
		}
		return Some((start, candidate.end));
	}
	for overlap in (MIN_OVERLAP
		..=rewrite
			.len()
			.min(normalized.text.len().saturating_sub(match_end)))
		.rev()
	{
		if normalized.text[match_end..match_end + overlap] != rewrite[rewrite.len() - overlap..] {
			continue;
		}
		let mut end = normalized
			.ends
			.get(match_end + overlap - 1)
			.copied()
			.unwrap_or(candidate.end);
		let line_end = content[end..]
			.find('\n')
			.map_or(content.len(), |at| end + at);
		if content[end..line_end]
			.bytes()
			.all(|byte| matches!(byte, b' ' | b'\t'))
		{
			end = line_end;
		}
		return Some((candidate.start, end));
	}
	None
}

pub(crate) fn resolve_references(rewrite: &str, removed: &[Option<String>]) -> Result<String, EditError> {
	let mut lines = Vec::new();
	for line in rewrite.split('\n') {
		let trimmed = line.trim();
		if let Some(number) = trimmed
			.strip_prefix('»')
			.and_then(|value| value.parse::<usize>().ok())
		{
			let Some(Some(value)) = removed.get(number.saturating_sub(1)) else {
				return Err(EditError::matched(format!(
					"»{number} must reference an earlier deletion operation."
				)));
			};
			lines.push(value.clone());
		} else {
			lines.push(line.to_owned());
		}
	}
	Ok(lines.join("\n"))
}

pub(crate) fn fnv_payload(input: &str) -> u64 {
	let mut hash = 2_166_136_261_u32;
	for unit in input.encode_utf16() {
		hash ^= u32::from(unit);
		hash = hash.wrapping_mul(16_777_619);
	}
	u64::from(hash)
}

pub(crate) fn no_op_error(
	context: &ApplyContext<'_>,
	payload: u64,
	operation: Option<usize>,
	preview: Option<(&str, usize)>,
	match_count: Option<usize>,
	hint: Option<&str>,
) -> EditError {
	let (count, escalated) = context.store.record_noop(context.canonical, payload);
	let base = if escalated {
		format!(
			"STOP: identical no-op repeated {count} times for {}. Re-read current code and send a \
			 changed payload, or move on.",
			context.path
		)
	} else if let Some(operation) = operation {
		if let Some(matches) = match_count {
			format!(
				"Operation {operation} <SM:EDIT all> matched {matches} occurrences but all make no \
				 change to {}.",
				context.path
			)
		} else {
			format!("Operation {operation} makes no change to {}.", context.path)
		}
	} else {
		format!("Edits to {} made no change.", context.path)
	};
	let grounding = preview.map_or(String::new(), |(content, offset)| {
		format!(
			"\nYour rewrite normalized to text identical to these lines. Indentation-only changes \
			 are applied verbatim; adjust the authored <SM:PUT> if another whitespace change was \
			 intended.\nCurrent file content near the closest match (no re-read needed):\n{}",
			numbered_preview(content, offset)
		)
	});
	let hint = hint.map_or(String::new(), |hint| format!("\n{hint}"));
	EditError::matched(format!("{base}{grounding}{hint}"))
}

pub(crate) fn reconcile_overlap(
	content: &str,
	left: &PlannedEdit,
	right: &PlannedEdit,
) -> Option<PlannedEdit> {
	let start = left.start.min(right.start);
	let end = left.end.max(right.end);
	let container = |outer: &PlannedEdit, inner: &PlannedEdit| {
		(outer.replacement.is_empty()
			&& !inner.replacement.is_empty()
			&& inner.start >= outer.start
			&& inner.end <= outer.end)
			.then(|| PlannedEdit {
				start:            outer.start,
				end:              outer.end,
				replacement:      if content.as_bytes().get(outer.end.wrapping_sub(1)) == Some(&b'\n')
					&& !inner.replacement.ends_with('\n')
				{
					format!("{}\n", inner.replacement)
				} else {
					inner.replacement.clone()
				},
				operation_number: inner.operation_number,
			})
	};
	if let Some(result) = container(left, right).or_else(|| container(right, left)) {
		return Some(result);
	}
	let project = |edit: &PlannedEdit| {
		format!("{}{}{}", &content[start..edit.start], edit.replacement, &content[edit.end..end])
	};
	(project(left) == project(right)).then(|| PlannedEdit {
		start,
		end,
		replacement: project(left),
		operation_number: left.operation_number,
	})
}
