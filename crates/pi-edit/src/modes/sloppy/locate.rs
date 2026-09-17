//! Sloppy candidate location: exact/fuzzy matching, boundary resolution, ambiguity detection.
//!
//! Port of `packages/coding-agent/src/edit/sloppy.ts` lines 2417–4198.

use super::types::{Candidate, CandidateResult, NormalizedText, Occurrence, Operation, OperationRewrite, ParsedPattern, PatternToken, SelectionPair, MAX_CANDIDATES, MAX_COMBINATIONS, markers::{GAP, SELECT_CLOSE, SELECT_DIVIDER, SELECT_OPEN}};
use super::parse::{has_marker_lines, missing_unmarked_lines, operation_payload};
use super::pattern::normalize_text;
use crate::error::EditError;
use crate::fuzzy::levenshtein_distance;
use std::collections::{BTreeSet, HashMap, HashSet};

fn exact_occurrences(content: &str, pattern: &str) -> Vec<Occurrence> {
	if pattern.is_empty() {
		return Vec::new();
	}
	let mut result = Vec::new();
	let mut from = 0;
	while from <= content.len().saturating_sub(pattern.len()) {
		let Some(relative) = content[from..].find(pattern) else {
			break;
		};
		let start = from + relative;
		result.push(Occurrence {
			start,
			end: start + pattern.len(),
			distance: 0,
			punctuation_edits: 0,
		});
		from = start + content[start..].chars().next().map_or(1, char::len_utf8);
	}
	result
}

fn operator_signature(text: &str) -> String {
	text
		.chars()
		.filter(|character| !(character.is_alphanumeric() || matches!(character, '_' | '$')))
		.collect()
}

fn differs_by_one_punctuation_insertion(left: &str, right: &str) -> bool {
	let left = left.chars().collect::<Vec<_>>();
	let right = right.chars().collect::<Vec<_>>();
	if left.len().abs_diff(right.len()) != 1 {
		return false;
	}
	let (shorter, longer) = if left.len() < right.len() {
		(&left, &right)
	} else {
		(&right, &left)
	};
	let mut short_index = 0;
	let mut inserted = None;
	for character in longer {
		if shorter.get(short_index) == Some(character) {
			short_index += 1;
			continue;
		}
		if inserted.is_some() {
			return false;
		}
		inserted = Some(*character);
	}
	inserted.is_some_and(|character| !matches!(character, '{' | '}' | '(' | ')' | '[' | ']'))
}

fn fuzzy_occurrences(content: &str, pattern: &str, allow_punctuation: bool) -> Vec<Occurrence> {
	// These are only work limits. JS measured UTF-16 units; byte lengths are
	// intentionally acceptable here.
	if content.is_empty() || content.len() > 50_000 {
		return Vec::new();
	}
	if pattern.len() < 6 {
		return exact_occurrences(content, pattern);
	}
	let limit = 3.min(1.max(((pattern.len() as f64) * 0.12).floor() as usize));
	let seed_length = 5.min(3.max(pattern.len().saturating_sub(limit)));
	let offsets = [0, (pattern.len() - seed_length) / 2, pattern.len() - seed_length];
	let structural = operator_signature(pattern);
	let mut starts = BTreeSet::new();
	for offset in offsets {
		if !pattern.is_char_boundary(offset) || !pattern.is_char_boundary(offset + seed_length) {
			continue;
		}
		let seed = &pattern[offset..offset + seed_length];
		let mut from = 0;
		while from <= content.len().saturating_sub(seed.len()) {
			let Some(relative) = content[from..].find(seed) else {
				break;
			};
			let found = from + relative;
			for delta in -(limit as isize)..=limit as isize {
				let start = found as isize - offset as isize + delta;
				if start >= 0
					&& (start as usize) < content.len()
					&& content.is_char_boundary(start as usize)
				{
					starts.insert(start as usize);
				}
			}
			from = found + seed.chars().next().map_or(1, char::len_utf8);
		}
	}
	// Same work-limit exception as the TypeScript implementation.
	if starts.is_empty() && content.len() <= 10_000 {
		starts.extend(content.char_indices().map(|(index, _)| index));
	}
	let mut raw = Vec::new();
	for start in starts {
		let mut best: Option<Occurrence> = None;
		for length in pattern.len().saturating_sub(limit).max(1)..=pattern.len() + limit {
			let end = start + length;
			if end > content.len() || !content.is_char_boundary(end) {
				continue;
			}
			let candidate = &content[start..end];
			let signature = operator_signature(candidate);
			let punctuation_edits = usize::from(signature != structural);
			if punctuation_edits != 0
				&& (!allow_punctuation
					|| !differs_by_one_punctuation_insertion(&structural, &signature))
			{
				continue;
			}
			let distance = levenshtein_distance(pattern, candidate);
			if distance > limit || best.is_some_and(|current| distance >= current.distance) {
				continue;
			}
			best = Some(Occurrence { start, end, distance, punctuation_edits });
		}
		if let Some(best) = best {
			raw.push(best);
		}
	}
	raw.sort_by_key(|entry| (entry.distance, entry.start));
	let mut distinct: Vec<Occurrence> = Vec::new();
	for candidate in raw {
		if distinct
			.iter()
			.any(|kept| candidate.start < kept.end && candidate.end > kept.start)
		{
			continue;
		}
		distinct.push(candidate);
	}
	distinct.sort_by_key(|entry| entry.start);
	distinct
}

pub(crate) fn source_start(normalized: &NormalizedText, offset: usize, fallback: usize) -> usize {
	normalized.starts.get(offset).copied().unwrap_or(fallback)
}
pub(crate) fn source_end(normalized: &NormalizedText, offset: usize, fallback: usize) -> usize {
	if offset == 0 {
		0
	} else {
		normalized.ends.get(offset - 1).copied().unwrap_or(fallback)
	}
}
fn preceding_literal(tokens: &[PatternToken], boundary: usize) -> Option<usize> {
	(0..boundary)
		.rev()
		.find(|index| matches!(tokens[*index], PatternToken::Literal { .. }))
}
fn following_literal(tokens: &[PatternToken], boundary: usize) -> Option<usize> {
	(boundary..tokens.len()).find(|index| matches!(tokens[*index], PatternToken::Literal { .. }))
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatchMode {
	Raw,
	Normalized,
	Fuzzy,
}

fn resolve_boundary(
	boundary: usize,
	kind: u8,
	pattern: &ParsedPattern,
	matches: &HashMap<usize, Occurrence>,
	normalized: &NormalizedText,
	mode: MatchMode,
	content: Option<&str>,
) -> usize {
	let previous_index = preceding_literal(&pattern.tokens, boundary);
	let next_index = following_literal(&pattern.tokens, boundary);
	let previous = previous_index.and_then(|index| matches.get(&index));
	let next = next_index.and_then(|index| matches.get(&index));
	let immediate_previous =
		boundary > 0 && matches!(pattern.tokens[boundary - 1], PatternToken::Literal { .. });
	let immediate_next = boundary < pattern.tokens.len()
		&& matches!(pattern.tokens[boundary], PatternToken::Literal { .. });
	let raw = mode == MatchMode::Raw;
	let start_at = |offset| {
		if raw {
			offset
		} else {
			source_start(normalized, offset, content.map_or(normalized.text.len(), str::len))
		}
	};
	let end_at = |offset| {
		if raw {
			offset
		} else {
			source_end(normalized, offset, content.map_or(normalized.text.len(), str::len))
		}
	};
	if kind == 2 {
		if let Some(next) = next {
			return start_at(next.start);
		}
		if let Some(previous) = previous {
			let offset = end_at(previous.end);
			if let Some(content) = content
				&& offset > 0
				&& content.as_bytes().get(offset - 1) != Some(&b'\n')
				&& let Some(newline) = content[offset..].find('\n')
			{
				return offset + newline + 1;
			}
			return offset;
		}
	}
	if kind == 0 {
		if immediate_next && let Some(next) = next {
			return start_at(next.start);
		}
		if let Some(previous) = previous {
			return end_at(previous.end);
		}
		if let Some(next) = next {
			return start_at(next.start);
		}
	}
	if immediate_previous && let Some(previous) = previous {
		return end_at(previous.end);
	}
	if let Some(next) = next {
		return start_at(next.start);
	}
	previous.map_or(0, |entry| end_at(entry.end))
}

pub(crate) fn collect_candidates(
	content: &str,
	normalized: &NormalizedText,
	pattern: &ParsedPattern,
	mode: MatchMode,
	allow_punctuation: bool,
) -> CandidateResult {
	let literal_indices = pattern
		.tokens
		.iter()
		.enumerate()
		.filter_map(|(index, token)| matches!(token, PatternToken::Literal { .. }).then_some(index))
		.collect::<Vec<_>>();
	let mut occurrences = HashMap::new();
	for index in &literal_indices {
		let PatternToken::Literal { text, normalized: needle } = &pattern.tokens[*index] else {
			unreachable!()
		};
		let values = match mode {
			MatchMode::Raw => exact_occurrences(content, text),
			MatchMode::Normalized => exact_occurrences(&normalized.text, needle),
			MatchMode::Fuzzy => fuzzy_occurrences(&normalized.text, needle, allow_punctuation),
		};
		if values.is_empty() {
			return CandidateResult { candidates: Vec::new(), overflow: false };
		}
		occurrences.insert(*index, values);
	}
	struct Search<'a> {
		content:           &'a str,
		normalized:        &'a NormalizedText,
		pattern:           &'a ParsedPattern,
		mode:              MatchMode,
		literal_indices:   &'a [usize],
		occurrences:       &'a HashMap<usize, Vec<Occurrence>>,
		allow_punctuation: bool,
		chosen:            HashMap<usize, Occurrence>,
		candidates:        Vec<Candidate>,
		combinations:      usize,
		overflow:          bool,
	}
	impl Search<'_> {
		fn source_start(&self, offset: usize) -> usize {
			if self.mode == MatchMode::Raw {
				offset
			} else {
				source_start(self.normalized, offset, self.content.len())
			}
		}

		fn source_end(&self, offset: usize) -> usize {
			if self.mode == MatchMode::Raw {
				offset
			} else {
				source_end(self.normalized, offset, self.content.len())
			}
		}

		fn visit(&mut self, position: usize) {
			if self.overflow {
				return;
			}
			if self.candidates.len() >= MAX_CANDIDATES || self.combinations >= MAX_COMBINATIONS {
				self.overflow = true;
				return;
			}
			if position == self.literal_indices.len() {
				if self.allow_punctuation
					&& self
						.literal_indices
						.iter()
						.map(|index| self.chosen[index].punctuation_edits)
						.sum::<usize>()
						> 1
				{
					return;
				}
				self.combinations += 1;
				let content = self.pattern.line_insertion.then_some(self.content);
				let start = resolve_boundary(
					self.pattern.selection_start,
					if self.pattern.insertion { 2 } else { 0 },
					self.pattern,
					&self.chosen,
					self.normalized,
					self.mode,
					content,
				);
				let end = resolve_boundary(
					self.pattern.selection_end,
					if self.pattern.insertion { 2 } else { 1 },
					self.pattern,
					&self.chosen,
					self.normalized,
					self.mode,
					content,
				);
				let first = self.chosen[&self.literal_indices[0]];
				let last = self.chosen[&self.literal_indices[self.literal_indices.len() - 1]];
				if start > end {
					return;
				}
				let capture_count = self
					.pattern
					.tokens
					.iter()
					.filter(|token| matches!(token, PatternToken::Gap { .. }))
					.count();
				let mut captures = vec![String::new(); capture_count];
				for (token_index, token) in self.pattern.tokens.iter().enumerate() {
					let PatternToken::Gap { capture_index, .. } = token else {
						continue;
					};
					let (Some(before_index), Some(after_index)) = (
						preceding_literal(&self.pattern.tokens, token_index),
						following_literal(&self.pattern.tokens, token_index + 1),
					) else {
						return;
					};
					let capture_start = self.source_end(self.chosen[&before_index].end);
					let capture_end = self.source_start(self.chosen[&after_index].start);
					self.content[capture_start..capture_end].clone_into(&mut captures[*capture_index]);
				}
				let selection_spans = self
					.pattern
					.selection_pairs
					.iter()
					.map(|pair| {
						let insertion = pair.start == pair.end;
						let content = pair.line_insertion.then_some(self.content);
						(
							resolve_boundary(
								pair.start,
								if insertion { 2 } else { 0 },
								self.pattern,
								&self.chosen,
								self.normalized,
								self.mode,
								content,
							),
							resolve_boundary(
								pair.end,
								if insertion { 2 } else { 1 },
								self.pattern,
								&self.chosen,
								self.normalized,
								self.mode,
								content,
							),
						)
					})
					.collect::<Vec<_>>();
				if selection_spans.iter().any(|(start, end)| start > end) {
					return;
				}
				let candidate = Candidate {
					start,
					end,
					match_start: self.source_start(first.start),
					match_end: self.source_end(last.end),
					captures,
					selection_spans,
					tuple: self
						.literal_indices
						.iter()
						.map(|index| self.chosen[index].start)
						.collect(),
				};
				if let Some(existing) = self.candidates.iter_mut().find(|existing| {
					existing.start == candidate.start
						&& existing.end == candidate.end
						&& self
							.pattern
							.selected_capture_indices
							.iter()
							.all(|index| existing.captures[*index] == candidate.captures[*index])
				}) {
					if candidate.match_end - candidate.match_start
						< existing.match_end - existing.match_start
					{
						*existing = candidate;
					}
				} else {
					self.candidates.push(candidate);
				}
				return;
			}
			let token_index = self.literal_indices[position];
			let previous_index = position.checked_sub(1).map(|at| self.literal_indices[at]);
			let previous = previous_index.and_then(|index| self.chosen.get(&index).copied());
			let gap_tokens =
				previous_index.map_or(&[][..], |index| &self.pattern.tokens[index + 1..token_index]);
			let has_gap = gap_tokens
				.iter()
				.any(|token| matches!(token, PatternToken::Gap { .. }));
			for occurrence in self.occurrences[&token_index].iter().copied() {
				if let Some(previous) = previous
					&& (if has_gap {
						occurrence.start < previous.end
					} else {
						occurrence.start != previous.end
					}) {
					continue;
				}
				if let Some(previous) = previous
					&& gap_tokens
						.iter()
						.any(|token| matches!(token, PatternToken::Gap { line_bounded: true, .. }))
					&& self.content[self.source_end(previous.end)..self.source_start(occurrence.start)]
						.contains('\n')
				{
					continue;
				}
				self.chosen.insert(token_index, occurrence);
				self.visit(position + 1);
				self.chosen.remove(&token_index);
			}
		}
	}
	let mut search = Search {
		content,
		normalized,
		pattern,
		mode,
		literal_indices: &literal_indices,
		occurrences: &occurrences,
		allow_punctuation,
		chosen: HashMap::new(),
		candidates: Vec::new(),
		combinations: 0,
		overflow: false,
	};
	search.visit(0);
	let all = search.candidates.clone();
	search.candidates.retain(|candidate| {
		!all.iter().any(|other| {
			other.match_start == candidate.match_start && other.match_end < candidate.match_end
		})
	});
	search.candidates.sort_by(|left, right| {
		(left.start, left.match_start, left.match_end, &left.tuple).cmp(&(
			right.start,
			right.match_start,
			right.match_end,
			&right.tuple,
		))
	});
	CandidateResult { candidates: search.candidates, overflow: search.overflow }
}

pub(crate) fn line_number_at(content: &str, offset: usize) -> usize {
	content[..offset.min(content.len())]
		.bytes()
		.filter(|byte| *byte == b'\n')
		.count()
		+ 1
}

pub(crate) fn numbered_preview(content: &str, offset: usize) -> String {
	let lines = content.split('\n').collect::<Vec<_>>();
	let anchor = line_number_at(content, offset.min(content.len())) - 1;
	let mut start = anchor.saturating_sub(4);
	if lines.len().saturating_sub(start) < 10 {
		start = lines.len().saturating_sub(10);
	}
	lines
		.iter()
		.skip(start)
		.take(10)
		.enumerate()
		.map(|(index, line)| format!("{}: {line}", start + index + 1))
		.collect::<Vec<_>>()
		.join("\n")
}

fn display_fragment(text: &str) -> String {
	if text.contains('\n') && text.split('\n').count() <= 8 {
		return format!("\n{text}");
	}
	let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
	let compact = if compact.chars().count() > 80 {
		format!("{}…", compact.chars().take(77).collect::<String>())
	} else {
		compact
	};
	serde_json::to_string(&compact).expect("string serializes")
}

fn first_literal(pattern: &ParsedPattern) -> Option<(&str, &str)> {
	pattern.tokens.iter().find_map(|token| match token {
		PatternToken::Literal { text, normalized } => Some((text.as_str(), normalized.as_str())),
		PatternToken::Gap { .. } => None,
	})
}

fn no_match_error(
	content: &str,
	pattern: &ParsedPattern,
	operation: &Operation,
	operation_number: usize,
	path: &str,
	standalone: bool,
) -> EditError {
	let normalized = normalize_text(content);
	let (literal, needle) =
		first_literal(pattern).unwrap_or((&operation.pattern_text, &operation.pattern_text));
	let occurrences = exact_occurrences(&normalized.text, needle);
	let closest = closest_fragment(content, needle);
	let (reason, offset) = if occurrences.is_empty() {
		(format!("Failed fragment: {} has 0 occurrences.", display_fragment(literal)), closest.1)
	} else {
		(
			format!("Failed fragment: {} could not align.", display_fragment(literal)),
			source_start(&normalized, occurrences[0].start, 0),
		)
	};
	let first = if operation.all {
		format!("Operation {operation_number} <SM:EDIT all> found 0 matches in {path}. {reason}")
	} else {
		format!("Operation {operation_number} did not match {path}. {reason}")
	};
	let missing = if has_marker_lines(&operation.source_pattern_text) {
		let lines = missing_unmarked_lines(content, &operation.source_pattern_text);
		if lines.is_empty() {
			String::new()
		} else {
			let listed = lines
				.iter()
				.take(3)
				.map(|line| display_fragment(line))
				.collect::<Vec<_>>()
				.join(", ");
			let more = if lines.len() > 3 { ", …" } else { "" };
			let verb = if lines.len() == 1 { "does" } else { "do" };
			format!(
				"\nUnmarked MATCH lines must already exist in the file; {listed}{more} {verb} not. \
				 Copy real lines from the file, and mark new lines to insert with ＋."
			)
		}
	} else {
		String::new()
	};
	let correction =
		if occurrences.is_empty() && closest.2 < 0.35 && !closest.0.is_empty() && standalone {
			let corrected = operation.pattern_text.replacen(literal, &closest.0, 1);
			format!(
				"Copy-ready corrected operation:\n{}",
				operation_payload(operation, if operation.all { "*" } else { "" }, Some(&corrected))
			)
		} else if standalone {
			"No copy-ready correction — the closest current text is only a fuzzy match. Re-read the \
			 region above and rebuild <SM:FIND> from the exact current text."
				.to_owned()
		} else {
			"No copy-ready correction — retrying this operation alone would drop sibling operations. \
			 Rebuild it inside the full payload."
				.to_owned()
		};
	EditError::matched(format!(
		"{first}{missing}\nCurrent file content near the closest match (no re-read \
		 needed):\n{}\n{correction}",
		numbered_preview(content, offset)
	))
}

fn closest_fragment(content: &str, pattern: &str) -> (String, usize, f64) {
	let mut ranked = Vec::new();
	let mut offset = 0;
	for line in content.split('\n') {
		let normalized = normalize_text(line);
		if !normalized.text.is_empty() {
			let denominator = pattern.len().max(normalized.text.len()).max(1);
			let score = levenshtein_distance(pattern, &normalized.text) as f64 / denominator as f64;
			ranked.push((line, offset, normalized, score));
			ranked.sort_by(|left, right| left.3.total_cmp(&right.3));
			ranked.truncate(3);
		}
		offset += line.len() + 1;
	}
	let Some(first) = ranked.first() else {
		return (pattern.to_owned(), 0, 1.0);
	};
	let mut best = (first.0.to_owned(), first.1, first.3);
	if pattern.len() <= 160 {
		for (line, line_offset, normalized, _) in ranked {
			let width = pattern.len().min(normalized.text.len());
			for start in normalized
				.text
				.char_indices()
				.map(|(index, _)| index)
				.chain(std::iter::once(normalized.text.len().saturating_sub(width)))
			{
				let end = start + width;
				if end > normalized.text.len() || !normalized.text.is_char_boundary(end) {
					continue;
				}
				let candidate = &normalized.text[start..end];
				let score = levenshtein_distance(pattern, candidate) as f64
					/ pattern.len().max(candidate.len()).max(1) as f64;
				if score >= best.2 {
					continue;
				}
				let raw_start = source_start(&normalized, start, 0);
				let raw_end = source_end(&normalized, end, line.len());
				best = (line[raw_start..raw_end].to_owned(), line_offset + raw_start, score);
			}
		}
	}
	best
}

fn same_rewrite_for_all(
	pattern: &ParsedPattern,
	operation: &Operation,
	candidates: &[Candidate],
) -> bool {
	match &operation.rewrite {
		OperationRewrite::Explicit { text } => {
			let gaps = text.matches(GAP).count();
			pattern
				.selected_capture_indices
				.iter()
				.take(gaps)
				.all(|index| {
					candidates
						.iter()
						.all(|candidate| candidate.captures[*index] == candidates[0].captures[*index])
				})
		},
		OperationRewrite::Inline { replacements } => {
			replacements
				.iter()
				.enumerate()
				.all(|(replacement_index, replacement)| {
					pattern
						.selection_pairs
						.get(replacement_index)
						.is_some_and(|pair| {
							pair
								.capture_indices
								.iter()
								.take(replacement.matches(GAP).count())
								.all(|index| {
									candidates.iter().all(|candidate| {
										candidate.captures[*index] == candidates[0].captures[*index]
									})
								})
						})
				})
		},
	}
}

pub(crate) fn locate(
	content: &str,
	pattern: &ParsedPattern,
	operation: &Operation,
	operation_number: usize,
	path: &str,
	exclusions: &[(usize, usize)],
	standalone_operation: bool,
) -> Result<Vec<Candidate>, EditError> {
	let normalized = normalize_text(content);
	let raw = collect_candidates(content, &normalized, pattern, MatchMode::Raw, false);
	if raw.overflow {
		return Err(EditError::matched(format!(
			"Operation {operation_number} pattern is too broad; add another distinctive {GAP} \
			 fragment."
		)));
	}
	let marker_op = has_marker_lines(&operation.source_pattern_text);
	if raw.candidates.is_empty()
		&& let Some(fallback) = &pattern.literal_fallback
	{
		let exact = exact_occurrences(&normalized.text, &fallback.normalized);
		if !exact.is_empty() && (operation.all || exact.len() == 1) {
			let candidates = exact
				.into_iter()
				.map(|occurrence| {
					let match_start = source_start(&normalized, occurrence.start, 0);
					let match_end = source_end(&normalized, occurrence.end, content.len());
					let fallback_start = occurrence.start + fallback.selection_start;
					let fallback_end = occurrence.start + fallback.selection_end;
					let start = if fallback.selection_start == fallback.normalized.len() {
						match_end
					} else {
						source_start(&normalized, fallback_start, match_end)
					};
					let end = if fallback.selection_end == fallback.normalized.len() {
						match_end
					} else if fallback.insertion {
						source_start(&normalized, fallback_end, match_end)
					} else {
						source_end(&normalized, fallback_end, match_end)
					};
					Candidate {
						start,
						end,
						match_start,
						match_end,
						captures: Vec::new(),
						selection_spans: (pattern.selection_pairs.len() == 1)
							.then_some(vec![(start, end)])
							.unwrap_or_default(),
						tuple: vec![occurrence.start],
					}
				})
				.collect::<Vec<_>>();
			return Ok(if operation.all {
				candidates
			} else {
				vec![candidates[0].clone()]
			});
		}
	}
	let mut result = if raw.candidates.is_empty() {
		collect_candidates(content, &normalized, pattern, MatchMode::Normalized, false)
	} else {
		raw
	};
	if result.candidates.is_empty() && !result.overflow && !marker_op {
		result = collect_candidates(content, &normalized, pattern, MatchMode::Fuzzy, false);
		if result.candidates.is_empty() && !result.overflow && !operation.all {
			let punctuation =
				collect_candidates(content, &normalized, pattern, MatchMode::Fuzzy, true);
			if !punctuation.overflow && punctuation.candidates.len() == 1 {
				result = punctuation;
			}
		}
	}
	if result.overflow {
		return Err(EditError::matched(format!(
			"Operation {operation_number} pattern is too broad; add another distinctive {GAP} \
			 fragment."
		)));
	}
	let mut candidates = result.candidates;
	if !exclusions.is_empty() && candidates.len() > 1 {
		let free = candidates
			.iter()
			.filter(|candidate| {
				!exclusions
					.iter()
					.any(|(start, end)| candidate.match_start < *end && *start < candidate.match_end)
			})
			.cloned()
			.collect::<Vec<_>>();
		if !free.is_empty() {
			candidates = free;
		}
	}
	if operation.all && !candidates.is_empty() {
		return Ok(candidates);
	}
	if candidates.len() == 1 {
		return Ok(candidates);
	}
	if candidates.is_empty() {
		return Err(no_match_error(
			content,
			pattern,
			operation,
			operation_number,
			path,
			standalone_operation,
		));
	}
	if candidates.len() <= 4 && !operation.desired_state {
		let outcomes = candidates
			.iter()
			.map(|candidate| match &operation.rewrite {
				OperationRewrite::Explicit { text } => {
					format!("{}{}{}", &content[..candidate.start], text, &content[candidate.end..])
				},
				OperationRewrite::Inline { replacements } => {
					let mut result = content.to_owned();
					let mut spans = candidate
						.selection_spans
						.iter()
						.copied()
						.zip(replacements)
						.collect::<Vec<_>>();
					spans.sort_by_key(|((start, _), _)| std::cmp::Reverse(*start));
					for ((start, end), replacement) in spans {
						result.replace_range(start..end, replacement);
					}
					result
				},
			})
			.map(|outcome| normalize_text(&outcome).text)
			.collect::<HashSet<_>>();
		if outcomes.len() == 1 {
			return Ok(vec![candidates[0].clone()]);
		}
	}
	let retries = candidates
		.iter()
		.take(2)
		.map(|candidate| {
			format!(
				"Near line {}:\n{}",
				line_number_at(content, candidate.start),
				operation_payload(operation, "", None)
			)
		})
		.collect::<Vec<_>>()
		.join("\n\n");
	let all_retry = if same_rewrite_for_all(pattern, operation, &candidates) {
		format!(
			"All candidates receive the same rewrite; retry every match:\n{}\n\n",
			operation_payload(operation, "*", None)
		)
	} else {
		String::new()
	};
	Err(EditError::matched(format!(
		"Operation {operation_number} is ambiguous: {} ordered tuples match.\n\n{all_retry}Add \
		 context that only the intended match has — one of these:\n\n{retries}",
		candidates.len()
	)))
}

pub(crate) fn closest_desired_block(content: &str, stated_text: &str) -> Option<String> {
	let stated = normalize_text(stated_text).text;
	if !(12..=1000).contains(&stated.len()) {
		return None;
	}
	let count = stated_text.split('\n').count();
	let lines = content.split('\n').collect::<Vec<_>>();
	if lines.len() < count {
		return None;
	}
	let mut scores = Vec::new();
	for index in 0..=lines.len() - count {
		let current = normalize_text(&lines[index..index + count].join("\n")).text;
		let max = stated.len().max(current.len()).max(1);
		let affix = stated.starts_with(&current)
			|| current.starts_with(&stated)
			|| stated.ends_with(&current)
			|| current.ends_with(&stated);
		let score = if current.is_empty()
			|| affix
			|| current.len().abs_diff(stated.len()) as f64 / max as f64 > 0.35
		{
			1.0
		} else {
			levenshtein_distance(&stated, &current) as f64 / max as f64
		};
		scores.push((index, score));
	}
	let best = scores
		.iter()
		.min_by(|left, right| left.1.total_cmp(&right.1))?;
	if best.1 == 0.0 || best.1 > 0.35 {
		return None;
	}
	if scores
		.iter()
		.any(|other| other.0.abs_diff(best.0) >= count && other.1 - best.1 < 0.1)
	{
		return None;
	}
	Some(lines[best.0..best.0 + count].join("\n"))
}

pub(crate) fn is_diff_shaped(pattern_text: &str) -> bool {
	if pattern_text.contains(SELECT_OPEN)
		|| pattern_text
			.lines()
			.any(|line| line.trim_start().starts_with(['＋', '－']))
	{
		return false;
	}
	let minus = pattern_text
		.lines()
		.any(|line| line.starts_with('-') && !line.starts_with("---"));
	minus
		&& pattern_text.lines().any(|line| {
			(line.starts_with('+') && !line.starts_with("+++"))
				|| line.trim().starts_with("@@")
				|| line.starts_with(' ')
					&& line
						.chars()
						.nth(1)
						.is_some_and(|character| !character.is_whitespace())
		})
}

pub(crate) fn diff_shaped_candidates(pattern_text: &str) -> Vec<String> {
	if !is_diff_shaped(pattern_text) {
		return Vec::new();
	}
	let lines = pattern_text.lines().collect::<Vec<_>>();
	let build = |strip_space: bool| {
		let mut out = Vec::new();
		let mut index = 0;
		while index < lines.len() {
			let line = lines[index];
			if line.starts_with("---") || line.starts_with("+++") {
				index += 1;
				continue;
			}
			if line.trim().starts_with("@@") {
				out.push(GAP.to_owned());
				index += 1;
				continue;
			}
			if line.starts_with('-') {
				let mut removed = Vec::new();
				while index < lines.len()
					&& lines[index].starts_with('-')
					&& !lines[index].starts_with("---")
				{
					removed.push(&lines[index][1..]);
					index += 1;
				}
				let mut added = Vec::new();
				while index < lines.len()
					&& lines[index].starts_with('+')
					&& !lines[index].starts_with("+++")
				{
					added.push(&lines[index][1..]);
					index += 1;
				}
				out.push(format!(
					"{SELECT_OPEN}{}{SELECT_DIVIDER}{}{SELECT_CLOSE}",
					removed.join("\n"),
					added.join("\n")
				));
				continue;
			}
			if line.starts_with('+') {
				let mut added = Vec::new();
				while index < lines.len()
					&& lines[index].starts_with('+')
					&& !lines[index].starts_with("+++")
				{
					added.push(&lines[index][1..]);
					index += 1;
				}
				if let Some(previous) = out.last_mut()
					&& previous.as_str() != GAP
					&& !previous.contains(SELECT_OPEN)
					&& !previous.trim().is_empty()
				{
					let old = previous.clone();
					*previous = format!(
						"{SELECT_OPEN}{old}{SELECT_DIVIDER}{}{SELECT_CLOSE}",
						std::iter::once(old.as_str())
							.chain(added.iter().copied())
							.collect::<Vec<_>>()
							.join("\n")
					);
				} else {
					out.extend(added.into_iter().map(|entry| format!("＋{entry}")));
				}
				continue;
			}
			out.push(if strip_space {
				line.strip_prefix(' ').unwrap_or(line).to_owned()
			} else {
				line.to_owned()
			});
			index += 1;
		}
		out.join("\n")
	};
	let spaced = build(true);
	let plain = build(false);
	if spaced == plain {
		vec![spaced]
	} else {
		vec![spaced, plain]
	}
}
