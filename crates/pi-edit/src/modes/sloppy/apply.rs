//! Sloppy apply: plan and execute atomic edits from parsed operations.
//!
//! Port of `packages/coding-agent/src/edit/sloppy.ts` lines 4785–4972.

use super::parse::{edit_header, operation_payload, parse_operations};
use super::recovery::{align_boundary_echoes, duplicate_collapse_span, expand_full_line_deletion, frame_line_insertion, fnv_payload, locate_with_recovery, no_op_error, positional_rewrite_segments, prepare_inline, reconcile_overlap, render_rewrite, resolve_references, rewrite_proves_whole_span, rewrite_selection_spans, snap_line_insertion_offset};
use super::types::{ATOMICITY_NOTICE, Candidate, EdgeGaps, OperationRewrite, ParsedPattern, PatternToken, PlannedEdit, SelectionPair, markers::GAP};
pub use super::pattern::normalize_text;
use super::locate::line_number_at;
use crate::{error::EditError, store::EditStore};
use std::{

	borrow::Cow,
	collections::{BTreeMap, HashSet},
	path::Path,
};

/// State shared by every operation in one file section.
pub struct ApplyContext<'a> {
	pub path:      &'a str,
	pub notes:     &'a mut Vec<String>,
	pub store:     &'a EditStore,
	pub canonical: &'a Path,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MatchMode {
	Raw,
	Normalized,
	Fuzzy,
}

/// Drop the `*** SM:PUT` ellipses that re-emit `*** SM:FIND`'s open edges. An
/// edge gap captured nothing, so re-emitting it writes nothing; a whole-line
/// edge `…` takes the newline joining it to the rest of the rewrite with it.
/// The leading edge is positional; the trailing one is only claimed by an
/// ellipsis left over after the inner captures, since `…` re-emits captures
/// in order.
pub(crate) fn strip_edge_gaps(rewrite: &str, edges: EdgeGaps, inner: usize) -> Cow<'_, str> {
	// A rewrite that is nothing but edge ellipses is context elision, not a
	// deletion; leave it for `render_rewrite` to reject.
	let keep = |stripped: String, current: &str| {
		if stripped.trim().is_empty() {
			current.to_owned()
		} else {
			stripped
		}
	};
	let mut text = Cow::Borrowed(rewrite);
	if edges.leading {
		let at = text.len() - text.trim_start().len();
		if text[at..].starts_with(GAP) {
			let after = at + GAP.len();
			let line_end = text[after..]
				.find('\n')
				.map_or(text.len(), |offset| after + offset);
			let stripped = if text[after..line_end].trim().is_empty() {
				let line_start = text[..at].rfind('\n').map_or(0, |offset| offset + 1);
				format!("{}{}", &text[..line_start], &text[(line_end + 1).min(text.len())..])
			} else {
				format!("{}{}", &text[..at], &text[after..])
			};
			text = Cow::Owned(keep(stripped, &text));
		}
	}
	if edges.trailing && text.matches(GAP).count() > inner {
		let end = text.trim_end().len();
		if text[..end].ends_with(GAP) {
			let at = end - GAP.len();
			let line_start = text[..at].rfind('\n').map_or(0, |offset| offset + 1);
			let cut = if text[line_start..at].trim().is_empty() {
				text[..line_start]
					.strip_suffix('\n')
					.map_or(line_start, |rest| rest.strip_suffix('\r').map_or(rest.len(), str::len))
			} else {
				at
			};
			let stripped = text[..cut].to_owned();
			text = Cow::Owned(keep(stripped, &text));
		}
	}
	text
}

/// Edges of `*** SM:FIND` that an inline selection borders; its replacement may
/// re-emit them.
fn selection_edges(
	pattern: &ParsedPattern,
	candidate: &Candidate,
	pair: &SelectionPair,
) -> EdgeGaps {
	if candidate.literal_gaps {
		return EdgeGaps::default();
	}
	EdgeGaps {
		leading:  pattern.edge_gaps.leading && pair.start == 0,
		trailing: pattern.edge_gaps.trailing && pair.end == pattern.tokens.len(),
	}
}

#[allow(clippy::suspicious_operation_groupings, reason = "paired index bounds are intentional")]
fn apply_operations(
	content: &str,
	input: &str,
	context: &mut ApplyContext<'_>,
) -> Result<String, EditError> {
	let payload = fnv_payload(input);
	let operations = parse_operations(input, content, context.path)?;
	let mut removed = vec![None; operations.len()];
	let mut planned = Vec::new();
	let mut recovery_notes = Vec::new();
	let mut deletion_notes = BTreeMap::new();
	let mut last_match = 0;
	let mut queue = (0..operations.len()).collect::<Vec<_>>();
	let mut deferred = HashSet::new();
	let mut cursor = 0;
	while cursor < queue.len() {
		let index = queue[cursor];
		cursor += 1;
		let number = index + 1;
		if let Some(note) = &operations[index].recovery_note {
			recovery_notes.push(note.clone());
		}
		let exclusions = if deferred.contains(&index) {
			planned
				.iter()
				.map(|edit: &PlannedEdit| (edit.start, edit.end))
				.collect::<Vec<_>>()
		} else {
			Vec::new()
		};
		let located = locate_with_recovery(
			content,
			&operations[index],
			number,
			context.path,
			&exclusions,
			operations.len() == 1,
		);
	let (operation, pattern, mut candidates) = match located {
			Ok(value) => value,
			Err(error)
				if !deferred.contains(&index) && error.to_string().contains(" is ambiguous: ") =>
			{
				deferred.insert(index);
				queue.push(index);
				continue;
			},
			Err(error) => return Err(error),
		};
		if operation.whitespace_matched {
			recovery_notes.push(format!(
				"Note: operation {number}'s *** SM:FIND differed from the file in whitespace only and \
				 was matched leniently. Inserted lines are written exactly as authored — verify their \
				 indentation."
			));
		}
		if operation.all {
			candidates.sort_by_key(|candidate| std::cmp::Reverse(candidate.start));
		}
		match &operation.rewrite {
			OperationRewrite::After { text } => {
				for candidate in &candidates {
					// Insert before the final anchor line's newline, retaining its EOF convention.
					let end = candidate.match_end;
					let offset = if end > 0 && content.as_bytes()[end - 1] == b'\n' {
						end - 1
					} else {
						content[end..]
							.find('\n')
							.map_or(content.len(), |at| end + at)
					};
					let mut replacement = String::with_capacity(text.len());
					replacement.push('\n');
					replacement.push_str(text.strip_suffix('\n').unwrap_or(text));
					planned.push(PlannedEdit {
						start: offset,
						end: offset,
						replacement,
						operation_number: number,
					});
					last_match = candidate.match_start;
				}
			},
			OperationRewrite::Inline { replacements } => {
				let replacements = replacements
					.iter()
					.map(|rewrite| resolve_references(rewrite, &removed))
					.collect::<Result<Vec<_>, _>>()?;
				if pattern.selection_pairs.len() != replacements.len() {
					return Err(EditError::matched(format!(
						"Operation {number} inline replacements do not align with its selections."
					)));
				}
				let mut changes = 0;
				for candidate in &candidates {
					let mut selections = candidate
						.selection_spans
						.iter()
						.copied()
						.enumerate()
						.collect::<Vec<_>>();
					selections.sort_by_key(|(_, (start, _))| std::cmp::Reverse(*start));
					for (selection_index, span) in selections {
						let pair = &pattern.selection_pairs[selection_index];
						let (prepared, replacement, deleted) = prepare_inline(
							content,
							candidate,
							span,
							pair,
							&replacements[selection_index],
							selection_edges(&pattern, candidate, pair),
							number,
							operation.whitespace_matched,
						)?;
						if content[prepared.start..prepared.end] == replacement {
							continue;
						}
						if candidates.len() == 1
							&& pattern.selection_pairs.len() == 1
							&& let Some(deleted) = deleted
						{
							removed[index] = Some(deleted);
						}
						planned.push(PlannedEdit {
							start: prepared.start,
							end: prepared.end,
							replacement,
							operation_number: number,
						});
						changes += 1;
					}
					last_match = candidate.match_start;
				}
				if changes == 0 {
					return Err(no_op_error(
						context,
						payload,
						Some(number),
						Some((content, candidates[0].match_start)),
						operation.all.then_some(candidates.len()),
						Some(
							"The stated text equals the current text and never changes the file. Restate \
							 the edit with the actual change; do not drop the operation.",
						),
					));
				}
			},
			OperationRewrite::Explicit { text } => {
				let resolved = resolve_references(text, &removed)?;
				let base = if resolved.trim().is_empty() {
					if pattern.insertion {
						"\n".to_owned()
					} else {
						String::new()
					}
				} else {
					resolved
				};
				if pattern.selection_ranges.len() > 1
					&& let Some(segments) = positional_rewrite_segments(
						&base,
						pattern.selection_ranges.len(),
						pattern
							.tokens
							.iter()
							.any(|token| matches!(token, PatternToken::Gap { .. })),
					) {
					let mut changes = 0;
					for candidate in &candidates {
						for ((start, end), replacement) in candidate
							.selection_spans
							.iter()
							.copied()
							.zip(&segments)
							.rev()
						{
							if content[start..end] == *replacement {
								continue;
							}
							planned.push(PlannedEdit {
								start,
								end,
								replacement: replacement.clone(),
								operation_number: number,
							});
							changes += 1;
						}
						last_match = candidate.match_start;
					}
					if changes == 0 {
						return Err(no_op_error(
							context,
							payload,
							Some(number),
							Some((content, candidates[0].match_start)),
							operation.all.then_some(candidates.len()),
							None,
						));
					}
					continue;
				}
				if pattern.selection_ranges.len() > 1
					&& !candidates
						.iter()
						.all(|candidate| rewrite_proves_whole_span(content, candidate, &base))
				{
					let one_line = base.split_whitespace().collect::<Vec<_>>().join(" ");
					let repeated = vec![one_line; pattern.selection_ranges.len()];
					let header = edit_header(context.path, operation.all);
					let candidate = &candidates[0];
					return Err(EditError::matched(
						[
							format!(
								"Operation {number} has {} selections, but *** SM:PUT proves neither \
								 positional substitution nor whole-span replacement.",
								pattern.selection_ranges.len()
							),
							"Copy-ready per-selection interpretation:".to_owned(),
							format!(
								"{header}\n*** SM:FIND\n{}\n*** SM:PUT\n{}",
								operation.pattern_text,
								repeated.join("\n")
							),
							"Copy-ready whole-span interpretation:".to_owned(),
							format!(
								"{header}\n*** SM:FIND\n{}\n*** SM:PUT\n{}",
								operation.pattern_text,
								rewrite_selection_spans(content, candidate, &repeated)
							),
						]
						.join("\n"),
					));
				}
				let mut changes = 0;
				for located in &candidates {
					let mut candidate = located.clone();
					if operation.whitespace_matched
						&& pattern.line_insertion
						&& candidate.start == candidate.end
					{
						let snapped =
							snap_line_insertion_offset(content, candidate.start, !base.starts_with('\n'));
						candidate.start = snapped;
						candidate.end = snapped;
					}
					let rewrite = if pattern.line_insertion {
						frame_line_insertion(content, candidate.start, &base, false)
					} else {
						base.clone()
					};
					let rendered = render_rewrite(
						&rewrite,
						if candidate.captures.is_empty() {
							&[]
						} else {
							&pattern.selected_capture_indices
						},
						&candidate.captures,
						if candidate.literal_gaps {
							EdgeGaps::default()
						} else {
							pattern.edge_gaps
						},
						number,
					)?;
					let replacement = align_boundary_echoes(content, &candidate, &rendered);
					let deleted = replacement
						.is_empty()
						.then(|| content[candidate.start..candidate.end].to_owned());
					if deleted.is_some() {
						candidate = expand_full_line_deletion(content, &candidate);
					}
					if candidates.len() == 1
						&& let Some(deleted) = &deleted
					{
						removed[index] = Some(deleted.clone());
					}
					if let Some(deleted) = deleted {
						let lines = deleted
							.lines()
							.filter(|line| !line.trim().is_empty())
							.count();
						deletion_notes.insert(
							number,
							if operation.assumed_deletion {
								format!(
									"Note: operation {number} had no *** SM:PUT and was applied as a move \
									 deletion (a later operation re-emits its block)."
								)
							} else {
								format!(
									"Note: operation {number} deleted {lines} line(s); an empty *** SM:PUT \
									 means deletion — resend with the final text if you meant to replace."
								)
							},
						);
					}
					last_match = candidate.match_start;
					let same = content[candidate.start..candidate.end] == replacement
						|| operation.desired_state
							&& normalize_text(&replacement).text
								== normalize_text(&content[candidate.start..candidate.end]).text;
					if same {
						if operation.all {
							continue;
						}
						if let Some((start, end)) =
							duplicate_collapse_span(content, &candidate, &replacement)
						{
							planned.push(PlannedEdit {
								start,
								end,
								replacement,
								operation_number: number,
							});
							changes += 1;
							continue;
						}
						if operation.desired_state {
							recovery_notes.push(format!(
								"Note: operation {number} already matches the file; no change was needed \
								 there."
							));
							changes += 1;
							continue;
						}
						return Err(no_op_error(
							context,
							payload,
							Some(number),
							Some((content, candidate.match_start)),
							None,
							None,
						));
					}
					planned.push(PlannedEdit {
						start: candidate.start,
						end: candidate.end,
						replacement,
						operation_number: number,
					});
					changes += 1;
				}
				if operation.all && changes == 0 {
					return Err(no_op_error(
						context,
						payload,
						Some(number),
						Some((content, candidates[0].match_start)),
						Some(candidates.len()),
						None,
					));
				}
			},
		}
	}
	if planned.is_empty() {
		return Err(no_op_error(context, payload, None, Some((content, last_match)), None, None));
	}
	planned.sort_by_key(|edit| (edit.start, edit.end));
	let mut ordered: Vec<PlannedEdit> = Vec::new();
	for current in planned {
		let Some(previous) = ordered.last().cloned() else {
			ordered.push(current);
			continue;
		};
		let overlaps = current.start < previous.end
			|| current.start == previous.start
				&& current.end == current.start
				&& previous.end == previous.start;
		if !overlaps {
			ordered.push(current);
			continue;
		}
		if let Some(merged) = reconcile_overlap(content, &previous, &current) {
			*ordered.last_mut().expect("present") = merged;
			continue;
		}
		if previous.operation_number == current.operation_number {
			continue;
		}
		let first_line = line_number_at(content, previous.start);
		let second_line = line_number_at(content, current.start);
		return Err(EditError::matched(format!(
			"Operations {} and {} target overlapping original spans near lines {first_line} and \
			 {second_line}.\n\nConflicting candidates:\n\nOperation {} near line \
			 {first_line}:\n{}\n\nOperation {} near line {second_line}:\n{}\n\nKeep whichever states \
			 the intended final text and drop the other.",
			previous.operation_number,
			current.operation_number,
			previous.operation_number,
			operation_payload(&operations[previous.operation_number - 1], context.path, false, None),
			current.operation_number,
			operation_payload(&operations[current.operation_number - 1], context.path, false, None)
		)));
	}
	let mut result = content.to_owned();
	for edit in ordered.into_iter().rev() {
		result.replace_range(edit.start..edit.end, &edit.replacement);
	}
	if result == content {
		return Err(no_op_error(context, payload, None, Some((content, last_match)), None, None));
	}
	context.store.reset_noop(context.canonical);
	context.notes.extend(recovery_notes);
	context.notes.extend(deletion_notes.into_values());
	Ok(result)
}

/// Parse, locate, and atomically apply one sloppy section.
pub fn apply_sloppy(
	content: &str,
	input: &str,
	mut context: ApplyContext<'_>,
) -> Result<String, EditError> {
	apply_operations(content, input, &mut context).map_err(|error| {
		let message = error.to_string();
		EditError::matched(if message.contains(ATOMICITY_NOTICE) {
			message
		} else {
			format!("{ATOMICITY_NOTICE}\n{message}")
		})
	})
}
