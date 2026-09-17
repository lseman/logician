//! jsdiff-compatible unified diff hunks.

use crate::myers::Run;
use crate::lines::LF;

/// One hunk of a unified diff.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hunk {
	/// 1-based first line of the hunk in the old text.
	pub old_start: u32,
	/// Number of old-text lines covered by the hunk.
	pub old_lines: u32,
	/// 1-based first line of the hunk in the new text.
	pub new_start: u32,
	/// Number of new-text lines covered by the hunk.
	pub new_lines: u32,
	/// Prefixed hunk body lines without trailing newlines.
	pub lines:     Vec<Vec<u16>>,
}

/// Prepend a unified-diff marker to a UTF-16 line.
fn prefixed_line(prefix: u8, line: &[u16]) -> Vec<u16> {
	let mut out = Vec::with_capacity(1 + line.len());
	out.push(u16::from(prefix));
	out.extend_from_slice(line);
	out
}

/// `\ No newline at end of file`, as UTF-16 code units.
fn no_newline_marker() -> Vec<u16> {
	"\\ No newline at end of file".encode_utf16().collect()
}

/// Build jsdiff-compatible unified hunks from UTF-16 texts.
pub fn structured_patch_hunks_u16(
	old_text: &[u16],
	new_text: &[u16],
	context: Option<u32>,
) -> Vec<Hunk> {
	let old_tokens: Vec<&[u16]> = crate::lines::line_tokens_u16(old_text);
	let new_tokens: Vec<&[u16]> = crate::lines::line_tokens_u16(new_text);
	let runs = diff_line_tokens(&old_tokens, &new_tokens);
	structured_patch_hunks_from_runs_u16(context, &old_tokens, &new_tokens, &runs)
}

/// Build jsdiff-compatible unified hunks from precomputed line runs.
pub fn structured_patch_hunks_from_runs_u16(
	context: Option<u32>,
	old_tokens: &[&[u16]],
	new_tokens: &[&[u16]],
	runs: &[Run],
) -> Vec<Hunk> {
	let context = context.map_or(4usize, |value| value as usize);

	// Change list with per-change line slices; the trailing sentinel mirrors
	// jsdiff's pushed empty change that flushes the final hunk.
	struct ChangeLines<'a> {
		added:   bool,
		removed: bool,
		lines:   &'a [&'a [u16]],
	}
	let mut list: Vec<ChangeLines> = Vec::with_capacity(runs.len() + 1);
	let mut old_pos = 0usize;
	let mut new_pos = 0usize;
	for run in runs {
		let count = run.count as usize;
		let lines: &[&[u16]] = if run.removed {
			let slice = &old_tokens[old_pos..old_pos + count];
			old_pos += count;
			slice
		} else {
			let slice = &new_tokens[new_pos..new_pos + count];
			new_pos += count;
			if !run.added {
				old_pos += count;
			}
			slice
		};
		list.push(ChangeLines { added: run.added, removed: run.removed, lines });
	}
	list.push(ChangeLines { added: false, removed: false, lines: &[] });

	// Hunk skeleton before the trailing-newline post-pass; lines stay `Vec<u16>`
	// so the pass below can pop terminators in place.
	struct RawHunk {
		old_start: usize,
		old_lines: usize,
		new_start: usize,
		new_lines: usize,
		lines:     Vec<Vec<u16>>,
	}
	let mut hunks: Vec<RawHunk> = Vec::new();
	let mut old_range_start = 0usize;
	let mut new_range_start = 0usize;
	let mut cur_range: Vec<Vec<u16>> = Vec::new();
	let mut old_line = 1usize;
	let mut new_line = 1usize;
	for i in 0..list.len() {
		let current = &list[i];
		if current.added || current.removed {
			// Open a hunk seeded with trailing context from the previous
			// common run.
			if old_range_start == 0 {
				old_range_start = old_line;
				new_range_start = new_line;
				if i > 0 && context > 0 {
					let prev_lines = list[i - 1].lines;
					let take = prev_lines.len().min(context);
					cur_range = prev_lines[prev_lines.len() - take..]
						.iter()
						.map(|line| prefixed_line(b' ', line))
						.collect();
					old_range_start -= cur_range.len();
					new_range_start -= cur_range.len();
				}
			}
			let marker = if current.added { b'+' } else { b'-' };
			for line in current.lines {
				cur_range.push(prefixed_line(marker, line));
			}
			if current.added {
				new_line += current.lines.len();
			} else {
				old_line += current.lines.len();
			}
		} else {
			if old_range_start != 0 {
				if current.lines.len() <= context * 2 && i + 2 < list.len() {
					// Common run small enough to join adjacent hunks.
					for line in current.lines {
						cur_range.push(prefixed_line(b' ', line));
					}
				} else {
					// Close the hunk with leading context.
					let context_size = current.lines.len().min(context);
					for line in &current.lines[..context_size] {
						cur_range.push(prefixed_line(b' ', line));
					}
					hunks.push(RawHunk {
						old_start: old_range_start,
						old_lines: old_line - old_range_start + context_size,
						new_start: new_range_start,
						new_lines: new_line - new_range_start + context_size,
						lines:     std::mem::take(&mut cur_range),
					});
					old_range_start = 0;
					new_range_start = 0;
				}
			}
			old_line += current.lines.len();
			new_line += current.lines.len();
		}
	}

	// Strip trailing newlines and add "no newline at EOF" markers.
	for hunk in &mut hunks {
		let mut i = 0;
		while i < hunk.lines.len() {
			if hunk.lines[i].last() == Some(&LF) {
				hunk.lines[i].pop();
			} else {
				hunk.lines.insert(i + 1, no_newline_marker());
				i += 1;
			}
			i += 1;
		}
	}
	hunks
		.into_iter()
		.map(|hunk| Hunk {
			old_start: hunk.old_start as u32,
			old_lines: hunk.old_lines as u32,
			new_start: hunk.new_start as u32,
			new_lines: hunk.new_lines as u32,
			lines:     hunk.lines,
		})
		.collect()
}

fn diff_line_tokens<T: std::hash::Hash + Eq + Copy>(old_tokens: &[T], new_tokens: &[T]) -> Vec<Run> {
	let (old_ids, new_ids) = crate::myers::intern(old_tokens, new_tokens);
	crate::myers::myers_diff(&old_ids, &new_ids)
}
