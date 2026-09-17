//! Word-level diff with jsdiff `diffWords` semantics.
//!
//! Tokenizes UTF-16 text into word tokens, interns by trimmed value, runs Myers
//! diff, then post-processes to deduplicate boundary whitespace.

use crate::myers::{build_changes, intern, Change};

const fn is_word_char(cp: u32) -> bool {
	matches!(cp,
		0x30..=0x39 // 0-9
		| 0x41..=0x5A // A-Z
		| 0x5F // _
		| 0x61..=0x7A // a-z
		| 0xAD
		| 0xC0..=0xD6
		| 0xD8..=0xF6
		| 0xF8..=0x2C6
		| 0x2C8..=0x2D7
		| 0x2DE..=0x2FF
		| 0x1E00..=0x1EFF)
}

/// JavaScript's `\s` / `String.prototype.trim` whitespace set (`WhiteSpace` +
/// `LineTerminator` productions). Every member is a single UTF-16 code unit,
/// so unit-level scans here match jsdiff's code-unit-level scans exactly.
const fn is_js_whitespace(cp: u32) -> bool {
	matches!(
		cp,
		0x09 | 0x0a | 0x0b | 0x0c | 0x0d | 0x20 | 0xa0 | 0x1680 | 0x2000
			..=0x200a | 0x2028 | 0x2029 | 0x202f | 0x205f | 0x3000 | 0xfeff
	)
}

const fn is_ws_unit(unit: u16) -> bool {
	is_js_whitespace(unit as u32)
}

fn trim_leading_ws(s: &[u16]) -> &[u16] {
	let start = s
		.iter()
		.position(|&unit| !is_ws_unit(unit))
		.unwrap_or(s.len());
	&s[start..]
}

fn trim_trailing_ws(s: &[u16]) -> &[u16] {
	let end = s
		.iter()
		.rposition(|&unit| !is_ws_unit(unit))
		.map_or(0, |i| i + 1);
	&s[..end]
}

fn leading_ws(s: &[u16]) -> &[u16] {
	&s[..s.len() - trim_leading_ws(s).len()]
}

fn trailing_ws(s: &[u16]) -> &[u16] {
	&s[trim_trailing_ws(s).len()..]
}

fn js_trim(s: &[u16]) -> &[u16] {
	trim_trailing_ws(trim_leading_ws(s))
}

/// Iterator over `(start, code_point, unit_len)` that pairs surrogates and
/// passes unpaired surrogates through as their own code points, exactly like
/// JS regex scanning under the `u` flag.
struct CodePoints<'a> {
	text: &'a [u16],
	pos:  usize,
}

impl Iterator for CodePoints<'_> {
	type Item = (usize, u32, usize);

	fn next(&mut self) -> Option<Self::Item> {
		let &unit = self.text.get(self.pos)?;
		let start = self.pos;
		if matches!(unit, 0xd800..=0xdbff)
			&& let Some(&low) = self.text.get(start + 1)
			&& matches!(low, 0xdc00..=0xdfff)
		{
			self.pos += 2;
			let cp = 0x10000 + ((u32::from(unit & 0x3ff) << 10) | u32::from(low & 0x3ff));
			return Some((start, cp, 2));
		}
		self.pos += 1;
		Some((start, u32::from(unit), 1))
	}
}

const fn code_points(text: &[u16]) -> CodePoints<'_> {
	CodePoints { text, pos: 0 }
}

/// Raw regex-equivalent scan: word runs, whitespace runs, or single other
/// code points (jsdiff `tokenizeIncludingWhitespace` with the `u` flag).
fn word_parts(text: &[u16]) -> Vec<&[u16]> {
	let mut parts = Vec::new();
	let mut iter = code_points(text).peekable();
	while let Some((start, cp, len)) = iter.next() {
		let class = if is_word_char(cp) {
			1u8
		} else if is_js_whitespace(cp) {
			2u8
		} else {
			0u8
		};
		let mut end = start + len;
		if class != 0 {
			while let Some(&(_, next_cp, next_len)) = iter.peek() {
				let same = if class == 1 {
					is_word_char(next_cp)
				} else {
					is_js_whitespace(next_cp)
				};
				if !same {
					break;
				}
				end += next_len;
				iter.next();
			}
		}
		parts.push(&text[start..end]);
	}
	parts
}

/// jsdiff `WordDiff.tokenize`: stitch whitespace runs onto adjacent word or
/// punctuation parts, duplicating interior whitespace into both neighbors.
fn word_tokens(text: &[u16]) -> Vec<Vec<u16>> {
	let parts = word_parts(text);
	let mut tokens: Vec<Vec<u16>> = Vec::with_capacity(parts.len());
	let mut prev_part: Option<&[u16]> = None;
	for part in parts {
		let part_is_ws = part.first().is_some_and(|&unit| is_ws_unit(unit));
		if part_is_ws {
			if prev_part.is_none() {
				tokens.push(part.to_vec());
			} else {
				let last = tokens
					.last_mut()
					.expect("tokens non-empty after first part");
				last.extend_from_slice(part);
			}
		} else if let Some(prev) =
			prev_part.filter(|p| p.first().is_some_and(|&unit| is_ws_unit(unit)))
		{
			if tokens.last().is_some_and(|last| last.as_slice() == prev) {
				let last = tokens.last_mut().expect("checked non-empty");
				last.extend_from_slice(part);
			} else {
				let mut token = Vec::with_capacity(prev.len() + part.len());
				token.extend_from_slice(prev);
				token.extend_from_slice(part);
				tokens.push(token);
			}
		} else {
			tokens.push(part.to_vec());
		}
		prev_part = Some(part);
	}
	tokens
}

/// jsdiff `WordDiff.join`: concatenate, stripping leading whitespace from
/// every token after the first.
fn word_join(tokens: &[&[u16]]) -> Vec<u16> {
	let mut out = Vec::new();
	for (i, token) in tokens.iter().enumerate() {
		if i == 0 {
			out.extend_from_slice(token);
		} else {
			out.extend_from_slice(trim_leading_ws(token));
		}
	}
	out
}

fn longest_common_prefix<'a>(a: &'a [u16], b: &[u16]) -> &'a [u16] {
	let len = a.iter().zip(b).take_while(|(x, y)| x == y).count();
	&a[..len]
}

fn longest_common_suffix<'a>(a: &'a [u16], b: &[u16]) -> &'a [u16] {
	let len = a
		.iter()
		.rev()
		.zip(b.iter().rev())
		.take_while(|(x, y)| x == y)
		.count();
	&a[a.len() - len..]
}

fn remove_prefix(s: &[u16], prefix: &[u16]) -> Vec<u16> {
	s.strip_prefix(prefix)
		.expect("value must start with recorded prefix")
		.to_vec()
}

fn remove_suffix(s: &[u16], suffix: &[u16]) -> Vec<u16> {
	s.strip_suffix(suffix)
		.expect("value must end with recorded suffix")
		.to_vec()
}

fn replace_prefix(s: &[u16], old_prefix: &[u16], new_prefix: &[u16]) -> Vec<u16> {
	let rest = s
		.strip_prefix(old_prefix)
		.expect("value must start with recorded prefix");
	let mut out = Vec::with_capacity(new_prefix.len() + rest.len());
	out.extend_from_slice(new_prefix);
	out.extend_from_slice(rest);
	out
}

fn replace_suffix(s: &[u16], old_suffix: &[u16], new_suffix: &[u16]) -> Vec<u16> {
	let rest = s
		.strip_suffix(old_suffix)
		.expect("value must end with recorded suffix");
	let mut out = Vec::with_capacity(rest.len() + new_suffix.len());
	out.extend_from_slice(rest);
	out.extend_from_slice(new_suffix);
	out
}

/// jsdiff `maximumOverlap`: the longest prefix of `b` that is also a suffix
/// of `a`, via the KMP failure function over code units.
fn maximum_overlap<'a>(a: &[u16], b: &'a [u16]) -> &'a [u16] {
	let start_a = a.len().saturating_sub(b.len());
	let end_b = b.len().min(a.len());
	if end_b == 0 {
		return &[];
	}
	let mut map = vec![0usize; end_b];
	let mut k = 0usize;
	for j in 1..end_b {
		if b[j] == b[k] {
			map[j] = map[k];
		} else {
			map[j] = k;
		}
		while k > 0 && b[j] != b[k] {
			k = map[k];
		}
		if b[j] == b[k] {
			k += 1;
		}
	}
	k = 0;
	for &unit in &a[start_a..] {
		while k > 0 && unit != b[k] {
			k = map[k];
		}
		if unit == b[k] {
			k += 1;
		}
	}
	&b[..k]
}

/// jsdiff `dedupeWhitespaceInChangeObjects` (no segmenter): trim whitespace
/// that the tokenizer duplicated across a keep/delete/insert boundary.
fn dedupe_whitespace(
	changes: &mut [Change<Vec<u16>>],
	start_keep: Option<usize>,
	deletion: Option<usize>,
	insertion: Option<usize>,
	end_keep: Option<usize>,
) {
	match (deletion, insertion) {
		(Some(del), Some(ins)) => {
			let old_ws_prefix = leading_ws(&changes[del].value).to_vec();
			let old_ws_suffix = trailing_ws(&changes[del].value).to_vec();
			let new_ws_prefix = leading_ws(&changes[ins].value).to_vec();
			let new_ws_suffix = trailing_ws(&changes[ins].value).to_vec();
			if let Some(start) = start_keep {
				let common_ws_prefix = longest_common_prefix(&old_ws_prefix, &new_ws_prefix).to_vec();
				changes[start].value =
					replace_suffix(&changes[start].value, &new_ws_prefix, &common_ws_prefix);
				changes[del].value = remove_prefix(&changes[del].value, &common_ws_prefix);
				changes[ins].value = remove_prefix(&changes[ins].value, &common_ws_prefix);
			}
			if let Some(end) = end_keep {
				let common_ws_suffix = longest_common_suffix(&old_ws_suffix, &new_ws_suffix).to_vec();
				changes[end].value =
					replace_prefix(&changes[end].value, &new_ws_suffix, &common_ws_suffix);
				changes[del].value = remove_suffix(&changes[del].value, &common_ws_suffix);
				changes[ins].value = remove_suffix(&changes[ins].value, &common_ws_suffix);
			}
		},
		(None, Some(ins)) => {
			if start_keep.is_some() {
				let ws_len = leading_ws(&changes[ins].value).len();
				changes[ins].value = changes[ins].value[ws_len..].to_vec();
			}
			if let Some(end) = end_keep {
				let ws_len = leading_ws(&changes[end].value).len();
				changes[end].value = changes[end].value[ws_len..].to_vec();
			}
		},
		(Some(del), None) => match (start_keep, end_keep) {
			(Some(start), Some(end)) => {
				let new_ws_full = leading_ws(&changes[end].value).to_vec();
				let del_ws_start = leading_ws(&changes[del].value).to_vec();
				let del_ws_end = trailing_ws(&changes[del].value).to_vec();
				let new_ws_start = longest_common_prefix(&new_ws_full, &del_ws_start).to_vec();
				changes[del].value = remove_prefix(&changes[del].value, &new_ws_start);
				let new_ws_end =
					longest_common_suffix(&new_ws_full[new_ws_start.len()..], &del_ws_end).to_vec();
				changes[del].value = remove_suffix(&changes[del].value, &new_ws_end);
				changes[end].value = replace_prefix(&changes[end].value, &new_ws_full, &new_ws_end);
				let start_ws = &new_ws_full[..new_ws_full.len() - new_ws_end.len()];
				changes[start].value = replace_suffix(&changes[start].value, &new_ws_full, start_ws);
			},
			(None, Some(end)) => {
				let end_keep_ws_prefix = leading_ws(&changes[end].value).to_vec();
				let deletion_ws_suffix = trailing_ws(&changes[del].value).to_vec();
				let overlap = maximum_overlap(&deletion_ws_suffix, &end_keep_ws_prefix).to_vec();
				changes[del].value = remove_suffix(&changes[del].value, &overlap);
			},
			(Some(start), None) => {
				let start_keep_ws_suffix = trailing_ws(&changes[start].value).to_vec();
				let deletion_ws_prefix = leading_ws(&changes[del].value).to_vec();
				let overlap = maximum_overlap(&start_keep_ws_suffix, &deletion_ws_prefix).to_vec();
				changes[del].value = remove_prefix(&changes[del].value, &overlap);
			},
			(None, None) => {},
		},
		(None, None) => {},
	}
}

/// jsdiff `WordDiff.postProcess` under default options.
fn word_post_process(changes: &mut [Change<Vec<u16>>]) {
	let mut last_keep: Option<usize> = None;
	let mut insertion: Option<usize> = None;
	let mut deletion: Option<usize> = None;
	for i in 0..changes.len() {
		if changes[i].added {
			insertion = Some(i);
		} else if changes[i].removed {
			deletion = Some(i);
		} else {
			if insertion.is_some() || deletion.is_some() {
				dedupe_whitespace(changes, last_keep, deletion, insertion, Some(i));
			}
			last_keep = Some(i);
			insertion = None;
			deletion = None;
		}
	}
	if insertion.is_some() || deletion.is_some() {
		dedupe_whitespace(changes, last_keep, deletion, insertion, None);
	}
}

/// Word diff with jsdiff `diffWords` semantics over UTF-16 code units.
pub fn diff_words_u16(old_text: &[u16], new_text: &[u16]) -> Vec<Change<Vec<u16>>> {
	let old_tokens = word_tokens(old_text);
	let new_tokens = word_tokens(new_text);
	let old_refs: Vec<&[u16]> = old_tokens.iter().map(Vec::as_slice).collect();
	let new_refs: Vec<&[u16]> = new_tokens.iter().map(Vec::as_slice).collect();
	// Equality is whitespace-insensitive: intern by trimmed text.
	let old_keys: Vec<&[u16]> = old_refs.iter().map(|token| js_trim(token)).collect();
	let new_keys: Vec<&[u16]> = new_refs.iter().map(|token| js_trim(token)).collect();
	let (old_ids, new_ids) = intern(&old_keys, &new_keys);
	let runs = crate::myers::myers_diff(&old_ids, &new_ids);
	let mut changes = build_changes(&runs, &old_refs, &new_refs, word_join);
	word_post_process(&mut changes);
	changes
}
