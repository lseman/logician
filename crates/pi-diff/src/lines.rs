//! jsdiff-compatible line-level diff primitives.
//!
//! UTF-16 entry points operate on JavaScript code units; UTF-8 helpers serve
//! native Rust callers with the same token and run semantics.

use std::hash::Hash;

use crate::myers::{build_changes, intern, Run};

/// UTF-16 code unit for `\n`.
pub const LF: u16 = 0x000a;

/// jsdiff line tokenization over UTF-16 code units.
pub fn line_tokens_u16(text: &[u16]) -> Vec<&[u16]> {
	text.split_inclusive(|&unit| unit == LF).collect()
}

/// jsdiff line tokenization over UTF-8 text.
pub fn line_tokens_str(text: &str) -> Vec<&str> {
	text.split_inclusive('\n').collect()
}

fn diff_line_tokens<T: Eq + Hash + Copy>(old_tokens: &[T], new_tokens: &[T]) -> Vec<Run> {
	let (old_ids, new_ids) = intern(old_tokens, new_tokens);
	crate::myers::myers_diff(&old_ids, &new_ids)
}

/// Myers runs for jsdiff line tokens over UTF-16 code units.
pub fn line_runs_u16(old: &[u16], new: &[u16]) -> Vec<Run> {
	let old_tokens = line_tokens_u16(old);
	let new_tokens = line_tokens_u16(new);
	diff_line_tokens(&old_tokens, &new_tokens)
}

/// Myers runs for jsdiff line tokens over UTF-8 text.
pub fn line_runs_str(old: &str, new: &str) -> Vec<Run> {
	let old_tokens = line_tokens_str(old);
	let new_tokens = line_tokens_str(new);
	diff_line_tokens(&old_tokens, &new_tokens)
}

/// Concatenate UTF-16 token slices.
pub fn concat_tokens_u16(tokens: &[&[u16]]) -> Vec<u16> {
	let mut out = Vec::with_capacity(tokens.iter().map(|token| token.len()).sum());
	for token in tokens {
		out.extend_from_slice(token);
	}
	out
}

fn concat_tokens_str(tokens: &[&str]) -> String {
	let mut out = String::with_capacity(tokens.iter().map(|token| token.len()).sum());
	for token in tokens {
		out.push_str(token);
	}
	out
}

/// Line changes with jsdiff `diffLines` semantics over UTF-16 code units.
pub fn diff_lines_u16(old: &[u16], new: &[u16]) -> Vec<crate::myers::Change<Vec<u16>>> {
	let old_tokens = line_tokens_u16(old);
	let new_tokens = line_tokens_u16(new);
	let runs = diff_line_tokens(&old_tokens, &new_tokens);
	build_changes(&runs, &old_tokens, &new_tokens, concat_tokens_u16)
}

/// Line changes with jsdiff `diffLines` semantics over UTF-8 text.
pub fn line_changes_str(old: &str, new: &str) -> Vec<crate::myers::Change<String>> {
	let old_tokens = line_tokens_str(old);
	let new_tokens = line_tokens_str(new);
	let runs = diff_line_tokens(&old_tokens, &new_tokens);
	build_changes(&runs, &old_tokens, &new_tokens, concat_tokens_str)
}
