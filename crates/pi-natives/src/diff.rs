//! N-API wrappers for jsdiff-compatible line and structured-patch diff
//! primitives.
//!
//! The Myers, line, and structured-patch implementation lives in `pi-diff`.
//! This module preserves JavaScript's UTF-16 code units at the N-API
//! boundary.
//!
//! # Example
//! ```ignore
//! // JS: native.diffLines("a\nb\n", "a\nc\n")
//! //   -> [{ value: "a\n", count: 1, added: false, removed: false },
//! //       { value: "b\n", count: 1, added: false, removed: true },
//! //       { value: "c\n", count: 1, added: true, removed: false }]
//! ```

use napi::{JsString, bindgen_prelude::*};
use napi_derive::napi;

use crate::js;

/// One jsdiff change object: a run of added, removed, or common tokens.
#[napi(object)]
pub struct DiffChange {
	/// Joined token text for this run (lines keep their `\n` terminators).
	pub value:   Utf16String,
	/// Number of tokens in this run.
	pub count:   u32,
	/// True when this run exists only in the new text.
	pub added:   bool,
	/// True when this run exists only in the old text.
	pub removed: bool,
}


/// One hunk of a unified diff, matching jsdiff `structuredPatch` hunks.
#[napi(object)]
pub struct PatchHunk {
	/// 1-based first line of the hunk in the old text.
	pub old_start: u32,
	/// Number of old-text lines covered by the hunk.
	pub old_lines: u32,
	/// 1-based first line of the hunk in the new text.
	pub new_start: u32,
	/// Number of new-text lines covered by the hunk.
	pub new_lines: u32,
	/// Hunk body: `+`/`-`/` `-prefixed lines without trailing newlines, plus
	/// `\ No newline at end of file` markers where applicable.
	pub lines:     Vec<Utf16String>,
}



// ═══════════════════════════════════════════════════════════════════════════
// Line diff
// ═══════════════════════════════════════════════════════════════════════════

/// Line diff with jsdiff `diffLines(oldText, newText)` semantics (default
/// options). Change values keep line terminators, and common runs are joined
/// from the new text.
#[napi]
pub fn diff_lines(old_text: JsString, new_text: JsString) -> Result<Vec<DiffChange>> {
	let old_text = js::utf16(old_text)?;
	let new_text = js::utf16(new_text)?;
	Ok(diff_lines_impl(&old_text, &new_text))
}

fn diff_lines_impl(old_text: &[u16], new_text: &[u16]) -> Vec<DiffChange> {
	pi_diff::diff_lines_u16(old_text, new_text)
		.into_iter()
		.map(|change| DiffChange {
			value:   change.value.into(),
			count:   change.count,
			added:   change.added,
			removed: change.removed,
		})
		.collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Structured patch
// ═══════════════════════════════════════════════════════════════════════════

/// Unified-diff hunks with jsdiff
/// `structuredPatch(_, _, oldText, newText, _, _, { context }).hunks`
/// semantics. `context` defaults to 4 like jsdiff.
#[napi]
pub fn structured_patch_hunks(
	old_text: JsString,
	new_text: JsString,
	context: Option<u32>,
) -> Result<Vec<PatchHunk>> {
	let old_text = js::utf16(old_text)?;
	let new_text = js::utf16(new_text)?;
	Ok(structured_patch_hunks_impl(&old_text, &new_text, context))
}

fn structured_patch_hunks_impl(
	old_text: &[u16],
	new_text: &[u16],
	context: Option<u32>,
) -> Vec<PatchHunk> {
	pi_diff::structured_patch_hunks_u16(old_text, new_text, context)
		.into_iter()
		.map(patch_hunk)
		.collect()
}

fn patch_hunk(hunk: pi_diff::Hunk) -> PatchHunk {
	PatchHunk {
		old_start: hunk.old_start,
		old_lines: hunk.old_lines,
		new_start: hunk.new_start,
		new_lines: hunk.new_lines,
		lines:     hunk.lines.into_iter().map(Utf16String::from).collect(),
	}
}
