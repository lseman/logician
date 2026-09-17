//! Shared types and constants for diff rendering.

use std::{collections::BTreeSet, sync::LazyLock};

use regex::Regex;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Rendered diff plus the 1-indexed first changed line in the new text.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DiffOutput {
	/// Numbered diff rows.
	pub diff: String,
	/// First line affected in the new text.
	pub first_changed_line: Option<u32>,
}

/// Where the source came from, so tree-sitter can pick a grammar.
#[derive(Debug, Clone, Default)]
pub struct BlockContextSource<'a> {
	/// File path used for language inference.
	pub path: Option<&'a str>,
	/// Explicit language alias.
	pub lang: Option<&'a str>,
}
/// Compact preview of a numbered diff for the model-visible result.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompactDiffPreview {
	/// Current-file numbered preview.
	pub preview: String,
	/// Number of added rows in the source diff.
	pub added_lines: usize,
	/// Number of removed rows in the source diff.
	pub removed_lines: usize,
}

/// Options for [`build_compact_diff_preview`].
#[derive(Debug, Clone, Default)]
pub struct CompactDiffOptions {
	/// Added lines kept on each side of a long added run.
	pub max_added_run_context: Option<usize>,
	/// Back-compatible alias for `max_added_run_context`.
	pub max_unchanged_run: Option<usize>,
}

/// One hunk of a `patch`-mode diff body.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DiffHunk {
	/// Optional textual or symbolic hunk anchor.
	pub change_context: Option<String>,
	/// Optional 1-indexed old-file line hint.
	pub old_start_line: Option<u32>,
	/// Optional 1-indexed new-file line hint.
	pub new_start_line: Option<u32>,
	/// Whether the hunk contains unchanged context.
	pub has_context_lines: bool,
	/// Expected old-file lines.
	pub old_lines: Vec<String>,
	/// Replacement new-file lines.
	pub new_lines: Vec<String>,
	/// Whether the hunk carries an end-of-file marker.
	pub is_end_of_file: bool,
}

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

pub(super) const DIFF_GAP_ROW: &str = "";
pub(super) const EOF_MARKER: &str = "*** End of File";
pub(super) const CHANGE_CONTEXT_MARKER: &str = "@@ ";
pub(super) const EMPTY_CHANGE_CONTEXT_MARKER: &str = "@@";
pub(super) const MULTI_FILE_MARKERS: [&str; 4] =
	["*** Update File:", "*** Add File:", "*** Delete File:", "diff --git "];
pub(super) const DIFF_METADATA_PREFIXES: [&str; 15] = [
	"*** Update File:",
	"*** Add File:",
	"*** Delete File:",
	"diff --git ",
	"index ",
	"--- ",
	"+++ ",
	"new file mode ",
	"deleted file mode ",
	"rename from ",
	"rename to ",
	"similarity index ",
	"dissimilarity index ",
	"old mode ",
	"new mode ",
];
pub(super) const PATCH_WRAPPER_PREFIXES: [&str; 2] = ["*** Begin Patch", "*** End Patch"];
pub(super) const DEFAULT_ADDED_RUN_CONTEXT_LINES: usize = 2;
pub(super) const PREVIEW_ELISION_MARKER: &str = "…";
pub(super) const PREVIEW_GAP_ROW: &str = "";

// ---------------------------------------------------------------------------
// Regex patterns (used by parse module)
// ---------------------------------------------------------------------------

pub(super) static UNIFIED_HUNK_HEADER_REGEX: LazyLock<Regex> = LazyLock::new(|| {
	Regex::new(r"^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@(?:\s*(.*))?$")
		.expect("valid unified hunk header regex")
});
pub(super) static LINE_HINT_REGEX: LazyLock<Regex> = LazyLock::new(|| {
	Regex::new(r"(?i)^lines?\s+(\d+)(?:\s*-\s*(\d+))?(?:\s*@@)?$").expect("valid line hint regex")
});
pub(super) static TOP_OF_FILE_REGEX: LazyLock<Regex> = LazyLock::new(|| {
	Regex::new(r"(?i)^(top|start|beginning)\s+of\s+file$").expect("valid top-of-file regex")
});
pub(super) static NUMBERED_LINE_REGEX: LazyLock<Regex> =
	LazyLock::new(|| Regex::new(r"^\s*(\d{1,6})\s+(.+)$").expect("valid numbered-line regex"));
