//! Model-facing diff rendering and unified-diff hunk parsing.

pub mod generate;
pub mod parse;
pub mod preview;
pub mod types;

pub use types::{
	BlockContextSource, CompactDiffOptions, CompactDiffPreview, DiffHunk, DiffOutput,
};
pub use generate::{find_block_context_lines, generate_diff_string, generate_unified_diff_string};
pub use preview::build_compact_diff_preview;
pub use parse::{normalize_create_content, normalize_diff, parse_diff_hunks};
