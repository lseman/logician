//! jsdiff-compatible diff primitives without an FFI dependency.
//!
//! The Myers O(ND) core and line, word, and structured-patch helpers preserve
//! jsdiff v9's default tie-breaking and change coalescing. UTF-16 entry points
//! operate on JavaScript code units, while UTF-8 line helpers serve native Rust
//! callers with the same token and run semantics.

mod lines;
mod myers;
mod unified;
mod words;

// -- Public re-exports --

pub use lines::{
	concat_tokens_u16, diff_lines_u16, line_changes_str, line_runs_str, line_runs_u16, line_tokens_str,
	line_tokens_u16, LF,
};
pub use myers::{build_changes, intern, Change, Run};
pub use unified::{structured_patch_hunks_from_runs_u16, structured_patch_hunks_u16, Hunk};
pub use words::diff_words_u16;


#[cfg(test)]
mod tests {
	use super::*;

	fn u16s(text: &str) -> Vec<u16> {
		text.encode_utf16().collect()
	}

	fn lines(old: &str, new: &str) -> Vec<(String, bool, bool)> {
		diff_lines_u16(&u16s(old), &u16s(new))
			.into_iter()
			.map(|change| (String::from_utf16(&change.value).unwrap(), change.added, change.removed))
			.collect()
	}

	#[test]
	fn line_diff_replaces_middle_line() {
		assert_eq!(lines("a\nb\nc\n", "a\nx\nc\n"), vec![
			("a\n".into(), false, false),
			("b\n".into(), false, true),
			("x\n".into(), true, false),
			("c\n".into(), false, false),
		]);
	}

	#[test]
	fn line_diff_treats_missing_trailing_newline_as_distinct() {
		assert_eq!(lines("a\nb", "a\nb\n"), vec![
			("a\n".into(), false, false),
			("b".into(), false, true),
			("b\n".into(), true, false),
		]);
	}

	#[test]
	fn utf8_and_utf16_line_changes_agree_for_ascii() {
		let old = "a\nb\nc\n";
		let new = "a\nx\nc\n";
		let utf8 = line_changes_str(old, new);
		let utf16 = diff_lines_u16(&u16s(old), &u16s(new));
		let utf16_shaped: Vec<Change<String>> = utf16
			.into_iter()
			.map(|change| Change {
				value:   String::from_utf16(&change.value).unwrap(),
				count:   change.count,
				added:   change.added,
				removed: change.removed,
			})
			.collect();
		assert_eq!(utf8, utf16_shaped);
	}

	#[test]
	fn common_runs_take_values_from_new_tokens() {
		let old = String::from("same");
		let new = String::from("same");
		let old_tokens = [old.as_str()];
		let new_tokens = [new.as_str()];
		let changes = build_changes(
			&[Run { count: 1, added: false, removed: false }],
			&old_tokens,
			&new_tokens,
			|tokens| tokens[0].as_ptr(),
		);
		assert_eq!(changes[0].value, new.as_ptr());
		assert_ne!(changes[0].value, old.as_ptr());
		assert_eq!(line_changes_str(&old, &new)[0].value, new);
	}

	#[test]
	fn structured_patch_marks_missing_eof_newline() {
		let hunks = structured_patch_hunks_u16(&u16s("a\nb"), &u16s("a\nc"), Some(3));
		assert_eq!(hunks.len(), 1);
		let body: Vec<String> = hunks[0]
			.lines
			.iter()
			.map(|line| String::from_utf16(line).unwrap())
			.collect();
		assert_eq!(body, vec![
			" a",
			"-b",
			"\\ No newline at end of file",
			"+c",
			"\\ No newline at end of file"
		]);
	}

	#[test]
	fn word_diff_dedupes_boundary_whitespace() {
		let changes = diff_words_u16(&u16s("foo bar baz"), &u16s("foo qux baz"));
		let shaped: Vec<(String, bool, bool)> = changes
			.into_iter()
			.map(|change| (String::from_utf16(&change.value).unwrap(), change.added, change.removed))
			.collect();
		assert_eq!(shaped, vec![
			("foo ".into(), false, false),
			("bar".into(), false, true),
			("qux".into(), true, false),
			(" baz".into(), false, false),
		]);
	}

	#[test]
	fn line_runs_preserve_empty_lines() {
		let old = u16s("a\n\nb");
		let new = u16s("a\n\nc");
		let runs = line_runs_u16(&old, &new);
		let shaped: Vec<(u32, bool, bool)> = runs
			.into_iter()
			.map(|run| (run.count, run.added, run.removed))
			.collect();
		assert_eq!(shaped, vec![(2, false, false), (1, false, true), (1, true, false)]);
	}

	#[test]
	fn unpaired_surrogates_diff_as_distinct_content() {
		let old = [0x61, 0xd800, LF];
		let new = [0x61, 0xd801, LF];
		let shaped: Vec<(Vec<u16>, bool, bool)> = diff_lines_u16(&old, &new)
			.into_iter()
			.map(|change| (change.value, change.added, change.removed))
			.collect();
		assert_eq!(shaped, vec![(old.to_vec(), false, true), (new.to_vec(), true, false)]);
	}

	#[test]
	fn word_scan_keeps_lone_surrogate_before_astral_pair_separate() {
		let old: Vec<u16> = [0xd800, 0xd83d, 0xde80].to_vec();
		let new: Vec<u16> = [0xd800, 0x78].to_vec();
		let shaped: Vec<(Vec<u16>, bool, bool)> = diff_words_u16(&old, &new)
			.into_iter()
			.map(|change| (change.value, change.added, change.removed))
			.collect();
		assert_eq!(shaped, vec![
			(vec![0xd800], false, false),
			(vec![0xd83d, 0xde80], false, true),
			(vec![0x78], true, false),
		]);
	}
}
