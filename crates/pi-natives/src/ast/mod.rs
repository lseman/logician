//! AST-aware structural search and rewrite powered by ast-grep.
//!
//! Four focused modules:
//! - **types** – all `#[napi(...)]` public types and shared internal types
//! - **find** – `astGrep` file-scoped pattern search
//! - **match** – `astMatch` in-memory pattern matching
//! - **edit** – `astEdit` structural rewrites across files
//!
//! Re-exports keep the public API surface identical to the flat `ast.rs`.

mod types;
mod find;
mod r#match;
mod edit;

// -- Public re-exports -------------------------------------------------------

pub use types::*;
pub use find::ast_grep;
pub use r#match::ast_match;
pub use edit::{ast_edit, ast_edit_blocking};

// -- Tests --------------------------------------------------------------------

#[cfg(test)]
mod tests {
	use std::{
		collections::{BinaryHeap, HashMap},
		fs,
		path::PathBuf,
		time::{SystemTime, UNIX_EPOCH},
	};
	use crate::task;
	use ast_grep_core::source::Edit;
	use pi_ast::SupportLang;

	use super::*;

	struct TempTree {
		root: PathBuf,
	}

	impl Drop for TempTree {
		fn drop(&mut self) {
			let _ = fs::remove_dir_all(&self.root);
		}
	}

	fn make_temp_tree() -> TempTree {
		let unique = SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.expect("system time should be after UNIX_EPOCH")
			.as_nanos();
		let root = std::env::temp_dir().join(format!("pi-ast-glob-test-{unique}"));
		fs::create_dir_all(root.join("nested")).expect("temp nested dir should be created");
		fs::write(root.join("a.ts"), "const a = 1;\n").expect("temp file a.ts should be written");
		fs::write(root.join("nested").join("b.ts"), "const b = 2;\n")
			.expect("temp file nested/b.ts should be written");
		TempTree { root }
	}

	fn retained_test_match(line: u32) -> RetainedAstFindMatch {
		RetainedAstFindMatch {
			key:            AstFindOrderKey {
				path:         "file.ts".to_string(),
				start_line:   line,
				start_column: 1,
				end_line:     line,
				end_column:   2,
				byte_start:   line - 1,
				byte_end:     line,
				sequence:     u64::from(line),
			},
			text:           String::new(),
			meta_variables: None,
		}
	}

	#[test]
	fn retained_find_matches_keep_only_page_window() {
		let capacity = retained_find_capacity(1, 2);
		let mut retained = BinaryHeap::new();
		let mut materialized_payloads = 0usize;
		for line in 1..=100 {
			let candidate = retained_test_match(line);
			if should_retain_match(&retained, capacity, &candidate.key) {
				materialized_payloads += 1;
				retain_bounded_match(&mut retained, capacity, candidate);
			}
		}

		let (page, limit_reached) = page_retained_matches(retained, 1, 2);
		let lines = page
			.into_iter()
			.map(|retained| retained.key.start_line)
			.collect::<Vec<_>>();

		assert_eq!(materialized_payloads, capacity);
		assert!(limit_reached);
		assert_eq!(lines, vec![2, 3]);
	}

	#[test]
	fn glob_star_matches_only_direct_children() {
		let tree = make_temp_tree();
		let ct = task::CancelToken::default();
		let candidates =
			collect_candidates(Some(tree.root.to_string_lossy().into_owned()), Some("*.ts"), &ct)
				.expect("candidate collection should succeed");
		let paths = candidates
			.into_iter()
			.map(|file| file.display_path)
			.collect::<Vec<_>>();
		assert_eq!(paths, vec!["a.ts".to_string()]);
	}

	#[test]
	fn glob_double_star_matches_recursively() {
		let tree = make_temp_tree();
		let ct = task::CancelToken::default();
		let candidates =
			collect_candidates(Some(tree.root.to_string_lossy().into_owned()), Some("**/*.ts"), &ct)
				.expect("candidate collection should succeed");
		let paths = candidates
			.into_iter()
			.map(|file| file.display_path)
			.collect::<Vec<_>>();
		assert_eq!(paths, vec!["a.ts".to_string(), "nested/b.ts".to_string()]);
	}

	fn make_mixed_temp_tree() -> TempTree {
		let unique = SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.expect("system time should be after UNIX_EPOCH")
			.as_nanos();
		let root = std::env::temp_dir().join(format!("pi-ast-mixed-lang-test-{unique}"));
		fs::create_dir_all(&root).expect("temp mixed-lang dir should be created");
		fs::write(root.join("a.ts"), "const f = (x) => x;\n")
			.expect("temp file a.ts should be written");
		fs::write(root.join("b.rs"), "fn main() {}\n").expect("temp file b.rs should be written");
		TempTree { root }
	}

	#[test]
	fn ast_edit_rewrites_mixed_language_tree_per_file() {
		let tree = make_mixed_temp_tree();
		let a_path = tree.root.join("a.ts");
		let b_path = tree.root.join("b.rs");

		// The arrow pattern only matches TypeScript; the Rust file is searched in
		// its own language and left untouched instead of failing the whole call.
		let mut rewrites = HashMap::new();
		rewrites.insert("($X) => $X".to_string(), "identity".to_string());

		let result = ast_edit_blocking(
			task::CancelToken::default(),
			Some(rewrites),
			None,
			Some(tree.root.to_string_lossy().into_owned()),
			None,
			None,
			None,
			Some(false),
			None,
			None,
			None,
		)
		.expect("mixed-language tree should rewrite per file");

		assert_eq!(result.total_replacements, 1, "only the TypeScript file matches");
		assert_eq!(result.files_searched, 2);
		assert_eq!(
			fs::read_to_string(&a_path).expect("a.ts should be readable"),
			"const f = identity;\n",
		);
		assert_eq!(
			fs::read_to_string(&b_path).expect("b.rs should be readable"),
			"fn main() {}\n",
			"the Rust file must be untouched",
		);
	}

	#[test]
	fn resolves_supported_language_aliases() {
		assert_eq!(resolve_supported_lang("ts").ok(), Some(SupportLang::TypeScript));
		assert_eq!(resolve_supported_lang("jsx").ok(), Some(SupportLang::JavaScript));
		assert_eq!(resolve_supported_lang("rs").ok(), Some(SupportLang::Rust));
		assert_eq!(resolve_supported_lang("kotlin").ok(), Some(SupportLang::Kotlin));
		assert_eq!(resolve_supported_lang("bash").ok(), Some(SupportLang::Bash));
		assert_eq!(resolve_supported_lang("c").ok(), Some(SupportLang::C));
		assert_eq!(resolve_supported_lang("cpp").ok(), Some(SupportLang::Cpp));
		assert_eq!(resolve_supported_lang("tla").ok(), Some(SupportLang::Tlaplus));
		assert_eq!(resolve_supported_lang("pluscal").ok(), Some(SupportLang::Tlaplus));
		assert_eq!(resolve_supported_lang("emacs-lisp").ok(), Some(SupportLang::EmacsLisp));
		assert_eq!(resolve_supported_lang("elisp").ok(), Some(SupportLang::EmacsLisp));
		assert_eq!(resolve_supported_lang("el").ok(), Some(SupportLang::EmacsLisp));
		assert_eq!(resolve_supported_lang("f90").ok(), Some(SupportLang::Fortran));
		assert!(resolve_supported_lang("brainfuck").is_err());
	}

	#[test]
	fn applies_non_overlapping_edits() {
		let source = "const answer = 41;";
		let edits = vec![
			Edit::<String> { position: 6, deleted_length: 6, inserted_text: b"value".to_vec() },
			Edit::<String> { position: 15, deleted_length: 2, inserted_text: b"42".to_vec() },
		];
		let output = apply_edits(source, &edits).expect("edits should apply");
		assert_eq!(output, "const value = 42;");
	}

	#[test]
	fn rejects_overlapping_edits() {
		let source = "abcdef";
		let edits = vec![
			Edit::<String> { position: 1, deleted_length: 3, inserted_text: b"x".to_vec() },
			Edit::<String> { position: 2, deleted_length: 1, inserted_text: b"y".to_vec() },
		];
		assert!(apply_edits(source, &edits).is_err());
	}

	#[test]
	fn dedupes_byte_identical_edits() {
		let source = "abcdef";
		let edits = vec![
			Edit::<String> { position: 1, deleted_length: 3, inserted_text: b"x".to_vec() },
			Edit::<String> { position: 1, deleted_length: 3, inserted_text: b"x".to_vec() },
		];
		let output = apply_edits(source, &edits).expect("identical edits should collapse to one");
		assert_eq!(output, "axef");
	}

	#[test]
	fn ast_edit_dedupes_identical_matches_across_rules() {
		let unique = SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.expect("system time should be after UNIX_EPOCH")
			.as_nanos();
		let root = std::env::temp_dir().join(format!("pi-ast-dedupe-{unique}"));
		fs::create_dir_all(&root).expect("temp dedupe dir should be created");
		let tree = TempTree { root };
		let file_path = tree.root.join("a.ts");
		fs::write(&file_path, "const b = foo(bar);\n").expect("temp file a.ts should be written");

		// Both rules match the same call node and produce the byte-identical
		// replacement; the deterministic edit must apply once, not error as an
		// ambiguous overlap.
		let mut rewrites = HashMap::new();
		rewrites.insert("foo($X)".to_string(), "qux($X)".to_string());
		rewrites.insert("foo(bar)".to_string(), "qux(bar)".to_string());

		let result = ast_edit_blocking(
			task::CancelToken::default(),
			Some(rewrites),
			Some("ts".to_string()),
			Some(tree.root.to_string_lossy().into_owned()),
			None,
			None,
			None,
			Some(false),
			None,
			None,
			None,
		)
		.expect("identical duplicate matches should apply cleanly");

		assert_eq!(result.total_replacements, 1, "duplicate match must be counted once");
		assert_eq!(
			fs::read_to_string(&file_path).expect("a.ts should be readable"),
			"const b = qux(bar);\n",
		);
	}

	fn make_apply_failure_tree() -> TempTree {
		let unique = SystemTime::now()
			.duration_since(UNIX_EPOCH)
			.expect("system time should be after UNIX_EPOCH")
			.as_nanos();
		let root = std::env::temp_dir().join(format!("pi-ast-apply-fail-{unique}"));
		fs::create_dir_all(&root).expect("temp apply-fail dir should be created");
		// `a.ts` rewrites cleanly under both rules (one applies, the other doesn't
		// match).
		fs::write(root.join("a.ts"), "const a = bar;\n").expect("temp file a.ts should be written");
		// `b.ts` matches both rules with nested ranges (`foo(bar)` contains `bar`),
		// so `apply_edits` rejects the combined edit set with an overlap error.
		fs::write(root.join("b.ts"), "const b = foo(bar);\n")
			.expect("temp file b.ts should be written");
		TempTree { root }
	}

	#[test]
	fn ast_edit_does_not_partially_write_when_apply_fails() {
		let tree = make_apply_failure_tree();
		let a_path = tree.root.join("a.ts");
		let b_path = tree.root.join("b.ts");
		let a_before = fs::read_to_string(&a_path).expect("a.ts should be readable");
		let b_before = fs::read_to_string(&b_path).expect("b.ts should be readable");

		let mut rewrites = HashMap::new();
		rewrites.insert("bar".to_string(), "baz".to_string());
		rewrites.insert("foo($X)".to_string(), "qux($X)".to_string());

		let result = ast_edit_blocking(
			task::CancelToken::default(),
			Some(rewrites),
			Some("ts".to_string()),
			Some(tree.root.to_string_lossy().into_owned()),
			None,
			None,
			None,
			Some(false),
			None,
			None,
			None,
		);
		assert!(result.is_err(), "expected ast_edit to error on overlapping edits");

		assert_eq!(
			fs::read_to_string(&a_path).expect("a.ts should still be readable"),
			a_before,
			"a.ts must not be written when the apply pass fails on a later file",
		);
		assert_eq!(
			fs::read_to_string(&b_path).expect("b.ts should still be readable"),
			b_before,
			"b.ts must remain unmodified after apply failure",
		);
	}
}
