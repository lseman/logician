//! The `astEdit` operation: structural search-and-rewrite across files.

use std::collections::{BTreeMap, HashMap};
use ast_grep_core::{matcher::Pattern, source::Edit, tree_sitter::LanguageExt};
use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::task;
use super::types::*;

/// Apply ast-grep rewrite rules to matching files; honors `dryRun` and returns
/// a result.
pub fn ast_edit_blocking(
	ct: crate::task::CancelToken,
	rewrites: Option<HashMap<String, String>>,
	lang: Option<String>,
	path: Option<String>,
	glob: Option<String>,
	selector: Option<String>,
	strictness: Option<AstMatchStrictness>,
	dry_run: Option<bool>,
	max_replacements: Option<u32>,
	max_files: Option<u32>,
	fail_on_parse_error: Option<bool>,
) -> Result<AstReplaceResult> {
	let rewrite_rules = normalize_rewrite_map(rewrites)?;
	let strictness = resolve_strictness(strictness);
	let dry_run = dry_run.unwrap_or(true);
	let max_replacements = max_replacements.unwrap_or(u32::MAX).max(1);
	let max_files = max_files.unwrap_or(u32::MAX).max(1);
	let fail_on_parse_error = fail_on_parse_error.unwrap_or(false);

	let lang_str = lang.as_deref().map(str::trim).filter(|v| !v.is_empty());
	let candidates: Vec<_> = collect_candidates(path, glob.as_deref(), &ct)?
		.into_iter()
		.filter(|candidate| is_supported_file(&candidate.absolute_path, lang_str))
		.collect();
	if let Some(explicit) = lang_str {
		resolve_supported_lang(explicit)?;
	} else if candidates.is_empty() {
		return Err(Error::from_reason(
			"ast_edit found no supported source files for the given path/glob".to_string(),
		));
	}

	let (resolved_candidates, languages) = resolve_candidates_for_find(candidates, lang_str, &ct)?;
	let files_searched = to_u32(resolved_candidates.len());

	let mut parse_errors = Vec::new();
	let mut compiled_rules: Vec<CompiledRewriteRule> = Vec::with_capacity(rewrite_rules.len());
	for (pattern, rewrite) in rewrite_rules {
		ct.heartbeat()?;
		let mut compiled_by_lang = HashMap::with_capacity(languages.len());
		let mut compile_errors_by_lang = HashMap::new();
		for (lang_key, &language) in &languages {
			ct.heartbeat()?;
			match compile_pattern(&pattern, selector.as_deref(), &strictness, language) {
				Ok(compiled) => {
					compiled_by_lang.insert(lang_key.clone(), compiled);
				},
				Err(err) => {
					compile_errors_by_lang.insert(lang_key.clone(), err.to_string());
				},
			}
		}
		// A pattern that parses in NO discovered language is a genuine pattern
		// error; failing in only some languages of a mixed tree is expected (the
		// pattern targets one language) and surfaces per skipped file below.
		if compiled_by_lang.is_empty() && !languages.is_empty() {
			let mut entries: Vec<_> = compile_errors_by_lang.iter().collect();
			entries.sort_by_key(|(lang_key, _)| lang_key.as_str());
			if fail_on_parse_error {
				let (_, err) = entries
					.first()
					.expect("compile failure recorded for every language");
				return Err(Error::from_reason(format!("{pattern}: {err}")));
			}
			for (lang_key, err) in entries {
				parse_errors.push(if languages.len() > 1 {
					format!("{pattern} ({lang_key}): {err}")
				} else {
					format!("{pattern}: {err}")
				});
			}
			continue;
		}
		compiled_rules.push(CompiledRewriteRule {
			pattern,
			rewrite,
			compiled_by_lang,
			compile_errors_by_lang,
		});
	}
	if compiled_rules.is_empty() {
		return Ok(AstReplaceResult {
			file_changes: vec![],
			total_replacements: 0,
			files_touched: 0,
			files_searched,
			applied: !dry_run,
			limit_reached: false,
			parse_errors: (!parse_errors.is_empty()).then_some(parse_errors),
			changes: vec![],
		});
	}

	let mut changes = Vec::new();
	let mut file_counts: BTreeMap<String, u32> = BTreeMap::new();
	let mut files_touched = 0u32;
	let mut limit_reached = false;
	// Stage writes in memory so a later compute error cannot leave earlier
	// files partially modified on disk; flush only after the whole pass succeeds.
	let mut pending_writes: Vec<PendingWrite> = Vec::new();

	for resolved in &resolved_candidates {
		ct.heartbeat()?;
		let ResolvedCandidate { candidate, language, language_error } = resolved;
		if let Some(error) = language_error.as_deref() {
			if fail_on_parse_error {
				return Err(Error::from_reason(format!("{}: {error}", candidate.display_path)));
			}
			parse_errors.push(format!("{}: {error}", candidate.display_path));
			continue;
		}
		let Some(language) = *language else {
			continue;
		};
		let lang_key = language.canonical_name();

		let mut runnable_rules: Vec<(&str, &Pattern)> = Vec::new();
		for rule in &compiled_rules {
			ct.heartbeat()?;
			if let Some(error) = rule.compile_errors_by_lang.get(lang_key) {
				parse_errors.push(format!("{}: {}: {error}", rule.pattern, candidate.display_path));
				continue;
			}
			if let Some(compiled) = rule.compiled_by_lang.get(lang_key) {
				runnable_rules.push((rule.rewrite.as_str(), compiled));
			}
		}
		if runnable_rules.is_empty() {
			continue;
		}

		let source = match std::fs::read_to_string(&candidate.absolute_path) {
			Ok(source) => source,
			Err(err) => {
				if fail_on_parse_error {
					return Err(Error::from_reason(format!("{}: {err}", candidate.display_path)));
				}
				parse_errors.push(format!("{}: {err}", candidate.display_path));
				continue;
			},
		};

		let ast = language.ast_grep(&source);
		if ast.root().dfs().any(|node| node.is_error()) {
			let parse_issue =
				format!("{}: parse error (syntax tree contains error nodes)", candidate.display_path);
			if fail_on_parse_error {
				return Err(Error::from_reason(parse_issue));
			}
			parse_errors.push(parse_issue);
			continue;
		}

		let mut file_changes = Vec::new();
		let mut reached_max_replacements = false;
		'patterns: for &(rewrite, compiled) in &runnable_rules {
			for matched in ast.root().find_all(compiled.clone()) {
				ct.heartbeat()?;
				let edit = matched.replace_by(rewrite);
				// Multiple rules matching the same node with the same output are one
				// deterministic edit; list and count it once instead of staging a
				// duplicate that trips the apply-time overlap check.
				let duplicate = file_changes.iter().any(|entry: &PendingFileChange| {
					entry.edit.position == edit.position
						&& entry.edit.deleted_length == edit.deleted_length
						&& entry.edit.inserted_text == edit.inserted_text
				});
				if duplicate {
					continue;
				}
				if changes.len() + file_changes.len() >= max_replacements as usize {
					limit_reached = true;
					reached_max_replacements = true;
					break 'patterns;
				}
				let range = matched.range();
				let start = matched.start_pos();
				let end = matched.end_pos();
				let after = String::from_utf8(edit.inserted_text.clone()).map_err(|err| {
					Error::from_reason(format!(
						"{}: replacement text is not valid UTF-8: {err}",
						candidate.display_path
					))
				})?;
				file_changes.push(PendingFileChange {
					change: AstReplaceChange {
						path: candidate.display_path.clone(),
						before: matched.text().into_owned(),
						after,
						byte_start: to_u32(range.start),
						byte_end: to_u32(range.end),
						deleted_length: to_u32(edit.deleted_length),
						start_line: to_u32(start.line().saturating_add(1)),
						start_column: to_u32(start.column(matched.get_node()).saturating_add(1)),
						end_line: to_u32(end.line().saturating_add(1)),
						end_column: to_u32(end.column(matched.get_node()).saturating_add(1)),
					},
					edit,
				});
			}
		}

		if file_changes.is_empty() {
			if reached_max_replacements {
				break;
			}
			continue;
		}
		if files_touched >= max_files {
			limit_reached = true;
			break;
		}
		files_touched = files_touched.saturating_add(1);
		file_counts.insert(candidate.display_path.clone(), to_u32(file_changes.len()));

		if !dry_run {
			let edits: Vec<Edit<String>> = file_changes
				.iter()
				.map(|entry| Edit {
					position:       entry.edit.position,
					deleted_length: entry.edit.deleted_length,
					inserted_text:  entry.edit.inserted_text.clone(),
				})
				.collect();
			let output = apply_edits(&source, &edits)?;
			if output != source {
				pending_writes
					.push(PendingWrite { path: candidate.absolute_path.clone(), output });
			}
		}

		changes.extend(file_changes.into_iter().map(|entry| entry.change));
		if reached_max_replacements {
			break;
		}
	}

	if !dry_run {
		for write in &pending_writes {
			ct.heartbeat()?;
			std::fs::write(&write.path, &write.output).map_err(|err| {
				Error::from_reason(format!("Failed to write {}: {err}", write.path.display()))
			})?;
		}
	}

	let file_changes = file_counts
		.into_iter()
		.map(|(path, count)| AstReplaceFileChange { path, count })
		.collect::<Vec<_>>();

	Ok(AstReplaceResult {
		file_changes,
		total_replacements: to_u32(changes.len()),
		files_touched,
		files_searched,
		applied: !dry_run,
		limit_reached,
		parse_errors: (!parse_errors.is_empty()).then_some(parse_errors),
		changes,
	})
}

/// Apply ast-grep rewrite rules to matching files; honors `dryRun` and returns
/// a promise.
#[napi]
pub fn ast_edit(options: AstReplaceOptions<'_>) -> task::Promise<AstReplaceResult> {
	let AstReplaceOptions {
		rewrites,
		lang,
		path,
		glob,
		selector,
		strictness,
		dry_run,
		max_replacements,
		max_files,
		fail_on_parse_error,
		signal,
		timeout_ms,
	} = options;

	let ct = crate::task::CancelToken::new(timeout_ms, signal);
	task::blocking("ast_edit", ct, move |ct| {
		ast_edit_blocking(
			ct,
			rewrites,
			lang,
			path,
			glob,
			selector,
			strictness,
			dry_run,
			max_replacements,
			max_files,
			fail_on_parse_error,
		)
	})
}
