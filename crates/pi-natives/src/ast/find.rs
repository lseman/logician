//! The `astGrep` operation: search source files with ast-grep patterns.

use std::collections::{BinaryHeap, BTreeSet, HashMap};
use ast_grep_core::{matcher::Pattern, tree_sitter::LanguageExt};
use napi_derive::napi;

use crate::task;
use super::types::*;

/// Search source files with ast-grep patterns; returns a promise resolved on a
/// worker thread.
#[napi]
pub fn ast_grep(options: AstFindOptions<'_>) -> task::Promise<AstFindResult> {
	let AstFindOptions {
		patterns,
		lang,
		path,
		glob,
		selector,
		strictness,
		limit,
		offset,
		include_meta,
		context: _,
		signal,
		timeout_ms,
	} = options;

	let ct = crate::task::CancelToken::new(timeout_ms, signal);
	let normalized_limit = limit.unwrap_or(DEFAULT_FIND_LIMIT).max(1);
	let normalized_offset = offset.unwrap_or(0);

	task::blocking("ast_grep", ct, move |ct| {
		let patterns = normalize_pattern_list(patterns)?;
		let strictness = resolve_strictness(strictness);
		let include_meta = include_meta.unwrap_or(false);
		let lang_str = lang.as_deref().map(str::trim).filter(|v| !v.is_empty());
		let candidates: Vec<_> = collect_candidates(path, glob.as_deref(), &ct)?
			.into_iter()
			.filter(|candidate| is_supported_file(&candidate.absolute_path, lang_str))
			.collect();

		let (resolved_candidates, languages) =
			resolve_candidates_for_find(candidates, lang_str, &ct)?;
		let compiled_patterns =
			compile_find_patterns(&patterns, &languages, selector.as_deref(), &strictness, &ct)?;
		let files_searched = to_u32(resolved_candidates.len());

		let retained_capacity = retained_find_capacity(normalized_offset, normalized_limit);
		let mut retained_matches = BinaryHeap::new();
		let mut parse_errors = Vec::new();
		let mut total_matches = 0u32;
		let mut match_sequence = 0u64;
		let mut files_with_matches = BTreeSet::new();
		for resolved in resolved_candidates {
			ct.heartbeat()?;
			let ResolvedCandidate { candidate, language, language_error } = resolved;

			if let Some(error) = language_error.as_deref() {
				for compiled in &compiled_patterns {
					parse_errors
						.push(format!("{}: {}: {error}", compiled.pattern, candidate.display_path));
				}
				continue;
			}

			let Some(language) = language else {
				continue;
			};
			let lang_key = language.canonical_name();
			let source = match std::fs::read_to_string(&candidate.absolute_path) {
				Ok(source) => source,
				Err(err) => {
					for compiled in &compiled_patterns {
						parse_errors
							.push(format!("{}: {}: {err}", compiled.pattern, candidate.display_path));
					}
					continue;
				},
			};

			let mut runnable_patterns: Vec<(&str, &Pattern)> = Vec::new();
			for compiled in &compiled_patterns {
				ct.heartbeat()?;
				if let Some(error) = compiled.compile_errors_by_lang.get(lang_key) {
					parse_errors
						.push(format!("{}: {}: {error}", compiled.pattern, candidate.display_path));
					continue;
				}
				if let Some(pattern) = compiled.compiled_by_lang.get(lang_key) {
					runnable_patterns.push((compiled.pattern.as_str(), pattern));
				}
			}
			if runnable_patterns.is_empty() {
				continue;
			}

			let ast = language.ast_grep(source);
			if ast.root().dfs().any(|node| node.is_error()) {
				parse_errors.push(format!(
					"{}: parse error (syntax tree contains error nodes)",
					candidate.display_path
				));
			}

			let mut file_had_match = false;
			for (_, pattern) in runnable_patterns {
				ct.heartbeat()?;
				for matched in ast.root().find_all(pattern.clone()) {
					ct.heartbeat()?;
					total_matches = total_matches.saturating_add(1);
					if !file_had_match {
						files_with_matches.insert(candidate.display_path.clone());
						file_had_match = true;
					}
					let range = matched.range();
					let start = matched.start_pos();
					let end = matched.end_pos();
					let key = AstFindOrderKey {
						path:         candidate.display_path.clone(),
						start_line:   to_u32(start.line().saturating_add(1)),
						start_column: to_u32(start.column(matched.get_node()).saturating_add(1)),
						end_line:     to_u32(end.line().saturating_add(1)),
						end_column:   to_u32(end.column(matched.get_node()).saturating_add(1)),
						byte_start:   to_u32(range.start),
						byte_end:     to_u32(range.end),
						sequence:     match_sequence,
					};
					match_sequence = match_sequence.saturating_add(1);
					if should_retain_match(&retained_matches, retained_capacity, &key) {
						let meta_variables = if include_meta {
							Some(HashMap::<String, String>::from(matched.get_env().clone()))
						} else {
							None
						};
						retain_bounded_match(
							&mut retained_matches,
							retained_capacity,
							RetainedAstFindMatch {
								key,
								text: matched.text().into_owned(),
								meta_variables,
							},
						);
					}
				}
			}
		}

		let (matches, limit_reached) =
			page_retained_matches(retained_matches, normalized_offset, normalized_limit);
		let matches = matches
			.into_iter()
			.map(retained_to_find_match)
			.collect::<Vec<_>>();

		Ok(AstFindResult {
			matches,
			total_matches,
			files_with_matches: to_u32(files_with_matches.len()),
			files_searched,
			limit_reached,
			parse_errors: (!parse_errors.is_empty()).then_some(parse_errors),
		})
	})
}
