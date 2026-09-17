//! The `astMatch` operation: match patterns against in-memory source.

use std::collections::{BinaryHeap, HashMap};
use ast_grep_core::tree_sitter::LanguageExt;
use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::task;
use super::types::*;

/// Match ast-grep patterns against an in-memory source string; returns a
/// promise resolved on a worker thread.
///
/// This is the file-free counterpart to [`ast_grep`]: callers that already hold
/// the source (streaming buffers, generated code, editor contents) avoid a
/// temp-file round trip. `lang` is required since there is no path to infer it
/// from.
#[napi]
pub fn ast_match(options: AstMatchOptions<'_>) -> task::Promise<AstMatchResult> {
	let AstMatchOptions {
		source,
		lang,
		patterns,
		selector,
		strictness,
		limit,
		offset,
		include_meta,
		signal,
		timeout_ms,
	} = options;

	let ct = crate::task::CancelToken::new(timeout_ms, signal);
	let normalized_limit = limit.unwrap_or(DEFAULT_FIND_LIMIT).max(1);
	let normalized_offset = offset.unwrap_or(0);

	task::blocking("ast_match", ct, move |ct| {
		let patterns = normalize_pattern_list(Some(patterns))?;
		let strictness = resolve_strictness(strictness);
		let include_meta = include_meta.unwrap_or(false);
		let lang_str = lang.trim();
		if lang_str.is_empty() {
			return Err(Error::from_reason("`lang` is required for ast_match".to_string()));
		}
		let language = resolve_supported_lang(lang_str)?;

		let mut parse_errors = Vec::new();
		let mut compiled_patterns = Vec::with_capacity(patterns.len());
		for pattern in &patterns {
			ct.heartbeat()?;
			match compile_pattern(pattern, selector.as_deref(), &strictness, language) {
				Ok(compiled) => compiled_patterns.push(compiled),
				Err(err) => parse_errors.push(format!("{pattern}: {err}")),
			}
		}

		let retained_capacity = retained_find_capacity(normalized_offset, normalized_limit);
		let mut retained_matches = BinaryHeap::new();
		let mut total_matches = 0u32;
		let mut match_sequence = 0u64;
		if !compiled_patterns.is_empty() {
			let ast = language.ast_grep(&source);
			if ast.root().dfs().any(|node| node.is_error()) {
				parse_errors.push("parse error (syntax tree contains error nodes)".to_string());
			}
			for pattern in &compiled_patterns {
				ct.heartbeat()?;
				for matched in ast.root().find_all(pattern.clone()) {
					ct.heartbeat()?;
					total_matches = total_matches.saturating_add(1);
					let range = matched.range();
					let start = matched.start_pos();
					let end = matched.end_pos();
					let key = AstFindOrderKey {
						path:         String::new(),
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

		Ok(AstMatchResult {
			matches,
			total_matches,
			limit_reached,
			parse_errors: (!parse_errors.is_empty()).then_some(parse_errors),
		})
	})
}
