//! Public types and shared internals for AST operations.
//!
//! All `#[napi(...)]` structs/enums live here so the three operation modules
//! (find, match, edit) can share them without circular dependencies.
//!
//! Shared helper functions are `pub(super)` so each operation module can
//! access them without duplicating code.

use std::{
	cmp::Ordering,
	collections::{BTreeSet, BinaryHeap, HashMap},
	path::{Path, PathBuf},
};

use ast_grep_core::{MatchStrictness, matcher::Pattern, source::Edit};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use pi_ast::{SupportLang, ops as shared_ops};

// ---------------------------------------------------------------------------
// Public enums / structs (#[napi])
// ---------------------------------------------------------------------------

/// ast-grep pattern strictness (controls how patterns match syntax).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[napi(string_enum)]
pub enum AstMatchStrictness {
	/// Match at the concrete syntax tree level.
	#[napi(value = "cst")]
	Cst,
	/// Balanced default suitable for most searches.
	#[napi(value = "smart")]
	Smart,
	/// Match at the AST level.
	#[napi(value = "ast")]
	Ast,
	/// More permissive matching.
	#[napi(value = "relaxed")]
	Relaxed,
	/// Match structural signatures.
	#[napi(value = "signature")]
	Signature,
	/// Template-style pattern matching.
	#[napi(value = "template")]
	Template,
}

impl From<AstMatchStrictness> for MatchStrictness {
	fn from(value: AstMatchStrictness) -> Self {
		match value {
			AstMatchStrictness::Cst => Self::Cst,
			AstMatchStrictness::Smart => Self::Smart,
			AstMatchStrictness::Ast => Self::Ast,
			AstMatchStrictness::Relaxed => Self::Relaxed,
			AstMatchStrictness::Signature => Self::Signature,
			AstMatchStrictness::Template => Self::Template,
		}
	}
}

/// Options for `astGrep`: patterns, scan scope, and match limits.
#[napi(object)]
pub struct AstFindOptions<'env> {
	/// ast-grep patterns to search for (OR across patterns).
	pub patterns:     Option<Vec<String>>,
	/// Language override; otherwise inferred from file extension per candidate.
	pub lang:         Option<String>,
	/// Single file or directory to scan (combined with `glob` when set).
	pub path:         Option<String>,
	/// Optional glob filter relative to the search root.
	pub glob:         Option<String>,
	/// Rule selector for multi-rule ast-grep configurations.
	pub selector:     Option<String>,
	/// Pattern strictness; defaults to smart matching when omitted.
	pub strictness:   Option<AstMatchStrictness>,
	/// Maximum matches to return after `offset` (default applies when omitted).
	pub limit:        Option<u32>,
	/// Number of leading matches to skip before applying `limit`.
	pub offset:       Option<u32>,
	/// When true, include meta-variable bindings per match.
	pub include_meta: Option<bool>,
	/// Reserved for contextual snippets; not used by the current native find
	/// path.
	pub context:      Option<u32>,
	/// Optional cancellation handle (library-specific).
	pub signal:       Option<Unknown<'env>>,
	/// Wall-clock timeout for the worker task in milliseconds.
	pub timeout_ms:   Option<u32>,
}

/// One ast-grep match with source range and optional meta-variables.
#[napi(object)]
pub struct AstFindMatch {
	/// Display path of the matching file.
	pub path:           String,
	/// Matched source text.
	pub text:           String,
	/// Start byte offset in the file (UTF-8 byte index).
	pub byte_start:     u32,
	/// End byte offset in the file (exclusive UTF-8 byte index).
	pub byte_end:       u32,
	/// 1-based start line.
	pub start_line:     u32,
	/// 1-based start column.
	pub start_column:   u32,
	/// 1-based end line.
	pub end_line:       u32,
	/// 1-based end column.
	pub end_column:     u32,
	/// Meta-variable name to captured text, when `includeMeta` was enabled.
	pub meta_variables: Option<HashMap<String, String>>,
}
/// Aggregated search statistics and any parse or compile diagnostics.
#[napi(object)]
pub struct AstFindResult {
	/// Page of matches after sort, offset, and limit.
	pub matches:            Vec<AstFindMatch>,
	/// Total matches found before paging (can exceed `matches.length`).
	pub total_matches:      u32,
	/// Distinct files that contained at least one match.
	pub files_with_matches: u32,
	/// Files examined for the query.
	pub files_searched:     u32,
	/// True when results were truncated by `limit`.
	pub limit_reached:      bool,
	/// Non-fatal parse or pattern errors collected during the run.
	pub parse_errors:       Option<Vec<String>>,
}

/// Options for `astMatch`: run ast-grep patterns against an in-memory source
/// string instead of files on disk.
#[napi(object)]
pub struct AstMatchOptions<'env> {
	/// Source code to match against (parsed in memory, never read from disk).
	pub source:       String,
	/// Language of `source` (required; e.g. "ts", "tsx", "rust", "python").
	pub lang:         String,
	/// ast-grep patterns to search for (OR across patterns).
	pub patterns:     Vec<String>,
	/// Rule selector for multi-rule ast-grep configurations.
	pub selector:     Option<String>,
	/// Pattern strictness; defaults to smart matching when omitted.
	pub strictness:   Option<AstMatchStrictness>,
	/// Maximum matches to return after `offset` (default applies when omitted).
	pub limit:        Option<u32>,
	/// Number of leading matches to skip before applying `limit`.
	pub offset:       Option<u32>,
	/// When true, include meta-variable bindings per match.
	pub include_meta: Option<bool>,
	/// Optional cancellation handle (library-specific).
	pub signal:       Option<Unknown<'env>>,
	/// Wall-clock timeout for the worker task in milliseconds.
	pub timeout_ms:   Option<u32>,
}

/// Result of an in-memory `astMatch` run.
#[napi(object)]
pub struct AstMatchResult {
	/// Page of matches after sort, offset, and limit.
	pub matches:       Vec<AstFindMatch>,
	/// Total matches found before paging (can exceed `matches.length`).
	pub total_matches: u32,
	/// True when results were truncated by `limit`.
	pub limit_reached: bool,
	/// Non-fatal parse or pattern-compile errors collected during the run.
	pub parse_errors:  Option<Vec<String>>,
}

#[derive(Eq, PartialEq)]
pub(super) struct RetainedAstFindMatch {
	pub(super) key:            AstFindOrderKey,
	pub(super) text:           String,
	pub(super) meta_variables: Option<HashMap<String, String>>,
}

impl Ord for RetainedAstFindMatch {
	fn cmp(&self, other: &Self) -> Ordering {
		self.key.cmp(&other.key)
	}
}

impl PartialOrd for RetainedAstFindMatch {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
		Some(self.cmp(other))
	}
}

/// Options for `astEdit`: rewrite rules, scan scope, safety limits, and
/// dry-run.
#[napi(object)]
pub struct AstReplaceOptions<'env> {
	/// Map of pattern string to replacement template.
	pub rewrites:            Option<HashMap<String, String>>,
	/// Language override applied to every file; otherwise inferred per file, so
	/// mixed-language paths rewrite each file in its own language.
	pub lang:                Option<String>,
	/// Single file or directory to rewrite.
	pub path:                Option<String>,
	/// Optional glob filter within the search root.
	pub glob:                Option<String>,
	/// Rule selector for multi-rule configurations.
	pub selector:            Option<String>,
	/// Pattern strictness for rewrites.
	pub strictness:          Option<AstMatchStrictness>,
	/// When true (default), compute changes without writing files.
	pub dry_run:             Option<bool>,
	/// Cap on replacement applications across all files.
	pub max_replacements:    Option<u32>,
	/// Cap on distinct files that may be modified.
	pub max_files:           Option<u32>,
	/// Fail the operation when a file cannot be parsed for rewriting.
	pub fail_on_parse_error: Option<bool>,
	/// Optional cancellation handle.
	pub signal:              Option<Unknown<'env>>,
	/// Wall-clock timeout for the worker task in milliseconds.
	pub timeout_ms:          Option<u32>,
}

/// One textual replacement applied to a file (before/after slice and
/// coordinates).
#[napi(object)]
pub struct AstReplaceChange {
	/// File path for this change.
	pub path:           String,
	/// Original matched text.
	pub before:         String,
	/// Replacement text.
	pub after:          String,
	/// Start byte offset of the replaced span.
	pub byte_start:     u32,
	/// End byte offset of the replaced span (exclusive).
	pub byte_end:       u32,
	/// Length of deleted text in bytes (may differ from `byteEnd - byteStart`
	/// for edge cases).
	pub deleted_length: u32,
	/// 1-based start line of the match.
	pub start_line:     u32,
	/// 1-based start column.
	pub start_column:   u32,
	/// 1-based end line.
	pub end_line:       u32,
	/// 1-based end column.
	pub end_column:     u32,
}

/// Per-file replacement count after an `astEdit` run.
#[napi(object)]
pub struct AstReplaceFileChange {
	/// File that had replacements.
	pub path:  String,
	/// Number of replacements in that file.
	pub count: u32,
}

/// Summary of an ast-grep rewrite pass, including whether disk writes occurred.
#[napi(object)]
pub struct AstReplaceResult {
	/// Individual replacement records (may be large).
	pub changes:            Vec<AstReplaceChange>,
	/// Replacement counts grouped by file.
	pub file_changes:       Vec<AstReplaceFileChange>,
	/// Total replacements applied or previewed.
	pub total_replacements: u32,
	/// Files that had at least one replacement.
	pub files_touched:      u32,
	/// Files considered for rewriting.
	pub files_searched:     u32,
	/// False when `dryRun` prevented writing.
	pub applied:            bool,
	/// True when limits stopped further replacements.
	pub limit_reached:      bool,
	/// Parse or pattern errors when not failing the whole operation.
	pub parse_errors:       Option<Vec<String>>,
}
// ---------------------------------------------------------------------------
// Shared internal types & utilities (pub(super) for sibling modules)
// ---------------------------------------------------------------------------

pub(super) const DEFAULT_FIND_LIMIT: u32 = 50;

/// A file discovered by candidate collection.
pub(super) struct FileCandidate {
	pub(super) absolute_path: PathBuf,
	pub(super) display_path:  String,
}

/// One change pending write in a file.
pub(super) struct PendingFileChange {
	pub(super) change: AstReplaceChange,
	pub(super) edit:   Edit<String>,
}

/// One file write to flush after the apply phase.
pub(super) struct PendingWrite {
	pub(super) path: PathBuf,
	pub(super) output: String,
}

pub(super) fn to_u32(value: usize) -> u32 {
	value.min(u32::MAX as usize) as u32
}
pub(super) fn resolve_supported_lang(value: &str) -> Result<SupportLang> {
	shared_ops::resolve_supported_lang(value).map_err(|err| Error::from_reason(err.to_string()))
}
pub(super) fn resolve_language(lang: Option<&str>, file_path: &Path) -> Result<SupportLang> {
	shared_ops::resolve_language(lang, file_path).map_err(|err| Error::from_reason(err.to_string()))
}
pub(super) fn is_supported_file(file_path: &Path, explicit_lang: Option<&str>) -> bool {
	shared_ops::is_supported_file(file_path, explicit_lang)
}

/// Resolve ast-grep strictness from an optional user-facing enum.
pub(super) fn resolve_strictness(value: Option<AstMatchStrictness>) -> MatchStrictness {
	value.map_or(MatchStrictness::Smart, Into::into)
}

/// Normalize and deduplicate a pattern list; errors if empty.
pub(super) fn normalize_pattern_list(patterns: Option<Vec<String>>) -> Result<Vec<String>> {
	let mut normalized = Vec::new();
	let mut seen = BTreeSet::new();
	for raw in patterns.unwrap_or_default() {
		let pattern = raw.trim();
		if pattern.is_empty() || seen.contains(pattern) {
			continue;
		}
		let owned = if pattern.len() == raw.len() { raw } else { pattern.to_string() };
		seen.insert(owned.clone());
		normalized.push(owned);
	}
	if normalized.is_empty() {
		return Err(Error::from_reason(
			"`patterns` is required and must include at least one non-empty pattern".to_string(),
		));
	}
	Ok(normalized)
}

/// Normalize a single search path to an absolute canonical path.
pub(super) fn normalize_search_path(path: Option<String>) -> Result<PathBuf> {
	let raw = path.unwrap_or_else(|| ".".to_string());
	let candidate = PathBuf::from(raw.trim());
	let absolute = if candidate.is_absolute() {
		candidate
	} else {
		std::env::current_dir()
			.map_err(|err| Error::from_reason(format!("Failed to resolve cwd: {err}")))?
			.join(candidate)
	};
	Ok(std::fs::canonicalize(&absolute).unwrap_or(absolute))
}

/// Collect files matching the search scope, respecting gitignore and the
/// optional glob filter.
pub(super) fn collect_candidates(
	path: Option<String>,
	glob: Option<&str>,
	ct: &crate::task::CancelToken,
) -> Result<Vec<FileCandidate>> {
	let search_path = normalize_search_path(path)?;
	let metadata = std::fs::metadata(&search_path)
		.map_err(|err| Error::from_reason(format!("Path not found: {err}")))?;
	if metadata.is_file() {
		let display_path = search_path
			.file_name()
			.and_then(|name| name.to_str())
			.map_or_else(
				|| search_path.to_string_lossy().into_owned(),
				std::string::ToString::to_string,
			);
		return Ok(vec![FileCandidate { absolute_path: search_path, display_path }]);
	}
	if !metadata.is_dir() {
		return Err(Error::from_reason(format!(
			"Search path must be a file or directory: {}",
			search_path.display()
		)));
	}

	let mentions_node_modules = glob.is_some_and(|value| value.contains("node_modules"));
	let mut filter =
		pi_walker::WalkFilter::files_only().node_modules_unless_mentioned(mentions_node_modules);
	if let Some(glob) = glob.map(str::trim).filter(|value| !value.is_empty()) {
		let pattern = crate::glob_util::build_glob_pattern(glob, false);
		let compiled = pi_walker::CompiledWalkGlob::new([pattern])
			.map_err(|err| Error::from_reason(format!("Invalid glob pattern: {err}")))?;
		filter = filter.glob(compiled);
	}
	let request = pi_walker::WalkRequest::new(&search_path)
		.hidden(true)
		.gitignore(true)
		.skip_git(true)
		.follow_links(pi_walker::FollowLinks::Never)
		.detail(pi_walker::WalkDetail::Minimal)
		.order(pi_walker::WalkOrder::Path)
		.emit_root(false)
		.depth(1, usize::MAX)
		.directory_errors(pi_walker::DirectoryErrorMode::SkipSkippable)
		.cache(true)
		.empty_recheck(pi_walker::EmptyRecheck::Configured)
		.filter(filter);
	let mut files: Vec<_> = request
		.collect_files_with_heartbeat(|| ct.heartbeat())
		.map_err(crate::iofs::map_walker_error)?
		.into_iter()
		.map(|entry| FileCandidate {
			absolute_path: entry.absolute_path(&search_path),
			display_path:  entry.path,
		})
		.collect();

	files.sort_by(|a, b| a.display_path.cmp(&b.display_path));
	Ok(files)
}

/// Compile a single ast-grep pattern for a given language.
pub(super) fn compile_pattern(
	pattern: &str,
	selector: Option<&str>,
	strictness: &MatchStrictness,
	lang: SupportLang,
) -> Result<Pattern> {
	shared_ops::compile_pattern(pattern, selector, strictness, lang)
		.map_err(|err| Error::from_reason(err.to_string()))
}

/// Apply edits to source text.
pub(super) fn apply_edits(content: &str, edits: &[Edit<String>]) -> Result<String> {
	shared_ops::apply_edits(content, edits).map_err(|err| Error::from_reason(err.to_string()))
}

/// Normalize a rewrite map; errors if empty or has empty keys.
pub(super) fn normalize_rewrite_map(
	rewrites: Option<HashMap<String, String>>,
) -> Result<Vec<(String, String)>> {
	let mut normalized = Vec::new();
	for (pattern, rewrite) in rewrites.unwrap_or_default() {
		if pattern.is_empty() {
			return Err(Error::from_reason(
				"`rewrites` keys must be non-empty pattern strings".to_string(),
			));
		}
		normalized.push((pattern, rewrite));
	}
	if normalized.is_empty() {
		return Err(Error::from_reason(
			"`rewrites` is required and must include at least one pattern->rewrite mapping"
				.to_string(),
		));
	}
	normalized.sort_by(|left, right| left.0.cmp(&right.0));
	Ok(normalized)
}

// ---------------------------------------------------------------------------
// Find-specific shared helpers (pub(super))
// ---------------------------------------------------------------------------

/// Ordered key for deduplicating matches across a single file.
#[derive(Clone, Eq, PartialEq)]
pub(super) struct AstFindOrderKey {
	pub(super) path:         String,
	pub(super) start_line:   u32,
	pub(super) start_column: u32,
	pub(super) end_line:     u32,
	pub(super) end_column:   u32,
	pub(super) byte_start:   u32,
	pub(super) byte_end:     u32,
	pub(super) sequence:     u64,
}

impl Ord for AstFindOrderKey {
	fn cmp(&self, other: &Self) -> Ordering {
		self.path
			.cmp(&other.path)
			.then(self.start_line.cmp(&other.start_line))
			.then(self.start_column.cmp(&other.start_column))
			.then(self.end_line.cmp(&other.end_line))
			.then(self.end_column.cmp(&other.end_column))
			.then(self.byte_start.cmp(&other.byte_start))
			.then(self.byte_end.cmp(&other.byte_end))
			.then(self.sequence.cmp(&other.sequence))
	}
}

impl PartialOrd for AstFindOrderKey {
	fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
		Some(self.cmp(other))
	}
}

/// Bounded-capacity helpers for the retained-match heap.
pub(super) fn retained_find_capacity(offset: u32, limit: u32) -> usize {
	usize::try_from(offset.saturating_add(limit).saturating_add(1)).unwrap_or(usize::MAX)
}

pub(super) fn should_retain_match(
	retained: &BinaryHeap<RetainedAstFindMatch>,
	capacity: usize,
	key: &AstFindOrderKey,
) -> bool {
	if retained.len() < capacity {
		return true;
	}
	retained
		.peek()
		.is_some_and(|worst_retained| key.cmp(&worst_retained.key).is_lt())
}

pub(super) fn retain_bounded_match(
	retained: &mut BinaryHeap<RetainedAstFindMatch>,
	capacity: usize,
	candidate: RetainedAstFindMatch,
) {
	if retained.len() < capacity {
		retained.push(candidate);
		return;
	}
	if let Some(mut worst_retained) = retained.peek_mut()
		&& candidate.key.cmp(&worst_retained.key).is_lt()
	{
		*worst_retained = candidate;
	}
}

pub(super) fn page_retained_matches(
	retained: BinaryHeap<RetainedAstFindMatch>,
	offset: u32,
	limit: u32,
) -> (Vec<RetainedAstFindMatch>, bool) {
	let mut retained_matches = retained.into_vec();
	retained_matches.sort_by(|left, right| left.key.cmp(&right.key));
	let offset = usize::try_from(offset).unwrap_or(usize::MAX);
	let limit = usize::try_from(limit).unwrap_or(usize::MAX);
	let limit_reached = retained_matches.len().saturating_sub(offset) > limit;
	let matches = retained_matches
		.into_iter()
		.skip(offset)
		.take(limit)
		.collect::<Vec<_>>();
	(matches, limit_reached)
}

/// Convert a retained match back to the public struct.
pub(super) fn retained_to_find_match(retained: RetainedAstFindMatch) -> AstFindMatch {
	let RetainedAstFindMatch { key, text, meta_variables } = retained;
	AstFindMatch {
		path: key.path,
		text,
		byte_start: key.byte_start,
		byte_end: key.byte_end,
		start_line: key.start_line,
		start_column: key.start_column,
		end_line: key.end_line,
		end_column: key.end_column,
		meta_variables,
	}
}

/// One pattern compiled for each language that appeared among candidates.
pub(super) struct CompiledFindPattern {
	pub(super) pattern:                String,
	pub(super) compiled_by_lang:       HashMap<String, Pattern>,
	pub(super) compile_errors_by_lang: HashMap<String, String>,
}

/// Candidate with its resolved (or failed) language.
pub(super) struct ResolvedCandidate {
	pub(super) candidate:      FileCandidate,
	pub(super) language:       Option<SupportLang>,
	pub(super) language_error: Option<String>,
}

pub(super) fn resolve_candidates_for_find(
	candidates: Vec<FileCandidate>,
	lang: Option<&str>,
	ct: &crate::task::CancelToken,
) -> Result<(Vec<ResolvedCandidate>, HashMap<String, SupportLang>)> {
	let mut resolved = Vec::with_capacity(candidates.len());
	let mut languages = HashMap::new();

	for candidate in candidates {
		ct.heartbeat()?;
		match resolve_language(lang, &candidate.absolute_path) {
			Ok(language) => {
				let key = language.canonical_name().to_string();
				languages.entry(key).or_insert(language);
				resolved.push(ResolvedCandidate {
					candidate,
					language: Some(language),
					language_error: None,
				});
			},
			Err(err) => {
				resolved.push(ResolvedCandidate {
					candidate,
					language: None,
					language_error: Some(err.to_string()),
				});
			},
		}
	}

	Ok((resolved, languages))
}

pub(super) fn compile_find_patterns(
	patterns: &[String],
	languages: &HashMap<String, SupportLang>,
	selector: Option<&str>,
	strictness: &MatchStrictness,
	ct: &crate::task::CancelToken,
) -> Result<Vec<CompiledFindPattern>> {
	let mut compiled = Vec::with_capacity(patterns.len());

	for pattern in patterns {
		ct.heartbeat()?;
		let mut compiled_by_lang = HashMap::with_capacity(languages.len());
		let mut compile_errors_by_lang = HashMap::new();

		for (lang_key, &language) in languages {
			ct.heartbeat()?;
			match compile_pattern(pattern, selector, strictness, language) {
				Ok(compiled_pattern) => {
					compiled_by_lang.insert(lang_key.clone(), compiled_pattern);
				},
				Err(err) => {
					compile_errors_by_lang.insert(lang_key.clone(), err.to_string());
				},
			}
		}

		compiled.push(CompiledFindPattern {
			pattern: pattern.clone(),
			compiled_by_lang,
			compile_errors_by_lang,
		});
	}

	Ok(compiled)
}
/// A rewrite rule compiled for every language discovered among the candidate
/// files. A rule that fails to parse in one language of a mixed tree skips
/// that language's files (reported as parse errors) instead of failing the
/// whole call.
pub(super) struct CompiledRewriteRule {
	pub(super) pattern:                String,
	pub(super) rewrite:                String,
	pub(super) compiled_by_lang:       HashMap<String, Pattern>,
	pub(super) compile_errors_by_lang: HashMap<String, String>,
}
