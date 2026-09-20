//! Policy enums, traversal options, and the high-level walk request builder.

use std::{fmt, path::{Path, PathBuf}};

// parallel_for_each is used via crate::cache::parallel_for_each in for_each_file_candidate_parallel

use crate::entry::{ClosureEntryVisitor, CollectedEntry, FileCandidate, RequestVisitor, WalkBackend, WalkDecision, WalkError, WalkOutcome, WalkRank, WalkStats, WalkStatus};
use crate::filter::WalkFilter;


/// Amount of metadata to collect while reading directories.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WalkDetail {
	/// Collect only the entry name and file kind.
	Minimal,
	/// Also collect mtime and byte size for regular files.
	Full,
}

/// Traversal order for entries within each directory.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WalkOrder {
	/// Visit entries in the order returned by the platform API.
	Unordered,
	/// Sort entries by filename before visiting them.
	Path,
}

/// How directory-open errors are handled during traversal.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DirectoryErrorMode {
	/// Preserve the native glob fast-path contract: silently skip common race or
	/// permission failures and fail on other directory errors.
	SkipSkippable,
	/// Deliver directory errors to [`EntryVisitor::visit_directory_error`] so
	/// GNU-style consumers can report them and continue.
	Visit,
}

/// Symbolic-link traversal policy.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FollowLinks {
	/// Never follow symbolic links.
	Never,
	/// Follow root operands when they are symbolic links, but not descendants.
	Roots,
	/// Follow symbolic links at every depth.
	Always,
}

impl From<bool> for FollowLinks {
	fn from(follow: bool) -> Self {
		if follow { Self::Always } else { Self::Never }
	}
}

impl FollowLinks {
	pub(crate) const fn follow_at_depth(self, depth: usize) -> bool {
		match self {
			Self::Never => false,
			Self::Roots => depth == 0,
			Self::Always => true,
		}
	}
}

/// Shared cache use policy for high-level walk requests.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CachePolicy {
	/// Collect without using or updating the shared scan cache.
	Disabled,
	/// Use the shared scan cache for owned-entry collection.
	Enabled,
}

/// Empty cached-result revalidation policy for [`WalkRequest::collect`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum EmptyRecheck {
	/// Never re-scan an empty cached result.
	Never,
	/// Re-scan empty cached results at or above the configured
	/// [`empty_recheck_ms`] threshold.
	Configured,
	/// Re-scan empty cached results at or above this cache age.
	AfterMillis(u64),
}

/// Size metadata policy for high-level requests.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SizeHintPolicy {
	/// Preserve the request's [`WalkDetail`] setting.
	FromDetail,
	/// Request minimal metadata even on platforms with cheap size hints.
	Never,
	/// Request full metadata only when the platform exposes cheap file sizes.
	WhenCheap,
	/// Request full metadata for every yielded entry.
	Always,
}

/// Directory visit order for high-level requests.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum VisitOrder {
	/// Yield a directory before its children.
	PreOrder,
	/// Yield a directory after its children when supported by the backend.
	ContentsFirst,
}

/// Options shared by native traversal consumers.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WalkOptions {
	/// Include dot-prefixed entries.
	pub include_hidden:    bool,
	/// Honor `.ignore`, `.gitignore`, repository excludes, and global gitignore.
	pub use_gitignore:     bool,
	/// Prune `.git` directories during traversal.
	pub skip_git:          bool,
	/// Prune `node_modules` directories during traversal.
	pub skip_node_modules: bool,
	/// Symbolic-link traversal policy.
	pub follow_links:      FollowLinks,
	/// Metadata detail requested for each yielded entry.
	pub detail:            WalkDetail,
	/// Per-directory visit order.
	pub order:             WalkOrder,
	/// Yield the traversal root as a depth-0 entry before its children.
	pub emit_root:         bool,
	/// Minimum depth yielded to the visitor. Root depth is 0.
	pub min_depth:         usize,
	/// Maximum depth traversed and yielded. Root depth is 0.
	pub max_depth:         usize,
	/// Yield directory entries after their children.
	pub contents_first:    bool,
	/// Directory-open error handling policy.
	pub directory_errors:  DirectoryErrorMode,
	/// Stay on the root filesystem when supported by the platform.
	pub same_file_system:  bool,
	/// Use the shared scan cache when collecting owned entries.
	pub cache:             bool,
}

impl Default for WalkOptions {
	fn default() -> Self {
		Self {
			include_hidden:    true,
			use_gitignore:     false,
			skip_git:          false,
			skip_node_modules: false,
			follow_links:      FollowLinks::Never,
			detail:            WalkDetail::Minimal,
			order:             WalkOrder::Path,
			emit_root:         false,
			min_depth:         1,
			max_depth:         usize::MAX,
			contents_first:    false,
			directory_errors:  DirectoryErrorMode::Visit,
			same_file_system:  false,
			cache:             false,
		}
	}
}

/// High-level traversal request that owns a root and wraps [`WalkOptions`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WalkRequest {
	root:             PathBuf,
	options:          WalkOptions,
	cache_policy:     CachePolicy,
	filter:           WalkFilter,
	limit:            Option<usize>,
	empty_recheck:    EmptyRecheck,
	visit_order:      VisitOrder,
	size_hint_policy: SizeHintPolicy,
}

impl WalkRequest {
	/// Create a request rooted at `root` with default [`WalkOptions`].
	pub fn new(root: impl Into<PathBuf>) -> Self {
		Self::from_options(root, WalkOptions::default())
	}

	/// Create a request from existing low-level options.
	pub fn from_options(root: impl Into<PathBuf>, options: WalkOptions) -> Self {
		let cache_policy = if options.cache {
			CachePolicy::Enabled
		} else {
			CachePolicy::Disabled
		};
		let visit_order = if options.contents_first {
			VisitOrder::ContentsFirst
		} else {
			VisitOrder::PreOrder
		};
		Self {
			root: root.into(),
			options,
			cache_policy,
			filter: WalkFilter::default(),
			limit: None,
			empty_recheck: EmptyRecheck::Configured,
			visit_order,
			size_hint_policy: SizeHintPolicy::FromDetail,
		}
	}

	/// Return the traversal root.
	pub fn root(&self) -> &Path {
		&self.root
	}

	/// Return low-level options after applying high-level policies.
	pub const fn options(&self) -> WalkOptions {
		self.effective_options()
	}

	/// Include or exclude dot-prefixed entries.
	pub const fn hidden(mut self, include_hidden: bool) -> Self {
		self.options.include_hidden = include_hidden;
		self
	}

	/// Enable or disable `.ignore`/gitignore matching.
	pub const fn gitignore(mut self, use_gitignore: bool) -> Self {
		self.options.use_gitignore = use_gitignore;
		self
	}

	/// Enable or disable pruning `.git` directories.
	pub const fn skip_git(mut self, skip_git: bool) -> Self {
		self.options.skip_git = skip_git;
		self
	}

	/// Enable or disable pruning `node_modules` directories during traversal.
	pub const fn skip_node_modules(mut self, skip_node_modules: bool) -> Self {
		self.options.skip_node_modules = skip_node_modules;
		self
	}

	/// Set symbolic-link traversal policy.
	pub const fn follow_links(mut self, follow_links: FollowLinks) -> Self {
		self.options.follow_links = follow_links;
		self
	}

	/// Set metadata detail collected for entries.
	pub const fn detail(mut self, detail: WalkDetail) -> Self {
		self.options.detail = detail;
		self
	}

	/// Set per-directory entry order.
	pub const fn order(mut self, order: WalkOrder) -> Self {
		self.options.order = order;
		self
	}

	/// Enable or disable emitting the root entry.
	pub const fn emit_root(mut self, emit_root: bool) -> Self {
		self.options.emit_root = emit_root;
		self
	}

	/// Set minimum and maximum traversal depth.
	pub const fn depth(mut self, min_depth: usize, max_depth: usize) -> Self {
		self.options.min_depth = min_depth;
		self.options.max_depth = max_depth;
		self
	}

	/// Set directory-open error handling.
	pub const fn directory_errors(mut self, directory_errors: DirectoryErrorMode) -> Self {
		self.options.directory_errors = directory_errors;
		self
	}

	/// Enable or disable staying on the root filesystem.
	pub const fn same_file_system(mut self, same_file_system: bool) -> Self {
		self.options.same_file_system = same_file_system;
		self
	}

	/// Enable or disable the shared scan cache for owned collection.
	pub const fn cache(mut self, cache: bool) -> Self {
		self.cache_policy = if cache {
			CachePolicy::Enabled
		} else {
			CachePolicy::Disabled
		};
		self.options.cache = cache;
		self
	}

	/// Set the static high-level filter.
	pub fn filter(mut self, filter: WalkFilter) -> Self {
		self.filter = filter;
		self
	}

	/// Limit the number of emitted entries after filtering.
	pub const fn limit(mut self, limit: usize) -> Self {
		self.limit = Some(limit);
		self
	}

	/// Remove any high-level entry limit.
	pub const fn no_limit(mut self) -> Self {
		self.limit = None;
		self
	}

	/// Set empty cached-result revalidation policy.
	pub const fn empty_recheck(mut self, empty_recheck: EmptyRecheck) -> Self {
		self.empty_recheck = empty_recheck;
		self
	}

	/// Set high-level directory visit order.
	pub const fn visit_order(mut self, visit_order: VisitOrder) -> Self {
		self.visit_order = visit_order;
		self.options.contents_first = matches!(visit_order, VisitOrder::ContentsFirst);
		self
	}

	/// Set size metadata policy.
	pub const fn size_hints(mut self, size_hint_policy: SizeHintPolicy) -> Self {
		self.size_hint_policy = size_hint_policy;
		self
	}

	/// Collect owned entries, then apply high-level filters, limits, and
	/// empty-cache rechecks.
	pub fn collect(&self) -> std::result::Result<WalkOutcome, WalkError<String>> {
		self.collect_with_heartbeat(|| Ok::<(), std::convert::Infallible>(()))
	}

	/// Collect owned entries with a caller-supplied heartbeat.
	pub fn collect_with_heartbeat<E, H>(
		&self,
		heartbeat: H,
	) -> std::result::Result<WalkOutcome, WalkError<String>>
	where
		H: Fn() -> std::result::Result<(), E> + Sync,
		E: fmt::Display,
	{
		self.collect_with_rank_and_limit(None, self.limit, heartbeat)
	}

	/// Collect owned entries, apply high-level filters, rank, then truncate to
	/// `limit`.
	///
	/// The request's stored [`WalkRequest::limit`] is intentionally not applied
	/// before ranking; `limit` is the top-N bound for this ranked collection.
	pub fn collect_ranked(
		&self,
		rank: WalkRank,
		limit: usize,
	) -> std::result::Result<WalkOutcome, WalkError<String>> {
		self.collect_ranked_with_heartbeat(rank, limit, || Ok::<(), std::convert::Infallible>(()))
	}

	/// Collect owned entries with a caller-supplied heartbeat, apply high-level
	/// filters, rank, then truncate to `limit`.
	///
	/// Ranking happens after all high-level filters and before truncation so
	/// top-N callers observe the best matching entries, not the first traversed
	/// entries.
	pub fn collect_ranked_with_heartbeat<E, H>(
		&self,
		rank: WalkRank,
		limit: usize,
		heartbeat: H,
	) -> std::result::Result<WalkOutcome, WalkError<String>>
	where
		H: Fn() -> std::result::Result<(), E> + Sync,
		E: fmt::Display,
	{
		self.collect_with_rank_and_limit(Some(rank), Some(limit), heartbeat)
	}

	/// Collect regular files accepted by this request.
	pub fn collect_files(&self) -> std::result::Result<Vec<CollectedEntry>, WalkError<String>> {
		self.collect_files_with_heartbeat(|| Ok::<(), std::convert::Infallible>(()))
	}

	/// Collect regular files accepted by this request with a caller-supplied
	/// heartbeat.
	pub fn collect_files_with_heartbeat<E, H>(
		&self,
		heartbeat: H,
	) -> std::result::Result<Vec<CollectedEntry>, WalkError<String>>
	where
		H: Fn() -> std::result::Result<(), E> + Sync,
		E: fmt::Display,
	{
		let outcome = self.collect_with_heartbeat(heartbeat)?;
		Ok(outcome
			.entries
			.into_iter()
			.filter(CollectedEntry::is_file)
			.collect())
	}

	/// Collect directories accepted by this request.
	pub fn collect_dirs(&self) -> std::result::Result<Vec<CollectedEntry>, WalkError<String>> {
		self.collect_dirs_with_heartbeat(|| Ok::<(), std::convert::Infallible>(()))
	}

	/// Collect directories accepted by this request with a caller-supplied
	/// heartbeat.
	pub fn collect_dirs_with_heartbeat<E, H>(
		&self,
		heartbeat: H,
	) -> std::result::Result<Vec<CollectedEntry>, WalkError<String>>
	where
		H: Fn() -> std::result::Result<(), E> + Sync,
		E: fmt::Display,
	{
		let outcome = self.collect_with_heartbeat(heartbeat)?;
		Ok(outcome
			.entries
			.into_iter()
			.filter(CollectedEntry::is_dir)
			.collect())
	}

	/// Collect regular-file candidates accepted by this request.
	pub fn collect_file_candidates(
		&self,
	) -> std::result::Result<Vec<FileCandidate>, WalkError<String>> {
		self.collect_file_candidates_with_heartbeat(|| Ok::<(), std::convert::Infallible>(()))
	}

	/// Collect regular-file candidates accepted by this request with a
	/// caller-supplied heartbeat.
	pub fn collect_file_candidates_with_heartbeat<E, H>(
		&self,
		heartbeat: H,
	) -> std::result::Result<Vec<FileCandidate>, WalkError<String>>
	where
		H: Fn() -> std::result::Result<(), E> + Sync,
		E: fmt::Display,
	{
		Ok(self
			.collect_file_candidates_with_stats_with_heartbeat(heartbeat)?
			.0)
	}

	/// Stream entries through `visitor` after applying high-level filters and
	/// limits.
	pub fn stream<V>(&self, visitor: &mut V) -> std::result::Result<WalkStatus, WalkError<V::Error>>
	where
		V: crate::entry::EntryVisitor,
	{
		self.stream_with_heartbeat(visitor, || Ok::<(), V::Error>(()))
	}

	/// Stream entries through `visitor` with a caller-supplied heartbeat.
	pub fn stream_with_heartbeat<V, H>(
		&self,
		visitor: &mut V,
		heartbeat: H,
	) -> std::result::Result<WalkStatus, WalkError<V::Error>>
	where
		V: crate::entry::EntryVisitor,
		H: FnMut() -> std::result::Result<(), V::Error>,
	{
		self.stream_with_predicate_and_heartbeat(visitor, crate::filter::IncludeAllPredicate, heartbeat)
	}

	/// Stream accepted entries through a closure after applying high-level
	/// filters and limits.
	pub fn for_each_entry<E, V>(&self, visit: V) -> std::result::Result<WalkStatus, WalkError<E>>
	where
		V: for<'entry> FnMut(crate::entry::EntryMeta<'entry>) -> std::result::Result<WalkDecision, E>,
	{
		self.for_each_entry_with_heartbeat(
			|| Ok::<(), E>(()),
			visit,
			|_| Ok::<WalkDecision, E>(WalkDecision::Include),
		)
	}

	/// Stream accepted entries through closures with a caller-supplied
	/// heartbeat.
	pub fn for_each_entry_with_heartbeat<E, H, V, D>(
		&self,
		heartbeat: H,
		visit: V,
		directory_error: D,
	) -> std::result::Result<WalkStatus, WalkError<E>>
	where
		H: FnMut() -> std::result::Result<(), E>,
		V: for<'entry> FnMut(crate::entry::EntryMeta<'entry>) -> std::result::Result<WalkDecision, E>,
		D: for<'error> FnMut(crate::entry::DirectoryError<'error>) -> std::result::Result<WalkDecision, E>,
	{
		let mut visitor = ClosureEntryVisitor { root: &self.root, visit, directory_error };
		self.stream_with_heartbeat(&mut visitor, heartbeat)
	}

	/// Stream entries through `visitor` with an additional dynamic predicate.
	pub fn stream_with_predicate<V, P>(
		&self,
		visitor: &mut V,
		predicate: P,
	) -> std::result::Result<WalkStatus, WalkError<V::Error>>
	where
		V: crate::entry::EntryVisitor,
		P: crate::filter::WalkPredicate,
	{
		self.stream_with_predicate_and_heartbeat(visitor, predicate, || Ok::<(), V::Error>(()))
	}

	/// Stream entries through `visitor` with a dynamic predicate and
	/// caller-supplied heartbeat.
	pub fn stream_with_predicate_and_heartbeat<V, P, H>(
		&self,
		visitor: &mut V,
		predicate: P,
		mut heartbeat: H,
	) -> std::result::Result<WalkStatus, WalkError<V::Error>>
	where
		V: crate::entry::EntryVisitor,
		P: crate::filter::WalkPredicate,
		H: FnMut() -> std::result::Result<(), V::Error>,
	{
		let options = self.effective_options();
		let mut adapter = RequestVisitor {
			root: &self.root,
			filter: &self.filter,
			limit: self.limit,
			emitted: 0,
			visitor,
			predicate,
		};
		crate::walk::walk_entries(&self.root, options, &mut adapter, &mut heartbeat)
	}

	/// Run `operation` for each accepted regular file.
	pub fn for_each_file<E>(
		&self,
		operation: impl Fn(&std::path::Path) -> std::result::Result<(), E> + Send + Sync,
	) -> std::result::Result<WalkStats, WalkError<String>>
	where
		E: fmt::Display + Send,
	{
		self.for_each_file_with_heartbeat(operation, || Ok::<(), std::convert::Infallible>(()))
	}

	/// Run `operation` for each accepted regular file with a caller-supplied
	/// heartbeat.
	pub fn for_each_file_with_heartbeat<E, HE, H>(
		&self,
		operation: impl Fn(&std::path::Path) -> std::result::Result<(), E> + Send + Sync,
		heartbeat: H,
	) -> std::result::Result<WalkStats, WalkError<String>>
	where
		E: fmt::Display + Send,
		H: Fn() -> std::result::Result<(), HE> + Sync,
		HE: fmt::Display,
	{
		self.for_each_file_candidate_with_heartbeat(|candidate| operation(&candidate.path), heartbeat)
	}

	/// Run `operation` for each accepted regular-file candidate.
	pub fn for_each_file_candidate<E>(
		&self,
		operation: impl Fn(&FileCandidate) -> std::result::Result<(), E> + Send + Sync,
	) -> std::result::Result<WalkStats, WalkError<String>>
	where
		E: fmt::Display + Send,
	{
		self.for_each_file_candidate_with_heartbeat(operation, || Ok::<(), std::convert::Infallible>(()))
	}

	/// Run `operation` for each accepted regular-file candidate with a
	/// caller-supplied heartbeat.
	pub fn for_each_file_candidate_with_heartbeat<E, HE, H>(
		&self,
		operation: impl Fn(&FileCandidate) -> std::result::Result<(), E> + Send + Sync,
		heartbeat: H,
	) -> std::result::Result<WalkStats, WalkError<String>>
	where
		E: fmt::Display + Send,
		H: Fn() -> std::result::Result<(), HE> + Sync,
		HE: fmt::Display,
	{
		let (candidates, stats) =
			self.collect_file_candidates_with_stats_with_heartbeat(heartbeat)?;
		execute_candidates(&candidates, |candidate| operation(candidate).map(|_| crate::entry::ParallelWalkControl::Continue))
			.map_err(|err| WalkError::Interrupted(err.to_string()))?;
		Ok(stats)
	}

	/// Visit accepted regular-file candidates using an unordered parallel walk.
	pub fn for_each_file_candidate_parallel<E>(
		&self,
		sink: impl Fn(&FileCandidate) -> std::result::Result<crate::entry::ParallelWalkControl, E> + Send + Sync,
		heartbeat: impl Fn() -> std::result::Result<(), E> + Send + Sync,
	) -> std::result::Result<WalkStatus, WalkError<E>>
	where
		E: Send,
	{
		heartbeat().map_err(|err| WalkError::Interrupted(err))?;
		let candidates = match self.collect_file_candidates() {
			Ok(c) => c,
			Err(WalkError::InvalidData { path, message }) => return Err(WalkError::InvalidData { path, message }),
			Err(WalkError::Interrupted(_)) => unreachable!("collect_file_candidates uses Infallible heartbeat"),
		};
		match crate::cache::parallel_for_each(&candidates, |candidate| {
			sink(candidate)
		}) {
			Ok(ctl) if ctl == crate::entry::ParallelWalkControl::Stop => Ok(WalkStatus::Stopped),
			Ok(_) => Ok(WalkStatus::Complete),
			Err(err) => Err(WalkError::Interrupted(err)),
		}
	}

	fn collect_with_rank_and_limit<E, H>(
		&self,
		rank: Option<WalkRank>,
		limit: Option<usize>,
		heartbeat: H,
	) -> std::result::Result<WalkOutcome, WalkError<String>>
	where
		H: Fn() -> std::result::Result<(), E> + Sync,
		E: fmt::Display,
	{
		let mut options = self.effective_options();
		if matches!(rank, Some(WalkRank::MtimeDescPathAsc)) {
			options.detail = WalkDetail::Full;
		}
		if !options.cache
			&& let (Some(rank), Some(limit)) = (rank, limit)
		{
			let mut collector = crate::entry::RankedCollectVisitor::new(&self.filter, rank, limit);
			crate::walk::walk_entries(&self.root, options, &mut collector, || {
				heartbeat().map_err(|err| err.to_string())
			})?;
			return Ok(collector.into_outcome());
		}
		let mut scan = self.collect_entries_with_options(options, &heartbeat)?;
		let mut backend = if scan.cache_age_ms == 0 {
			WalkBackend::Fresh
		} else {
			WalkBackend::Cached
		};
		let filter_entries = |entries: &mut Vec<CollectedEntry>| {
			let scanned_entries = entries.len();
			entries.retain(|entry| self.filter.accepts_collected(entry));
			(scanned_entries, scanned_entries - entries.len())
		};
		let (mut scanned_entries, mut filtered_entries) = filter_entries(&mut scan.entries);
		if scan.entries.is_empty() && self.should_recheck_empty(scan.cache_age_ms) {
			options.cache = false;
			scan = self.collect_entries_with_options(options, &heartbeat)?;
			backend = WalkBackend::Fresh;
			(scanned_entries, filtered_entries) = filter_entries(&mut scan.entries);
		}
		if let Some(rank) = rank {
			Self::rank_entries(&mut scan.entries, rank);
		}
		let limited_entries = if let Some(limit) = limit {
			let limited_entries = scan.entries.len().saturating_sub(limit);
			scan.entries.truncate(limit);
			limited_entries
		} else {
			0
		};
		let stats = WalkStats {
			cache_age_ms: scan.cache_age_ms,
			scanned_entries,
			filtered_entries,
			limited_entries,
		};
		Ok(WalkOutcome { entries: scan.entries, backend, stats })
	}

	fn rank_entries(entries: &mut [CollectedEntry], rank: WalkRank) {
		match rank {
			WalkRank::PathAsc => entries.sort_by(|left, right| left.path.cmp(&right.path)),
			WalkRank::MtimeDescPathAsc => entries.sort_by(Self::compare_mtime_desc_path_asc),
		}
	}

	pub(crate) fn compare_mtime_desc_path_asc(left: &CollectedEntry, right: &CollectedEntry) -> std::cmp::Ordering {
		let mtime_order = match (left.mtime, right.mtime) {
			(Some(left_mtime), Some(right_mtime)) => right_mtime.total_cmp(&left_mtime),
			(Some(_), None) => std::cmp::Ordering::Less,
			(None, Some(_)) => std::cmp::Ordering::Greater,
			(None, None) => std::cmp::Ordering::Equal,
		};
		mtime_order.then_with(|| left.path.cmp(&right.path))
	}

	const fn effective_options(&self) -> WalkOptions {
		let mut options = self.options;
		options.cache = matches!(self.cache_policy, CachePolicy::Enabled);
		options.contents_first = matches!(self.visit_order, VisitOrder::ContentsFirst);
		match self.size_hint_policy {
			SizeHintPolicy::FromDetail => {},
			SizeHintPolicy::Never => options.detail = WalkDetail::Minimal,
			SizeHintPolicy::WhenCheap => {
				options.detail = if crate::ignore::supports_cheap_size_hints() {
					WalkDetail::Full
				} else {
					WalkDetail::Minimal
				};
			},
			SizeHintPolicy::Always => options.detail = WalkDetail::Full,
		}
		if self.filter.max_file_size.is_some() {
			options.detail = WalkDetail::Full;
		}
		options
	}

	fn collect_entries_with_options<E, H>(
		&self,
		options: WalkOptions,
		heartbeat: &H,
	) -> std::result::Result<crate::entry::CollectedEntries, WalkError<String>>
	where
		H: Fn() -> std::result::Result<(), E> + Sync,
		E: fmt::Display,
	{
		crate::walk::collect_entries(&self.root, options, heartbeat)
	}

	fn should_recheck_empty(&self, cache_age_ms: u64) -> bool {
		if cache_age_ms == 0 {
			return false;
		}
		match self.empty_recheck {
			EmptyRecheck::Never => false,
			EmptyRecheck::Configured => {
				let threshold = crate::cache::empty_recheck_ms();
				threshold > 0 && cache_age_ms >= threshold
			},
			EmptyRecheck::AfterMillis(threshold) => cache_age_ms >= threshold,
		}
	}

	fn collect_file_candidates_with_stats_with_heartbeat<E, H>(
		&self,
		heartbeat: H,
	) -> std::result::Result<(Vec<FileCandidate>, WalkStats), WalkError<String>>
	where
		H: Fn() -> std::result::Result<(), E> + Sync,
		E: fmt::Display,
	{
		let outcome = self.collect_with_heartbeat(heartbeat)?;
		let candidates = outcome
			.entries
			.into_iter()
			.filter(CollectedEntry::is_file)
			.map(|entry| FileCandidate::from_entry(&self.root, entry))
			.collect();
		Ok((candidates, outcome.stats))
	}
}

/// Execute work for regular-file candidates using the centralized walker pool.
pub fn execute_candidates<E>(
	candidates: &[FileCandidate],
	operation: impl Fn(&FileCandidate) -> std::result::Result<crate::entry::ParallelWalkControl, E> + Send + Sync,
) -> std::result::Result<crate::entry::ParallelWalkControl, E>
where
	E: Send,
{
	crate::cache::parallel_for_each(candidates, operation)
}

/// Execute work for regular-file candidates with per-worker state.
pub fn execute_candidates_init<S, E>(
	candidates: &[FileCandidate],
	init: impl Fn() -> S + Send + Sync,
	operation: impl Fn(&mut S, &FileCandidate) -> std::result::Result<crate::entry::ParallelWalkControl, E> + Send + Sync,
) -> std::result::Result<crate::entry::ParallelWalkControl, E>
where
	S: Send,
	E: Send,
{
	crate::cache::parallel_for_each_init(candidates, init, operation)
}
