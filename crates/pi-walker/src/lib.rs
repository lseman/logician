//! Reusable platform directory traversal primitives.
//!
//! # Overview
//! `pi-walker` owns the native directory-read fast path that higher-level tools
//! use for globbing, grep candidate discovery, AST scans, and shell builtins.
//! The crate exposes plain Rust types, visitor interfaces, cache policy, and a
//! caller-supplied heartbeat so consumers do not inherit N-API dependencies.

mod cache;
mod entry;
mod filter;
mod ignore;
mod policy;
mod walk;

pub use cache::{
	cache_ttl_ms, classify_file_type, contains_component, empty_recheck_ms, invalidate_all,
	invalidate_path, invalidate_path_string, max_cache_entries, normalize_relative_path,
	parallel_for_each, parallel_for_each_init, resolve_search_path, should_parallelize,
	should_skip_path, walk_workers,
};
pub use entry::{
	CollectedEntry, CollectedEntries, CompiledWalkGlob, DirectoryError,
	Entry, EntryMeta, EntryStat, EntryVisitor, FileCandidate, FileType,
	ParallelWalkControl, PreDescendDecision,
	WalkBackend, WalkControl, WalkDecision, WalkError, WalkFilter, WalkOutcome, WalkPredicate, WalkRank,
	WalkStatus, WalkStats,
	// Path helpers
	compare_depth_first_paths, is_path_on_root_file_system, is_relative_ancestor,
	is_under_pruned_relative_dir, root_device_id, sort_collected_depth_first,
};
pub use policy::{
	DirectoryErrorMode, EmptyRecheck, FollowLinks, SizeHintPolicy, VisitOrder,
	WalkDetail, WalkOptions, WalkOrder, WalkRequest,
	// Public execution helpers
	execute_candidates, execute_candidates_init,
};
pub use ignore::supports_cheap_size_hints;
pub use walk::{collect_entries, walk_entries};
