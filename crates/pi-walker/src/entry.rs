//! Owned and borrowed entry types, visitor trait, filter logic, and platform helpers.
//!

#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::{
	borrow::Cow, fmt, hash::{Hash, Hasher}, io,
	path::{Path, PathBuf},
	sync::Arc,
};
use globset::{GlobBuilder, GlobSetBuilder};


// ── Core types (moved from policy.rs and filter.rs) ─────────────────

/// Filesystem entry kind reported by the walker.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FileType {
	/// Regular file.
	File,
	/// Directory.
	Dir,
	/// Symbolic link.
	Symlink,
}

/// Traversal decision returned by [`WalkPredicate`] and closure streaming APIs.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WalkDecision {
	/// Emit this entry and continue traversal.
	Include,
	/// Do not emit this entry, but continue traversal.
	Skip,
	/// Do not emit this directory and do not descend into it.
	SkipDescend,
	/// Stop traversal immediately.
	Stop,
}

/// Concrete compiled glob filter for normalized walk-relative paths.
///
/// Patterns use `/` separators compiled with [`GlobBuilder::literal_separator`].
#[derive(Clone)]
pub struct CompiledWalkGlob {
	patterns: Arc<[String]>,
	matcher:  Arc<globset::GlobSet>,
}

impl CompiledWalkGlob {
	pub fn new<P, I>(patterns: I) -> Result<Self, globset::Error>
	where
		P: Into<String>,
		I: IntoIterator<Item = P>,
	{
		let mut normalized_patterns = Vec::new();
		let mut builder = GlobSetBuilder::new();
		for pattern in patterns {
			let pattern = pattern.into();
			let glob = GlobBuilder::new(&pattern).literal_separator(true).build()?;
			builder.add(glob);
			normalized_patterns.push(pattern);
		}
		Ok(Self { patterns: normalized_patterns.into(), matcher: Arc::new(builder.build()?) })
	}

	pub fn is_match(&self, relative: &str) -> bool {
		self.matcher.is_match(relative)
	}

	pub fn patterns(&self) -> &[String] {
		&self.patterns
	}
}

impl fmt::Debug for CompiledWalkGlob {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("CompiledWalkGlob")
			.field("patterns", &self.patterns)
			.finish()
	}
}

impl PartialEq for CompiledWalkGlob {
	fn eq(&self, other: &Self) -> bool {
		self.patterns == other.patterns
	}
}

impl Eq for CompiledWalkGlob {}

impl Hash for CompiledWalkGlob {
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.patterns.hash(state);
	}
}

/// High-level entry filter applied by collection and streaming APIs.
#[derive(Clone)]
pub struct WalkFilter {
	kind: WalkFilterKind,
	pub(crate) max_file_size: Option<u64>,
	skip_node_modules_unless_seen: bool,
	mentions_node_modules: bool,
	glob: Option<CompiledWalkGlob>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum WalkFilterKind {
	All,
	Files,
	Dirs,
}

impl Default for WalkFilter {
	fn default() -> Self {
		Self::all()
	}
}

impl fmt::Debug for WalkFilter {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("WalkFilter")
			.field("kind", &self.kind)
			.field("max_file_size", &self.max_file_size)
			.field("skip_node_modules_unless_seen", &self.skip_node_modules_unless_seen)
			.field("mentions_node_modules", &self.mentions_node_modules)
			.field("glob", &self.glob)
			.finish()
	}
}

impl PartialEq for WalkFilter {
	fn eq(&self, other: &Self) -> bool {
		self.kind == other.kind
			&& self.max_file_size == other.max_file_size
			&& self.skip_node_modules_unless_seen == other.skip_node_modules_unless_seen
			&& self.mentions_node_modules == other.mentions_node_modules
			&& self.glob == other.glob
	}
}

impl Eq for WalkFilter {}

impl Hash for WalkFilter {
	fn hash<H: Hasher>(&self, state: &mut H) {
		self.kind.hash(state);
		self.max_file_size.hash(state);
		self.skip_node_modules_unless_seen.hash(state);
		self.mentions_node_modules.hash(state);
		self.glob.hash(state);
	}
}

impl WalkFilter {
	pub const fn all() -> Self {
		Self {
			kind: WalkFilterKind::All,
			max_file_size: None,
			skip_node_modules_unless_seen: false,
			mentions_node_modules: false,
			glob: None,
		}
	}

	pub const fn files_only() -> Self {
		Self {
			kind: WalkFilterKind::Files,
			max_file_size: None,
			skip_node_modules_unless_seen: false,
			mentions_node_modules: false,
			glob: None,
		}
	}

	pub const fn dirs_only() -> Self {
		Self {
			kind: WalkFilterKind::Dirs,
			max_file_size: None,
			skip_node_modules_unless_seen: false,
			mentions_node_modules: false,
			glob: None,
		}
	}

	pub const fn max_file_size(mut self, max_file_size: u64) -> Self {
		self.max_file_size = Some(max_file_size);
		self
	}

	pub const fn node_modules_unless_mentioned(mut self, mentions_node_modules: bool) -> Self {
		self.skip_node_modules_unless_seen = true;
		self.mentions_node_modules = mentions_node_modules;
		self
	}

	pub fn glob(mut self, glob: CompiledWalkGlob) -> Self {
		self.glob = Some(glob);
		self
	}

	fn accepts_path(&self, relative_path: &str) -> bool {
		self.glob.as_ref().is_none_or(|glob| glob.is_match(relative_path))
	}

	pub(crate) fn accepts_collected(&self, entry: &CollectedEntry) -> bool {
		if self.skip_node_modules_unless_seen
			&& !self.mentions_node_modules
			&& entry.path.split('/').any(|component| component == "node_modules")
		{
			return false;
		}
		if self.max_file_size.is_some_and(|max| {
			entry.file_type == FileType::File && entry.size.is_some_and(|size| size > max as f64)
		}) {
			return false;
		}
		let accepts_kind = match self.kind {
			WalkFilterKind::All => true,
			WalkFilterKind::Files => entry.file_type == FileType::File,
			WalkFilterKind::Dirs => entry.file_type == FileType::Dir,
		};
		accepts_kind && self.accepts_path(&entry.path)
	}

	pub fn stream_decision(&self, meta: &EntryMeta<'_>) -> WalkDecision {
		if self.skip_node_modules_unless_seen
			&& !self.mentions_node_modules
			&& meta.relative_path.split('/').any(|component| component == "node_modules")
		{
			return if meta.file_type == FileType::Dir {
				WalkDecision::SkipDescend
			} else {
				WalkDecision::Skip
			};
		}
		if self.max_file_size.is_some_and(|max| {
			meta.file_type == FileType::File && meta.size.is_some_and(|size| size > max as f64)
		}) {
			return WalkDecision::Skip;
		}
		let accepts_kind = match self.kind {
			WalkFilterKind::All => true,
			WalkFilterKind::Files => meta.file_type == FileType::File,
			WalkFilterKind::Dirs => meta.file_type == FileType::Dir,
		};
		if !accepts_kind {
			return WalkDecision::Skip;
		}
		if self.accepts_path(meta.relative_path) {
			WalkDecision::Include
		} else {
			WalkDecision::Skip
		}
	}
}

/// Predicate hook for dynamic walk consumers.
pub trait WalkPredicate {
	fn decide(&mut self, entry: &EntryMeta<'_>) -> WalkDecision;
}

impl<F> WalkPredicate for F
where
	F: for<'entry, 'meta> FnMut(&'entry EntryMeta<'meta>) -> WalkDecision,
{
	fn decide(&mut self, entry: &EntryMeta<'_>) -> WalkDecision {
		self(entry)
	}
}

#[derive(Clone, Copy, Debug, Default)]
pub struct IncludeAllPredicate;

impl WalkPredicate for IncludeAllPredicate {
	fn decide(&mut self, _entry: &EntryMeta<'_>) -> WalkDecision {
		WalkDecision::Include
	}
}

// ── Traversal control ────────────────────────────────────────────────

/// Visitor decision for streaming traversal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WalkControl {
	Continue,
	SkipDescend,
	Quit,
}

/// Status returned by native traversal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WalkStatus {
	Complete,
	Stopped,
}

/// Control returned by unordered parallel file-candidate sinks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParallelWalkControl {
	Continue,
	Stop,
}

// ── Metadata and result types ────────────────────────────────────────

pub struct EntryMeta<'a> {
	pub root:          &'a Path,
	pub absolute_path: Cow<'a, Path>,
	pub relative_path: &'a str,
	pub file_type:     FileType,
	pub mtime:         Option<f64>,
	pub size:          Option<f64>,
	pub depth:         usize,
}

impl<'a> EntryMeta<'a> {
	pub fn from_collected(root: &'a Path, entry: &'a CollectedEntry) -> Self {
		Self {
			root,
			absolute_path: Cow::Owned(entry.absolute_path(root)),
			relative_path: &entry.path,
			file_type: entry.file_type,
			mtime: entry.mtime,
			size: entry.size,
			depth: entry.depth(),
		}
	}

	pub const fn from_entry(root: &'a Path, entry: &Entry<'a>) -> Self {
		Self {
			root,
			absolute_path: Cow::Borrowed(entry.path),
			relative_path: entry.relative,
			file_type: entry.file_type,
			mtime: entry.mtime,
			size: entry.size,
			depth: entry.depth,
		}
	}

	pub const fn from_path_and_metadata(
		root: &'a Path,
		path: &'a Path,
		metadata: &crate::entry::EntryStat,
		relative: &'a str,
		depth: usize,
	) -> Self {
		Self {
			root,
			absolute_path: Cow::Borrowed(path),
			relative_path: relative,
			file_type: metadata.file_type,
			mtime: metadata.mtime,
			size: metadata.size,
			depth,
		}
	}
}

#[derive(Clone, Debug, PartialEq)]
pub struct FileCandidate {
	pub path:     PathBuf,
	pub relative: String,
	pub mtime:    Option<f64>,
	pub size:     Option<f64>,
}

impl FileCandidate {
	pub(crate) fn from_entry(root: &Path, entry: CollectedEntry) -> Self {
		Self {
			path: entry.absolute_path(root),
			relative: entry.path,
			mtime: entry.mtime,
			size: entry.size,
		}
	}

	pub fn depth(&self) -> usize {
		if self.relative.is_empty() {
			0
		} else {
			self.relative.split('/').filter(|c| !c.is_empty()).count()
		}
	}
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WalkBackend {
	Fresh,
	Cached,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum WalkRank {
	PathAsc,
	MtimeDescPathAsc,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WalkStats {
	pub cache_age_ms:     u64,
	pub scanned_entries:  usize,
	pub filtered_entries: usize,
	pub limited_entries:  usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WalkOutcome {
	pub entries: Vec<CollectedEntry>,
	pub backend: WalkBackend,
	pub stats:   WalkStats,
}

// ── Entry types ──────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct CollectedEntry {
	pub path:      String,
	pub file_type: FileType,
	pub mtime:     Option<f64>,
	pub size:      Option<f64>,
}

impl CollectedEntry {
	pub fn absolute_path(&self, root: &Path) -> PathBuf {
		if self.path.is_empty() {
			root.to_path_buf()
		} else {
			root.join(&self.path)
		}
	}

	pub fn depth(&self) -> usize {
		if self.path.is_empty() {
			0
		} else {
			self.path.split('/').filter(|component| !component.is_empty()).count()
		}
	}

	pub const fn is_file(&self) -> bool {
		matches!(self.file_type, FileType::File)
	}

	pub const fn is_dir(&self) -> bool {
		matches!(self.file_type, FileType::Dir)
	}
}

#[derive(Clone, Debug, PartialEq)]
pub struct CollectedEntries {
	pub entries:      Vec<CollectedEntry>,
	pub cache_age_ms: u64,
}

pub struct Entry<'a> {
	pub path:      &'a Path,
	pub relative:  &'a str,
	pub name:      &'a std::ffi::OsStr,
	pub file_type: FileType,
	pub mtime:     Option<f64>,
	pub size:      Option<f64>,
	pub depth:     usize,
}

pub struct DirectoryError<'a> {
	pub path:  &'a Path,
	pub error: &'a io::Error,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreDescendDecision {
	pub emit:    bool,
	pub descend: bool,
	pub stop:    bool,
}

// ── Path helpers ─────────────────────────────────────────────────────

pub fn is_relative_ancestor(parent: &str, child: &str) -> bool {
	if parent == child { return false; }
	if parent.is_empty() { return !child.is_empty(); }
	child.strip_prefix(parent).is_some_and(|suffix| suffix.starts_with('/'))
}

pub fn compare_depth_first_paths(left: &str, right: &str) -> std::cmp::Ordering {
	if left == right { return std::cmp::Ordering::Equal; }
	if is_relative_ancestor(left, right) { return std::cmp::Ordering::Greater; }
	if is_relative_ancestor(right, left) { return std::cmp::Ordering::Less; }
	left.cmp(right)
}

pub fn sort_collected_depth_first(entries: &mut [CollectedEntry]) {
	entries.sort_unstable_by(|left, right| compare_depth_first_paths(&left.path, &right.path));
}

pub fn is_under_pruned_relative_dir(relative: &str, pruned_dirs: &[String]) -> bool {
	pruned_dirs.iter().any(|dir| is_relative_ancestor(dir, relative))
}

// ── Visitor trait ────────────────────────────────────────────────────

pub trait EntryVisitor {
	type Error;
	fn visit(&mut self, entry: Entry<'_>) -> std::result::Result<WalkControl, Self::Error>;
	fn visit_directory_error(
		&mut self,
		_error: DirectoryError<'_>,
	) -> std::result::Result<WalkControl, Self::Error> {
		Ok(WalkControl::Continue)
	}
	fn decide_pre_descend(
		&mut self,
		_meta: &EntryMeta<'_>,
	) -> std::result::Result<PreDescendDecision, Self::Error> {
		Ok(PreDescendDecision { emit: true, descend: true, stop: false })
	}
	fn visit_pre_decided(
		&mut self,
		entry: Entry<'_>,
	) -> std::result::Result<WalkControl, Self::Error> {
		self.visit(entry)
	}
	fn entry_count(&self) -> Option<usize> { None }
}

pub(crate) struct RequestVisitor<'a, V, P> {
	pub(crate) root:      &'a Path,
	pub(crate) filter:    &'a WalkFilter,
	pub(crate) limit:     Option<usize>,
	pub(crate) emitted:   usize,
	pub(crate) visitor:   &'a mut V,
	pub(crate) predicate: P,
}

impl<V, P> EntryVisitor for RequestVisitor<'_, V, P>
where
	V: EntryVisitor,
	P: WalkPredicate,
{
	type Error = V::Error;

	fn visit(&mut self, entry: Entry<'_>) -> std::result::Result<WalkControl, Self::Error> {
		if self.limit.is_some_and(|limit| self.emitted >= limit) {
			return Ok(WalkControl::Quit);
		}
		let meta = EntryMeta {
			root:          self.root,
			absolute_path: Cow::Borrowed(entry.path),
			relative_path: entry.relative,
			file_type:     entry.file_type,
			mtime:         entry.mtime,
			size:          entry.size,
			depth:         entry.depth,
		};
		match self.filter.stream_decision(&meta) {
			WalkDecision::Include => {},
			WalkDecision::Skip => return Ok(WalkControl::Continue),
			WalkDecision::SkipDescend => return Ok(WalkControl::SkipDescend),
			WalkDecision::Stop => return Ok(WalkControl::Quit),
		}
		match self.predicate.decide(&meta) {
			WalkDecision::Include => {},
			WalkDecision::Skip => return Ok(WalkControl::Continue),
			WalkDecision::SkipDescend => return Ok(WalkControl::SkipDescend),
			WalkDecision::Stop => return Ok(WalkControl::Quit),
		}
		self.emitted += 1;
		self.visitor.visit(entry)
	}

	fn visit_directory_error(
		&mut self,
		error: DirectoryError<'_>,
	) -> std::result::Result<WalkControl, Self::Error> {
		self.visitor.visit_directory_error(error)
	}

	fn visit_pre_decided(
		&mut self,
		entry: Entry<'_>,
	) -> std::result::Result<WalkControl, Self::Error> {
		if self.limit.is_some_and(|limit| self.emitted >= limit) {
			return Ok(WalkControl::Quit);
		}
		self.emitted += 1;
		self.visitor.visit(entry)
	}

	fn decide_pre_descend(
		&mut self,
		meta: &EntryMeta<'_>,
	) -> std::result::Result<PreDescendDecision, Self::Error> {
		let is_dir = meta.file_type == FileType::Dir;
		match self.filter.stream_decision(meta) {
			WalkDecision::Include => {},
			WalkDecision::Skip => {
				return Ok(PreDescendDecision { emit: false, descend: is_dir, stop: false });
			},
			WalkDecision::SkipDescend => {
				return Ok(PreDescendDecision { emit: false, descend: false, stop: false });
			},
			WalkDecision::Stop => {
				return Ok(PreDescendDecision { emit: false, descend: false, stop: true });
			},
		}
		match self.predicate.decide(meta) {
			WalkDecision::Include => {
				Ok(PreDescendDecision { emit: true, descend: is_dir, stop: false })
			},
			WalkDecision::Skip => {
				Ok(PreDescendDecision { emit: false, descend: is_dir, stop: false })
			},
			WalkDecision::SkipDescend => {
				Ok(PreDescendDecision { emit: false, descend: false, stop: false })
			},
			WalkDecision::Stop => {
				Ok(PreDescendDecision { emit: false, descend: false, stop: true })
			},
		}
	}
}

pub(crate) struct ClosureEntryVisitor<'a, V, D> {
	pub(crate) root:            &'a Path,
	pub(crate) visit:           V,
	pub(crate) directory_error: D,
}

impl<E, V, D> EntryVisitor for ClosureEntryVisitor<'_, V, D>
where
	V: for<'entry> FnMut(EntryMeta<'entry>) -> std::result::Result<WalkDecision, E>,
	D: for<'error> FnMut(DirectoryError<'error>) -> std::result::Result<WalkDecision, E>,
{
	type Error = E;

	fn visit(&mut self, entry: Entry<'_>) -> std::result::Result<WalkControl, Self::Error> {
		let meta = EntryMeta {
			root:          self.root,
			absolute_path: Cow::Borrowed(entry.path),
			relative_path: entry.relative,
			file_type:     entry.file_type,
			mtime:         entry.mtime,
			size:          entry.size,
			depth:         entry.depth,
		};
		(self.visit)(meta).map(walk_decision_to_control)
	}

	fn visit_directory_error(
		&mut self,
		error: DirectoryError<'_>,
	) -> std::result::Result<WalkControl, Self::Error> {
		(self.directory_error)(error).map(walk_decision_to_control)
	}
}

pub(crate) const fn walk_decision_to_control(decision: WalkDecision) -> WalkControl {
	match decision {
		WalkDecision::Include | WalkDecision::Skip => WalkControl::Continue,
		WalkDecision::SkipDescend => WalkControl::SkipDescend,
		WalkDecision::Stop => WalkControl::Quit,
	}
}

// ── Error ────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum WalkError<E> {
	Interrupted(E),
	InvalidData { path: PathBuf, message: String },
}

impl<E: fmt::Display> fmt::Display for WalkError<E> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Interrupted(err) => write!(f, "native directory scan interrupted: {err}"),
			Self::InvalidData { path, message } => {
				write!(f, "native directory scan failed for {}: {message}", path.display())
			},
		}
	}
}

impl<E> std::error::Error for WalkError<E>
where
	E: std::error::Error + 'static,
{
	fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
		match self {
			Self::Interrupted(err) => Some(err),
			Self::InvalidData { .. } => None,
		}
	}
}

// ── Collect visitor ──────────────────────────────────────────────────
pub(crate) struct CollectedVisitor<E> {
	pub(crate) entries: Vec<CollectedEntry>,
	_error:  std::marker::PhantomData<fn() -> E>,
}

impl<E> CollectedVisitor<E> {
	pub(crate) const fn new() -> Self {
		Self { entries: Vec::new(), _error: std::marker::PhantomData }
	}
}

impl<E> EntryVisitor for CollectedVisitor<E> {
	type Error = E;

	fn visit(&mut self, entry: Entry<'_>) -> std::result::Result<WalkControl, Self::Error> {
		self.entries.push(CollectedEntry {
			path: entry.relative.to_string(),
			file_type: entry.file_type,
			mtime: entry.mtime,
			size: entry.size,
		});
		Ok(WalkControl::Continue)
	}
}

// ── Platform-specific filesystem helpers ─────────────────────────────

#[allow(clippy::missing_const_for_fn, reason = "calls non-const root_device_id on unix")]
pub fn root_device_id(path: &Path, follow_links: crate::policy::FollowLinks) -> Option<u64> {
	#[cfg(unix)]
	{
		use std::os::unix::fs::MetadataExt;
		metadata_for_follow_policy(path, follow_links.follow_at_depth(0))
			.ok()
			.map(|metadata| metadata.dev())
	}
	#[cfg(not(unix))]
	{
		let _ = path;
		let _ = follow_links;
		None
	}
}

#[cfg(unix)]
pub fn is_path_on_root_file_system(
	path: &Path,
	depth: usize,
	follow_links: crate::policy::FollowLinks,
	root_device: Option<u64>,
) -> bool {
	use std::os::unix::fs::MetadataExt;
	let Some(root_device) = root_device else { return true; };
	metadata_for_follow_policy(path, follow_links.follow_at_depth(depth))
		.is_ok_and(|metadata| metadata.dev() == root_device)
}

#[cfg(not(unix))]
pub const fn is_path_on_root_file_system(
	_path: &Path, _depth: usize, _follow_links: crate::policy::FollowLinks, _root_device: Option<u64>,
) -> bool {
	true
}

#[cfg(unix)]
pub(crate) fn is_effective_path_on_root_file_system(
	path: &Path,
	depth: usize,
	follow_links: crate::policy::FollowLinks,
	root_device: Option<u64>,
	followed_metadata: Option<&std::fs::Metadata>,
) -> bool {
	use std::os::unix::fs::MetadataExt;
	let Some(root_device) = root_device else { return true; };
	if let Some(metadata) = followed_metadata {
		return metadata.dev() == root_device;
	}
	metadata_for_follow_policy(path, follow_links.follow_at_depth(depth))
		.is_ok_and(|metadata| metadata.dev() == root_device)
}

#[cfg(not(unix))]
pub(crate) const fn is_effective_path_on_root_file_system(
	_path: &Path, _depth: usize, _follow_links: crate::policy::FollowLinks,
	_root_device: Option<u64>, _followed_metadata: Option<&std::fs::Metadata>,
) -> bool {
	true
}

#[cfg(unix)]
pub(crate) fn metadata_for_follow_policy(path: &Path, follow: bool) -> io::Result<std::fs::Metadata> {
	if follow { std::fs::metadata(path) } else { std::fs::symlink_metadata(path) }
}

#[allow(clippy::missing_const_for_fn, reason = "calls non-const root_device_id on unix")]
pub(crate) fn root_device_for_options(root: &Path, options: crate::policy::WalkOptions) -> Option<u64> {
	if options.same_file_system { root_device_id(root, options.follow_links) } else { None }
}

// ── Directory scanning types ─────────────────────────────────────────
pub(crate) struct RawDirEntry<'a> {
	pub(crate) name:      Cow<'a, std::ffi::OsStr>,
	pub(crate) file_type: FileType,
	pub(crate) mtime:     Option<f64>,
	pub(crate) size:      Option<f64>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadDirControl {
	Continue,
	Stop,
}

pub(crate) enum ReadDirError<E> {
	Io(io::Error),
	Walk(WalkError<E>),
}

impl<E> From<io::Error> for ReadDirError<E> {
	fn from(err: io::Error) -> Self { Self::Io(err) }
}

fn file_type_from_metadata(metadata: &std::fs::Metadata) -> Option<FileType> {
	let file_type = metadata.file_type();
	if file_type.is_symlink() { Some(FileType::Symlink) }
	else if file_type.is_dir() { Some(FileType::Dir) }
	else if file_type.is_file() { Some(FileType::File) }
	else { None }
}
pub(crate) struct RootEntry {
	pub(crate) file_type: FileType,
	pub(crate) mtime:     Option<f64>,
	pub(crate) size:      Option<f64>,
}

pub(crate) fn root_entry<E>(
	root: &Path,
	detail: crate::policy::WalkDetail,
	follow_links: crate::policy::FollowLinks,
) -> std::result::Result<Option<RootEntry>, WalkError<E>> {
	let metadata = if follow_links.follow_at_depth(0) {
		match std::fs::metadata(root) {
			Ok(metadata) => metadata,
			Err(err) if is_missing_metadata_error(&err) => {
				std::fs::symlink_metadata(root).map_err(|err| WalkError::InvalidData {
					path: root.to_path_buf(), message: err.to_string(),
				})?
			},
			Err(err) => {
				return Err(WalkError::InvalidData { path: root.to_path_buf(), message: err.to_string() });
			},
		}
	} else {
		std::fs::symlink_metadata(root).map_err(|err| WalkError::InvalidData {
			path: root.to_path_buf(), message: err.to_string(),
		})?
	};
	Ok(entry_from_metadata(&metadata, detail))
}

fn entry_from_metadata(metadata: &std::fs::Metadata, detail: crate::policy::WalkDetail) -> Option<RootEntry> {
	let file_type = file_type_from_metadata(metadata)?;
	let size = if detail == crate::policy::WalkDetail::Full && file_type == FileType::File {
		Some(metadata.len() as f64)
	} else { None };
	let mtime = if detail == crate::policy::WalkDetail::Full {
		metadata.modified().ok()
			.and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
			.map(|duration| duration.as_millis() as f64)
	} else { None };
	Some(RootEntry { file_type, mtime, size })
}

fn is_missing_metadata_error(err: &io::Error) -> bool {
	matches!(err.kind(), io::ErrorKind::NotFound | io::ErrorKind::NotADirectory)
}

#[derive(Default)]
pub(crate) struct DirScratch {
	pub(crate) entries:     Vec<DirEntryRecord>,
	#[cfg(unix)]
	pub(crate) name_bytes:  Vec<u8>,
	pub(crate) read_buffer: Vec<u8>,
}
// DirEntryRecord has non-default FileType field, derive(Default) removed
pub(crate) struct DirEntryRecord {
	#[cfg(unix)]
	pub(crate) name_start: usize,
	#[cfg(unix)]
	pub(crate) name_len:   usize,
	pub(crate) file_type:  FileType,
	pub(crate) mtime:      Option<f64>,
	pub(crate) size:       Option<f64>,
	#[cfg(not(unix))]
	pub(crate) name:       std::ffi::OsString,
}
impl DirScratch {
	pub(crate) fn clear_listing(&mut self) {
		self.entries.clear();
		#[cfg(unix)]
		self.name_bytes.clear();
	}

	pub(crate) fn push(&mut self, entry: RawDirEntry<'_>) {
		let name_os: &std::ffi::OsStr = entry.name.as_ref();
		let name = name_os.as_bytes();
		let name_start = self.name_bytes.len();
		self.name_bytes.extend_from_slice(name);
		self.entries.push(DirEntryRecord {
			name_start, name_len: name.len(),
			file_type: entry.file_type, mtime: entry.mtime, size: entry.size,
		});
	}

	#[cfg(not(unix))]
	fn push(&mut self, entry: RawDirEntry<'_>) {
		self.entries.push(DirEntryRecord {
			name: entry.name.into_owned(),
			file_type: entry.file_type, mtime: entry.mtime, size: entry.size,
		});
	}

	#[cfg(unix)]
	pub(crate) fn sort_by_name(&mut self) {
		let names = &self.name_bytes;
		self.entries.sort_unstable_by(|left, right| {
			let left_name = &names[left.name_start..left.name_start + left.name_len];
			let right_name = &names[right.name_start..right.name_start + right.name_len];
			left_name.cmp(right_name)
		});
	}

	#[cfg(not(unix))]
	pub(crate) fn sort_by_name(&mut self) {
		self.entries.sort_unstable_by(|left, right| left.name.cmp(&right.name));
	}

	#[cfg(unix)]
	pub(crate) fn name<'a>(&'a self, entry: &DirEntryRecord) -> &'a std::ffi::OsStr {
		std::ffi::OsStr::from_bytes(&self.name_bytes[entry.name_start..entry.name_start + entry.name_len])
	}

	#[cfg(not(unix))]
	fn name<'a>(&'a self, entry: &'a DirEntryRecord) -> &'a std::ffi::OsStr {
		&entry.name
	}
}

// ── Collect entries native ───────────────────────────────────────────

pub fn collect_entries_native<E, H>(
	root: &Path,
	options: crate::policy::WalkOptions,
	heartbeat: H,
) -> std::result::Result<CollectedEntries, WalkError<E>>
where
	H: FnMut() -> std::result::Result<(), E>,
{
	let mut collector = CollectedVisitor::new();
	let _status = crate::walk::walk_entries(root, options, &mut collector, heartbeat)?;
	if options.contents_first {
		sort_collected_depth_first(&mut collector.entries);
	} else {
		collector.entries.sort_unstable_by(|a, b| a.path.cmp(&b.path));
	}
	Ok(CollectedEntries { entries: collector.entries, cache_age_ms: 0 })
}

// ── Directory identity ───────────────────────────────────────────────

#[derive(PartialEq, Eq)]
pub(crate) enum DirectoryIdentity {
	#[cfg(unix)]
	Unix { dev: u64, ino: u64 },
	#[cfg(not(unix))]
	Generic(PathBuf),
}

#[derive(Default)]
pub(crate) struct SymlinkAncestorStack {
	stack: Vec<DirectoryIdentity>,
}

impl SymlinkAncestorStack {
	pub(crate) fn contains(&self, identity: &DirectoryIdentity) -> bool {
		self.stack.contains(identity)
	}
	pub(crate) fn push(&mut self, identity: DirectoryIdentity) {
		self.stack.push(identity);
	}
	pub(crate) fn pop(&mut self) {
		self.stack.pop();
	}
}

pub fn directory_identity(path: &Path) -> io::Result<DirectoryIdentity> {
	#[cfg(unix)]
	{
		use std::os::unix::fs::MetadataExt;
		let metadata = std::fs::metadata(path)?;
		Ok(DirectoryIdentity::Unix { dev: metadata.dev(), ino: metadata.ino() })
	}
	#[cfg(not(unix))]
	{
		let canonical = std::fs::canonicalize(path)?;
		Ok(DirectoryIdentity::Generic(canonical))
	}
}

// ── Shared stat type for walk module ─────────────────────────────────

#[derive(Clone, Copy)]
pub struct EntryStat {
	pub file_type: FileType,
	pub mtime:     Option<f64>,
	pub size:      Option<f64>,
}
