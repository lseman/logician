//! Core walking logic: [`WalkContext`], [`walk_entries`], [`collect_entries`].

use std::{borrow::Cow, fmt, io, path::{Path, PathBuf}, sync::Arc};

use crate::entry::{
	CollectedEntries, CollectedVisitor, DirectoryError, DirScratch, Entry, EntryMeta, EntryVisitor, ReadDirControl,
	ReadDirError, SymlinkAncestorStack, WalkControl, WalkError, WalkStatus, root_device_for_options, sort_collected_depth_first,
	directory_identity, is_effective_path_on_root_file_system,
};
use crate::ignore::{
	FastIgnore, IgnoreEntryNames, IgnoreState, handle_read_dir_error,
	is_dot_entry, is_git_name, is_hidden_name, is_node_modules_name,
	is_skippable_directory_error, entry_name, push_relative_name,
};
use crate::policy::{DirectoryErrorMode, FollowLinks, WalkDetail, WalkOptions};
pub(crate) struct WalkContext<'a, H> {
	root_path:         &'a Path,
	options:           WalkOptions,
	root_device:       Option<u64>,
	symlink_ancestors: SymlinkAncestorStack,
	matcher:           FastIgnore,
	absolute_path:     PathBuf,
	relative_path:     String,
	scratch_pool:      Vec<DirScratch>,
	visited:           usize,
	heartbeat:         H,
}

impl<H> WalkContext<'_, H> {
	fn walk_root<V>(
		&mut self,
		root: &Path,
		root_ignore: &Arc<IgnoreState>,
		visitor: &mut V,
	) -> std::result::Result<WalkStatus, WalkError<V::Error>>
	where
		V: EntryVisitor,
		H: FnMut() -> std::result::Result<(), V::Error>,
	{
		let root_entry = root_entry(root, self.options.detail, self.options.follow_links)?;
		let Some(root_entry) = root_entry else {
			return Ok(WalkStatus::Complete);
		};

		// Seed the ancestor stack only when descendant symlink traversal can loop.
		if self.options.follow_links == FollowLinks::Always
			&& let Ok(id) = directory_identity(root)
		{
			self.symlink_ancestors.push(id);
		}

		let meta = EntryMeta {
			root,
			absolute_path: Cow::Borrowed(root),
			relative_path: "",
			file_type: root_entry.file_type,
			mtime: root_entry.mtime,
			size: root_entry.size,
			depth: 0,
		};

		let decision = visitor
			.decide_pre_descend(&meta)
			.map_err(WalkError::Interrupted)?;
		if decision.stop {
			return Ok(WalkStatus::Stopped);
		}

		if !self.options.contents_first
			&& self.options.emit_root
			&& self.options.min_depth == 0
			&& decision.emit
		{
			let name = root.file_name().unwrap_or(root.as_os_str());
			match visitor
				.visit_pre_decided(Entry {
					path: root,
					relative: "",
					name,
					file_type: root_entry.file_type,
					mtime: root_entry.mtime,
					size: root_entry.size,
					depth: 0,
				})
				.map_err(WalkError::Interrupted)?
			{
				WalkControl::Quit => return Ok(WalkStatus::Stopped),
				WalkControl::SkipDescend => return Ok(WalkStatus::Complete),
				WalkControl::Continue => {},
			}
		}

		let stopped =
			if root_entry.file_type == crate::entry::FileType::Dir && self.options.max_depth > 0 && decision.descend
			{
				self.walk_dir(0, root_ignore, false, visitor)?
			} else {
				false
			};

		if stopped {
			return Ok(WalkStatus::Stopped);
		}

		if self.options.contents_first
			&& self.options.emit_root
			&& self.options.min_depth == 0
			&& decision.emit
		{
			let name = root.file_name().unwrap_or(root.as_os_str());
			match visitor
				.visit_pre_decided(Entry {
					path: root,
					relative: "",
					name,
					file_type: root_entry.file_type,
					mtime: root_entry.mtime,
					size: root_entry.size,
					depth: 0,
				})
				.map_err(WalkError::Interrupted)?
			{
				WalkControl::Quit => return Ok(WalkStatus::Stopped),
				WalkControl::SkipDescend | WalkControl::Continue => {},
			}
		}

		Ok(WalkStatus::Complete)
	}

	fn take_scratch(&mut self) -> DirScratch {
		let mut scratch = self.scratch_pool.pop().unwrap_or_default();
		scratch.clear_listing();
		scratch
	}

	fn recycle_scratch(&mut self, mut scratch: DirScratch) {
		scratch.clear_listing();
		self.scratch_pool.push(scratch);
	}

	fn push_entry_path(&mut self, name: &std::ffi::OsStr, name_str: &str) -> usize {
		let relative_len = self.relative_path.len();
		self.absolute_path.push(name);
		push_relative_name(&mut self.relative_path, name_str);
		relative_len
	}

	fn pop_entry_path(&mut self, relative_len: usize) {
		self.relative_path.truncate(relative_len);
		self.absolute_path.pop();
	}

	fn walk_dir<V>(
		&mut self,
		depth: usize,
		ignore_state: &Arc<IgnoreState>,
		derive_ignore_from_entries: bool,
		visitor: &mut V,
	) -> std::result::Result<bool, WalkError<V::Error>>
	where
		V: EntryVisitor,
		H: FnMut() -> std::result::Result<(), V::Error>,
	{
		if self.options.follow_links == FollowLinks::Always
			&& let Ok(identity) = directory_identity(&self.absolute_path)
		{
			self.symlink_ancestors.push(identity);
			let result = self.walk_dir_inner(depth, ignore_state, derive_ignore_from_entries, visitor);
			self.symlink_ancestors.pop();
			return result;
		}

		self.walk_dir_inner(depth, ignore_state, derive_ignore_from_entries, visitor)
	}

	fn walk_dir_inner<V>(
		&mut self,
		depth: usize,
		ignore_state: &Arc<IgnoreState>,
		derive_ignore_from_entries: bool,
		visitor: &mut V,
	) -> std::result::Result<bool, WalkError<V::Error>>
	where
		V: EntryVisitor,
		H: FnMut() -> std::result::Result<(), V::Error>,
	{
		let mut scratch = self.take_scratch();
		let ignore_entries = match collect_directory_entries(
			&self.absolute_path,
			self.options.detail,
			&mut scratch,
			&self.matcher,
			derive_ignore_from_entries,
		) {
			Ok(ignore_entries) => ignore_entries,
			Err(err) => {
				let dir = self.absolute_path.clone();
				self.recycle_scratch(scratch);
				return handle_read_dir_error(&dir, err, self.options, visitor);
			},
		};
		if self.options.order == crate::policy::WalkOrder::Path {
			scratch.sort_by_name();
		}
		let dir_ignore = self.matcher.state_from_entries(
			ignore_state,
			&self.absolute_path,
			ignore_entries,
			derive_ignore_from_entries,
		);

		for index in 0..scratch.entries.len() {
			if self.visited == 0 || self.visited >= 128 {
				self.visited = 0;
				(self.heartbeat)().map_err(WalkError::Interrupted)?;
			}
			self.visited += 1;

			let entry = &scratch.entries[index];
			let name = scratch.name(entry);
			if is_dot_entry(name) {
				continue;
			}
			if !self.options.include_hidden && is_hidden_name(name) {
				continue;
			}
			if (self.options.skip_git && is_git_name(name))
				|| (self.options.skip_node_modules && is_node_modules_name(name))
			{
				continue;
			}

			let name_str = entry_name(name);
			if name_str.is_empty() {
				continue;
			}
			let next_depth = depth + 1;
			if next_depth > self.options.max_depth {
				continue;
			}

			let relative_len = self.push_entry_path(name, &name_str);
			let entry_result = self.walk_current_entry(
				name,
				entry.file_type,
				entry.mtime,
				entry.size,
				next_depth,
				&dir_ignore,
				visitor,
			);
			self.pop_entry_path(relative_len);
			if entry_result? {
				self.recycle_scratch(scratch);
				return Ok(true);
			}
		}

		self.recycle_scratch(scratch);
		Ok(false)
	}

	fn walk_current_entry<V>(
		&mut self,
		name: &std::ffi::OsStr,
		entry_file_type: crate::entry::FileType,
		entry_mtime: Option<f64>,
		entry_size: Option<f64>,
		next_depth: usize,
		dir_ignore: &Arc<IgnoreState>,
		visitor: &mut V,
	) -> std::result::Result<bool, WalkError<V::Error>>
	where
		V: EntryVisitor,
		H: FnMut() -> std::result::Result<(), V::Error>,
	{
		let mut file_type = entry_file_type;
		let mut mtime = entry_mtime;
		let mut size = entry_size;
		let mut is_dir = entry_file_type == crate::entry::FileType::Dir;
		let mut descend = is_dir;
		let mut followed_symlink_dir = false;
		let followed_metadata = if entry_file_type == crate::entry::FileType::Symlink
			&& self.options.follow_links == FollowLinks::Always
		{
			match std::fs::metadata(&self.absolute_path) {
				Ok(metadata) => Some(metadata),
				Err(err) => {
					if self.options.directory_errors == DirectoryErrorMode::SkipSkippable
						&& is_skippable_directory_error(&err)
					{
						return Ok(false);
					}
					if self.options.directory_errors == DirectoryErrorMode::Visit {
						match visitor
							.visit_directory_error(DirectoryError {
								path:  &self.absolute_path,
								error: &err,
							})
							.map_err(WalkError::Interrupted)?
						{
							WalkControl::Quit => return Ok(true),
							WalkControl::SkipDescend | WalkControl::Continue => {
								return Ok(false);
							},
						}
					}
					return Err(WalkError::InvalidData {
						path:    self.absolute_path.clone(),
						message: err.to_string(),
					});
				},
			}
		} else {
			None
		};

		if let Some(target_metadata) = followed_metadata.as_ref() {
			let Some(target_file_type) = file_type_from_metadata(target_metadata) else {
				return Ok(false);
			};
			file_type = target_file_type;
			if target_file_type == crate::entry::FileType::Dir {
				is_dir = true;
				descend = true;
				followed_symlink_dir = true;
			} else {
				is_dir = false;
				descend = false;
			}
			if self.options.detail == WalkDetail::Full {
				if target_file_type == crate::entry::FileType::File {
					size = Some(target_metadata.len() as f64);
				} else {
					size = None;
				}
				mtime = target_metadata
					.modified()
					.ok()
					.and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
					.map(|duration| duration.as_millis() as f64);
			}
		}

		if self
			.matcher
			.is_ignored(dir_ignore, &self.absolute_path, is_dir)
		{
			return Ok(false);
		}

		if !is_effective_path_on_root_file_system(
			&self.absolute_path,
			next_depth,
			self.options.follow_links,
			self.root_device,
			followed_metadata.as_ref(),
		) {
			return Ok(false);
		}

		if followed_symlink_dir && descend {
			match directory_identity(&self.absolute_path) {
				Ok(target_id) => {
					if self.symlink_ancestors.contains(&target_id) {
						let loop_err = io::Error::other("filesystem loop detected");
						if self.options.directory_errors == DirectoryErrorMode::Visit {
							match visitor
								.visit_directory_error(DirectoryError {
									path:  &self.absolute_path,
									error: &loop_err,
								})
								.map_err(WalkError::Interrupted)?
							{
								WalkControl::Quit => return Ok(true),
								WalkControl::SkipDescend | WalkControl::Continue => {
									return Ok(false);
								},
							}
						} else if self.options.directory_errors == DirectoryErrorMode::SkipSkippable {
							return Ok(false);
						}
						return Err(WalkError::InvalidData {
							path:    self.absolute_path.clone(),
							message: "filesystem loop detected".to_string(),
						});
					}
				},
				Err(err) => {
					if self.options.directory_errors == DirectoryErrorMode::SkipSkippable
						&& is_skippable_directory_error(&err)
					{
						return Ok(false);
					}
					if self.options.directory_errors == DirectoryErrorMode::Visit {
						match visitor
							.visit_directory_error(DirectoryError {
								path:  &self.absolute_path,
								error: &err,
							})
							.map_err(WalkError::Interrupted)?
						{
							WalkControl::Quit => return Ok(true),
							WalkControl::SkipDescend | WalkControl::Continue => {
								return Ok(false);
							},
						}
					}
					return Err(WalkError::InvalidData {
						path:    self.absolute_path.clone(),
						message: err.to_string(),
					});
				},
			}
		}

		let decision = {
			let meta = EntryMeta {
				root: self.root_path,
				absolute_path: Cow::Borrowed(self.absolute_path.as_path()),
				relative_path: &self.relative_path,
				file_type,
				mtime,
				size,
				depth: next_depth,
			};
			visitor
				.decide_pre_descend(&meta)
				.map_err(WalkError::Interrupted)?
		};
		if decision.stop {
			return Ok(true);
		}

		if !self.options.contents_first && decision.emit && next_depth >= self.options.min_depth {
			match visitor
				.visit_pre_decided(Entry {
					path: self.absolute_path.as_path(),
					relative: &self.relative_path,
					name,
					file_type,
					mtime,
					size,
					depth: next_depth,
				})
				.map_err(WalkError::Interrupted)?
			{
				WalkControl::Quit => return Ok(true),
				WalkControl::SkipDescend => return Ok(false),
				WalkControl::Continue => {},
			}
		}

		let child_stopped = if descend && next_depth < self.options.max_depth && decision.descend {
			self.walk_dir(next_depth, dir_ignore, true, visitor)?
		} else {
			false
		};

		if child_stopped {
			return Ok(true);
		}

		if self.options.contents_first && decision.emit && next_depth >= self.options.min_depth {
			match visitor
				.visit_pre_decided(Entry {
					path: self.absolute_path.as_path(),
					relative: &self.relative_path,
					name,
					file_type,
					mtime,
					size,
					depth: next_depth,
				})
				.map_err(WalkError::Interrupted)?
			{
				WalkControl::Quit => return Ok(true),
				WalkControl::SkipDescend | WalkControl::Continue => {},
			}
		}

		Ok(false)
	}
}

// ── Public API ───────────────────────────────────────────────────────

/// Collect entries by walking the filesystem.
pub fn collect_entries<E, H>(
	root: &Path,
	options: WalkOptions,
	heartbeat: H,
) -> std::result::Result<CollectedEntries, WalkError<String>>
where
	H: Fn() -> std::result::Result<(), E> + Sync,
	E: fmt::Display,
{
	let result = collect_entries_native(root, options, || heartbeat().map_err(|err| err.to_string()));
	match result {
		Ok(entries) => Ok(entries),
		Err(err) => Err(WalkError::InvalidData {
			path: root.to_path_buf(),
			message: err.to_string(),
		}),
	}
}

fn collect_entries_native<E, H>(
	root: &Path,
	options: WalkOptions,
	heartbeat: H,
) -> std::result::Result<CollectedEntries, WalkError<E>>
where
	H: FnMut() -> std::result::Result<(), E>,
{
	let mut collector = CollectedVisitor::new();
	let _status = walk_entries(root, options, &mut collector, heartbeat)?;
	if options.contents_first {
		sort_collected_depth_first(&mut collector.entries);
	} else {
		collector.entries.sort_unstable_by(|a, b| a.path.cmp(&b.path));
	}
	Ok(CollectedEntries { entries: collector.entries, cache_age_ms: 0 })
}

/// Stream entries using the native scanner.
pub fn walk_entries<V, H>(
	root: &Path,
	options: WalkOptions,
	visitor: &mut V,
	heartbeat: H,
) -> std::result::Result<WalkStatus, WalkError<V::Error>>
where
	V: EntryVisitor,
	H: FnMut() -> std::result::Result<(), V::Error>,
{
	if options.min_depth > options.max_depth {
		return Ok(WalkStatus::Complete);
	}

	let root_device = root_device_for_options(root, options);
	let matcher = FastIgnore::new(options.use_gitignore);
	let root_ignore = matcher.root_state(root);

	let mut context = WalkContext {
		root_path: root,
		options,
		root_device,
		symlink_ancestors: SymlinkAncestorStack::default(),
		matcher,
		absolute_path: root.to_path_buf(),
		relative_path: String::new(),
		scratch_pool: Vec::new(),
		visited: 0,
		heartbeat,
	};

	context.walk_root(root, &root_ignore, visitor)
}

// ── Helper functions ─────────────────────────────────────────────────

fn collect_directory_entries<E>(
	dir: &Path,
	detail: WalkDetail,
	scratch: &mut DirScratch,
	matcher: &FastIgnore,
	derive_ignore_from_entries: bool,
) -> std::result::Result<IgnoreEntryNames, ReadDirError<E>> {
	scratch.clear_listing();
	let mut ignore_entries = IgnoreEntryNames::default();
	let track_ignore_entries = derive_ignore_from_entries && matcher.use_gitignore;
	let mut read_buffer = std::mem::take(&mut scratch.read_buffer);
	let result = crate::ignore::platform::read_dir_entries(dir, detail, &mut read_buffer, |entry| {
		if track_ignore_entries {
			ignore_entries.record(entry.name.as_ref(), entry.file_type);
		}
		scratch.push(entry);
		Ok(ReadDirControl::Continue)
	});
	scratch.read_buffer = read_buffer;
	result?;
	Ok(ignore_entries)
}

fn file_type_from_metadata(metadata: &std::fs::Metadata) -> Option<crate::entry::FileType> {
	let file_type = metadata.file_type();
	if file_type.is_symlink() { Some(crate::entry::FileType::Symlink) }
	else if file_type.is_dir() { Some(crate::entry::FileType::Dir) }
	else if file_type.is_file() { Some(crate::entry::FileType::File) }
	else { None }
}

fn is_missing_metadata_error(err: &io::Error) -> bool {
	matches!(err.kind(), io::ErrorKind::NotFound | io::ErrorKind::NotADirectory)
}

fn root_entry<E>(
	root: &Path,
	detail: WalkDetail,
	follow_links: FollowLinks,
) -> std::result::Result<Option<crate::entry::RootEntry>, WalkError<E>> {
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

fn entry_from_metadata(metadata: &std::fs::Metadata, detail: WalkDetail) -> Option<crate::entry::RootEntry> {
	let file_type = file_type_from_metadata(metadata)?;
	let size = if detail == WalkDetail::Full && file_type == crate::entry::FileType::File {
		Some(metadata.len() as f64)
	} else { None };
	let mtime = if detail == WalkDetail::Full {
		metadata.modified().ok()
			.and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
			.map(|duration| duration.as_millis() as f64)
	} else { None };
	Some(crate::entry::RootEntry { file_type, mtime, size })
}
