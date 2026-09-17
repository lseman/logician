//! Ignore system, directory scanning helpers, and platform-specific read.

use std::{
	borrow::Cow,
	ffi::OsStr,
	io::{self, BufRead, BufReader},
	path::Path,
	sync::Arc,
};

use crate::entry::{DirectoryError, EntryVisitor, FileType, ReadDirControl, ReadDirError, WalkControl, WalkError};
use crate::policy::{DirectoryErrorMode, WalkDetail, WalkOptions};
pub(crate) fn read_dir_entries<F, E>(
	dir: &Path,
	detail: WalkDetail,
	scratch: &mut crate::entry::DirScratch,
	matcher: &FastIgnore,
	derive_ignore_from_entries: bool,
) -> std::result::Result<IgnoreEntryNames, ReadDirError<E>>
where
	F: FnMut(crate::entry::RawDirEntry<'_>) -> std::result::Result<WalkControl, WalkError<E>>,
{
	let mut read_buffer = std::mem::take(&mut scratch.read_buffer);
	let mut ignore_entries = IgnoreEntryNames::default();
	let track_ignore_entries = derive_ignore_from_entries && matcher.use_gitignore;
	let result = platform::read_dir_entries(dir, detail, &mut read_buffer, |entry| {
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

/// Return whether [`WalkDetail::Full`] provides file sizes without per-entry
/// metadata syscalls on this platform.
pub const fn supports_cheap_size_hints() -> bool {
	platform::CHEAP_SIZE_HINTS
}

pub(crate) fn handle_read_dir_error<V>(
	dir: &Path,
	err: ReadDirError<V::Error>,
	options: WalkOptions,
	visitor: &mut V,
) -> std::result::Result<bool, WalkError<V::Error>>
where
	V: EntryVisitor,
{
	match err {
		ReadDirError::Walk(err) => Err(err),
		ReadDirError::Io(err)
			if options.directory_errors == DirectoryErrorMode::SkipSkippable
				&& is_skippable_directory_error(&err) =>
		{
			Ok(false)
		},
		ReadDirError::Io(err) if options.directory_errors == DirectoryErrorMode::Visit => {
			match visitor
				.visit_directory_error(DirectoryError { path: dir, error: &err })
				.map_err(WalkError::Interrupted)?
			{
				WalkControl::Quit => Ok(true),
				WalkControl::SkipDescend | WalkControl::Continue => Ok(false),
			}
		},
		ReadDirError::Io(err) => {
			Err(WalkError::InvalidData { path: dir.to_path_buf(), message: err.to_string() })
		},
	}
}

pub(crate) fn is_skippable_directory_error(err: &io::Error) -> bool {
	matches!(
		err.kind(),
		io::ErrorKind::NotFound | io::ErrorKind::NotADirectory | io::ErrorKind::PermissionDenied
	)
}

pub(crate) fn is_dot_entry(name: &OsStr) -> bool {
	name == OsStr::new(".") || name == OsStr::new("..")
}

pub(crate) fn is_git_name(name: &OsStr) -> bool {
	name == OsStr::new(".git")
}

pub(crate) fn is_node_modules_name(name: &OsStr) -> bool {
	name == OsStr::new("node_modules")
}

pub(crate) fn entry_name(name: &OsStr) -> Cow<'_, str> {
	name
		.to_str()
		.map_or_else(|| name.to_string_lossy(), Cow::Borrowed)
}

#[cfg(unix)]
pub(crate) fn is_hidden_name(name: &OsStr) -> bool {
	use std::os::unix::ffi::OsStrExt;
	name.as_bytes().first() == Some(&b'.')
}

#[cfg(windows)]
pub(crate) fn is_hidden_name(name: &OsStr) -> bool {
	use std::os::windows::ffi::OsStrExt;
	name.encode_wide().next() == Some(b'.' as u16)
}

#[cfg(not(any(unix, windows)))]
pub(crate) fn is_hidden_name(name: &OsStr) -> bool {
	name
		.to_str()
		.is_some_and(|value| value.as_bytes().first() == Some(&b'.'))
}

pub(crate) fn push_relative_name(relative: &mut String, name: &str) {
	if !relative.is_empty() {
		relative.push('/');
	}
	relative.push_str(name);
}

pub(crate) fn mtime_millis(seconds: i64, nanos: i64) -> Option<f64> {
	if seconds < 0 {
		return None;
	}
	Some((seconds as f64).mul_add(1000.0, nanos.max(0) as f64 / 1_000_000.0))
}

pub(crate) struct IgnoreState {
	parent:              Option<Arc<Self>>,
	ignore_matcher:      Option<ignore::gitignore::Gitignore>,
	gitignore_matcher:   Option<ignore::gitignore::Gitignore>,
	git_exclude_matcher: Option<ignore::gitignore::Gitignore>,
	has_git:             bool,
	chain_has_matchers:  bool,
	any_git:             bool,
}

pub(crate) struct FastIgnore {
	pub(crate) global:        Option<ignore::gitignore::Gitignore>,
	pub(crate) use_gitignore: bool,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct IgnoreEntryNames {
	pub(crate) ignore_file:    bool,
	pub(crate) gitignore_file: bool,
	pub(crate) git_dir:        bool,
	pub(crate) repo_marker:    bool,
}

impl IgnoreEntryNames {
	pub(crate) fn record(&mut self, name: &OsStr, file_type: FileType) {
		if matches!(file_type, FileType::File | FileType::Symlink) {
			if name == OsStr::new(".ignore") {
				self.ignore_file = true;
			} else if name == OsStr::new(".gitignore") {
				self.gitignore_file = true;
			}
		}
		if name == OsStr::new(".git") {
			self.git_dir = true;
			self.repo_marker = true;
		} else if name == OsStr::new(".jj") {
			self.repo_marker = true;
		}
	}

	const fn has_relevant(self) -> bool {
		self.ignore_file || self.gitignore_file || self.git_dir || self.repo_marker
	}
}

fn has_repo_marker(dir: &Path) -> bool {
	dir.join(".git").exists() || dir.join(".jj").exists()
}

fn ignore_line_covers_root(
	matcher_root: &Path,
	source: &Path,
	line: &str,
	explicit_root: &Path,
) -> bool {
	let mut builder = ignore::gitignore::GitignoreBuilder::new(matcher_root);
	builder.add_line(Some(source.to_path_buf()), line).is_ok()
		&& builder.build().is_ok_and(|matcher| {
			matcher
				.matched_path_or_any_parents(explicit_root, true)
				.is_ignore()
		})
}

fn load_gitignore(
	matcher_root: &Path,
	file: &Path,
	explicit_root: Option<&Path>,
) -> Option<ignore::gitignore::Gitignore> {
	if !file.is_file() {
		return None;
	}
	let mut builder = ignore::gitignore::GitignoreBuilder::new(matcher_root);
	let _ = builder.add(file);
	let matcher = builder.build().ok().filter(|matcher| !matcher.is_empty())?;
	let Some(explicit_root) = explicit_root else {
		return Some(matcher);
	};
	if !matcher
		.matched_path_or_any_parents(explicit_root, true)
		.is_ignore()
	{
		return Some(matcher);
	}

	let handle = std::fs::File::open(file).ok()?;
	let mut filtered = ignore::gitignore::GitignoreBuilder::new(matcher_root);
	let source = Some(file.to_path_buf());
	for (index, line) in BufReader::new(handle).lines().enumerate() {
		let Ok(line) = line else {
			break;
		};
		let line = if index == 0 {
			line.trim_start_matches('\u{feff}')
		} else {
			line.as_str()
		};
		if ignore_line_covers_root(matcher_root, file, line, explicit_root) {
			continue;
		}
		let _ = filtered.add_line(source.clone(), line);
	}
	filtered.build().ok().filter(|matcher| !matcher.is_empty())
}

impl IgnoreState {
	fn build(dir: &Path, parent: Option<Arc<Self>>) -> Arc<Self> {
		let has_git = has_repo_marker(dir);
		let git_exclude = dir.join(".git/info/exclude");
		Self::new(
			parent,
			load_gitignore(dir, &dir.join(".ignore"), None),
			load_gitignore(dir, &dir.join(".gitignore"), None),
			if has_git {
				load_gitignore(dir, &git_exclude, None)
			} else {
				None
			},
			has_git,
		)
	}

	fn build_parent(dir: &Path, parent: Option<Arc<Self>>, explicit_root: &Path) -> Arc<Self> {
		let has_git = has_repo_marker(dir);
		let git_exclude = dir.join(".git/info/exclude");
		Self::new(
			parent,
			load_gitignore(dir, &dir.join(".ignore"), Some(explicit_root)),
			load_gitignore(dir, &dir.join(".gitignore"), Some(explicit_root)),
			if has_git {
				load_gitignore(dir, &git_exclude, Some(explicit_root))
			} else {
				None
			},
			has_git,
		)
	}

	fn build_from_entry_names(dir: &Path, parent: &Arc<Self>, names: IgnoreEntryNames) -> Arc<Self> {
		if !names.has_relevant() {
			return Arc::clone(parent);
		}
		let git_exclude = dir.join(".git/info/exclude");
		Self::new(
			Some(Arc::clone(parent)),
			if names.ignore_file {
				load_gitignore(dir, &dir.join(".ignore"), None)
			} else {
				None
			},
			if names.gitignore_file {
				load_gitignore(dir, &dir.join(".gitignore"), None)
			} else {
				None
			},
			if names.git_dir {
				load_gitignore(dir, &git_exclude, None)
			} else {
				None
			},
			names.repo_marker,
		)
	}

	fn new(
		parent: Option<Arc<Self>>,
		ignore_matcher: Option<ignore::gitignore::Gitignore>,
		gitignore_matcher: Option<ignore::gitignore::Gitignore>,
		git_exclude_matcher: Option<ignore::gitignore::Gitignore>,
		has_git: bool,
	) -> Arc<Self> {
		let parent_has_matchers = parent
			.as_ref()
			.is_some_and(|parent| parent.chain_has_matchers);
		let parent_has_git = parent.as_ref().is_some_and(|parent| parent.any_git);
		let has_matchers =
			ignore_matcher.is_some() || gitignore_matcher.is_some() || git_exclude_matcher.is_some();
		Arc::new(Self {
			parent,
			ignore_matcher,
			gitignore_matcher,
			git_exclude_matcher,
			has_git,
			chain_has_matchers: has_matchers || parent_has_matchers,
			any_git: has_git || parent_has_git,
		})
	}

	fn build_parents(root: &Path, use_gitignore: bool) -> Option<Arc<Self>> {
		if !use_gitignore {
			return None;
		}
		let mut ancestors = Vec::new();
		let mut current = root.parent();
		let mut repo_start = None;
		while let Some(path) = current {
			ancestors.push(path);
			if repo_start.is_none() && has_repo_marker(path) {
				repo_start = Some(ancestors.len() - 1);
			}
			current = path.parent();
		}

		let repo_start = repo_start?;
		let mut parent = None;
		for ancestor in ancestors[..=repo_start].iter().rev() {
			parent = Some(Self::build_parent(ancestor, parent, root));
		}
		parent
	}
}

impl FastIgnore {
	pub(crate) fn new(use_gitignore: bool) -> Self {
		let global = if use_gitignore {
			let (matcher, _err) = ignore::gitignore::Gitignore::global();
			if matcher.is_empty() {
				None
			} else {
				Some(matcher)
			}
		} else {
			None
		};
		Self { global, use_gitignore }
	}

	pub(crate) fn root_state(&self, root: &Path) -> Arc<IgnoreState> {
		IgnoreState::build(root, IgnoreState::build_parents(root, self.use_gitignore))
	}

	pub(crate) fn state_from_entries(
		&self,
		parent: &Arc<IgnoreState>,
		dir: &Path,
		names: IgnoreEntryNames,
		derive_ignore_from_entries: bool,
	) -> Arc<IgnoreState> {
		if self.use_gitignore && derive_ignore_from_entries {
			IgnoreState::build_from_entry_names(dir, parent, names)
		} else {
			Arc::clone(parent)
		}
	}

	pub(crate) fn is_ignored(&self, state: &Arc<IgnoreState>, path: &Path, is_dir: bool) -> bool {
		if !self.use_gitignore {
			return false;
		}

		let any_git = state.any_git;
		let global_matcher_applies = any_git && self.global.is_some();
		if !state.chain_has_matchers && !global_matcher_applies {
			return false;
		}

		let mut saw_git = false;
		let mut ignore_match = ignore::Match::None;
		let mut gitignore_match = ignore::Match::None;
		let mut git_exclude_match = ignore::Match::None;

		if state.chain_has_matchers {
			let mut current = Some(state.as_ref());
			while let Some(frame) = current {
				if ignore_match.is_none()
					&& let Some(matcher) = &frame.ignore_matcher
				{
					ignore_match = matcher.matched(path, is_dir);
				}
				if gitignore_match.is_none()
					&& let Some(matcher) = &frame.gitignore_matcher
				{
					gitignore_match = matcher.matched(path, is_dir);
				}
				if any_git
					&& !saw_git
					&& git_exclude_match.is_none()
					&& let Some(matcher) = &frame.git_exclude_matcher
				{
					git_exclude_match = matcher.matched(path, is_dir);
				}
				saw_git = saw_git || frame.has_git;
				current = frame.parent.as_deref();
			}
		}
		match ignore_match {
			ignore::Match::Ignore(_) => return true,
			ignore::Match::Whitelist(_) => return false,
			ignore::Match::None => {},
		}
		match gitignore_match {
			ignore::Match::Ignore(_) => return true,
			ignore::Match::Whitelist(_) => return false,
			ignore::Match::None => {},
		}
		match git_exclude_match {
			ignore::Match::Ignore(_) => return true,
			ignore::Match::Whitelist(_) => return false,
			ignore::Match::None => {},
		}
		if any_git && let Some(global) = &self.global {
			match global.matched(path, is_dir) {
				ignore::Match::Ignore(_) => return true,
				ignore::Match::Whitelist(_) => return false,
				ignore::Match::None => {},
			}
		}
		false
	}
}

// ── Platform module ──────────────────────────────────────────────────

#[cfg(target_os = "macos")]
mod platform {
	use std::{
		borrow::Cow,
		ffi::{CString, OsStr},
		io,
		mem::size_of,
		os::{fd::RawFd, unix::ffi::OsStrExt},
		path::Path,
	};

	use crate::entry::{RawDirEntry, ReadDirControl, ReadDirError, WalkError, FileType};
	use crate::policy::WalkDetail;
	use super::mtime_millis;

	pub const CHEAP_SIZE_HINTS: bool = false;

	const BUFFER_SIZE: usize = 256 * 1024;
	const VREG: u32 = 1;
	const VDIR: u32 = 2;
	const VLNK: u32 = 5;

	struct FdGuard(RawFd);

	impl Drop for FdGuard {
		fn drop(&mut self) {
			// SAFETY: `FdGuard` owns this file descriptor and closes it exactly once.
			unsafe { libc::close(self.0) };
		}
	}

	pub fn read_dir_entries<F, E>(
		path: &Path,
		detail: WalkDetail,
		buffer: &mut Vec<u8>,
		mut emit: F,
	) -> std::result::Result<ReadDirControl, ReadDirError<E>>
	where
		F: FnMut(RawDirEntry<'_>) -> std::result::Result<ReadDirControl, WalkError<E>>,
	{
		let fd = open_dir(path)?;
		let mut attrs = libc::attrlist {
			bitmapcount: libc::ATTR_BIT_MAP_COUNT,
			reserved:    0,
			commonattr:  libc::ATTR_CMN_NAME | libc::ATTR_CMN_OBJTYPE,
			volattr:     0,
			dirattr:     0,
			fileattr:    0,
			forkattr:    0,
		};
		if detail == WalkDetail::Full {
			attrs.commonattr |= libc::ATTR_CMN_MODTIME;
			attrs.fileattr |= libc::ATTR_FILE_DATALENGTH;
		}

		if buffer.len() != BUFFER_SIZE {
			buffer.resize(BUFFER_SIZE, 0);
		}
		loop {
			let count = unsafe {
				libc::getattrlistbulk(
					fd.0,
					std::ptr::addr_of_mut!(attrs).cast(),
					buffer.as_mut_ptr().cast(),
					buffer.len(),
					libc::FSOPT_NOFOLLOW as u64,
				)
			};
			if count == 0 {
				break;
			}
			if count < 0 {
				let err = io::Error::last_os_error();
				if err.kind() == io::ErrorKind::Interrupted {
					continue;
				}
				if is_unsupported_dir_scan(&err) {
					return read_dir_entries_std(path, detail, emit);
				}
				return Err(ReadDirError::Io(err));
			}

			let mut offset = 0usize;
			for _ in 0..count {
				if offset + size_of::<u32>() > buffer.len() {
					return Err(invalid_data("truncated getattrlistbulk record length").into());
				}
				let record_len = u32::from_ne_bytes(
					buffer[offset..offset + size_of::<u32>()]
						.try_into()
						.expect("slice length checked"),
				) as usize;
				if record_len < size_of::<u32>() || offset + record_len > buffer.len() {
					return Err(invalid_data("invalid getattrlistbulk record length").into());
				}
				let record = &buffer[offset..offset + record_len];
				if let Some(entry) = parse_record(record, detail)?
					&& emit(entry).map_err(ReadDirError::Walk)? == ReadDirControl::Stop
				{
					return Ok(ReadDirControl::Stop);
				}
				offset += record_len;
			}
		}
		Ok(ReadDirControl::Continue)
	}

	fn read_dir_entries_std<F, E>(
		path: &Path,
		detail: WalkDetail,
		mut emit: F,
	) -> std::result::Result<ReadDirControl, ReadDirError<E>>
	where
		F: FnMut(RawDirEntry<'_>) -> std::result::Result<ReadDirControl, WalkError<E>>,
	{
		let read_dir = std::fs::read_dir(path)?;
		for entry in read_dir {
			let entry = entry?;
			let file_type = match entry.file_type() {
				Ok(file_type) => file_type,
				Err(err) if is_skippable_entry_error(&err) => continue,
				Err(err) => return Err(err.into()),
			};
			let file_type = if file_type.is_symlink() {
				Some(FileType::Symlink)
			} else if file_type.is_dir() {
				Some(FileType::Dir)
			} else if file_type.is_file() {
				Some(FileType::File)
			} else {
				None
			};
			let Some(file_type) = file_type else {
				continue;
			};

			let mut mtime = None;
			let mut size = None;
			if detail == WalkDetail::Full {
				match std::fs::symlink_metadata(entry.path()) {
					Ok(metadata) => {
						if file_type == FileType::File {
							size = Some(metadata.len() as f64);
						}
						mtime = metadata
							.modified()
							.ok()
							.and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
							.map(|duration| duration.as_millis() as f64);
					},
					Err(err) if is_skippable_entry_error(&err) => continue,
					Err(err) => return Err(err.into()),
				}
			}

			let raw_entry =
				RawDirEntry { name: Cow::Owned(entry.file_name()), file_type, mtime, size };

			if emit(raw_entry).map_err(ReadDirError::Walk)? == ReadDirControl::Stop {
				return Ok(ReadDirControl::Stop);
			}
		}
		Ok(ReadDirControl::Continue)
	}

	fn open_dir(path: &Path) -> io::Result<FdGuard> {
		let path = CString::new(path.as_os_str().as_bytes())
			.map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains NUL"))?;
		let fd =
			unsafe { libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC) };
		if fd < 0 {
			Err(io::Error::last_os_error())
		} else {
			Ok(FdGuard(fd))
		}
	}

	fn parse_record(record: &[u8], detail: WalkDetail) -> io::Result<Option<RawDirEntry<'_>>> {
		let mut cursor = size_of::<u32>();
		let name_ref_start = cursor;
		let name_ref = read_value::<libc::attrreference_t>(record, &mut cursor)?;
		let obj_type = read_value::<u32>(record, &mut cursor)?;
		let (mtime, data_length) = if detail == WalkDetail::Full {
			let modified = read_value::<libc::timespec>(record, &mut cursor)?;
			let data_length = read_value::<u64>(record, &mut cursor)?;
			(mtime_millis(modified.tv_sec as i64, modified.tv_nsec as i64), Some(data_length))
		} else {
			(None, None)
		};

		let name_start = checked_attr_offset(name_ref_start, name_ref.attr_dataoffset)?;
		let name_len = name_ref.attr_length as usize;
		if name_len == 0 || name_start + name_len > record.len() {
			return Err(invalid_data("invalid getattrlistbulk name reference"));
		}
		let name_bytes = trim_nul(&record[name_start..name_start + name_len]);
		if name_bytes.is_empty() {
			return Ok(None);
		}

		let Some(file_type) = file_type_from_vtype(obj_type) else {
			return Ok(None);
		};
		let size = if file_type == FileType::File {
			data_length.map(|value| value as f64)
		} else {
			None
		};
		Ok(Some(RawDirEntry { name: OsStr::from_bytes(name_bytes).into(), file_type, mtime, size }))
	}

	fn read_value<T: Copy>(record: &[u8], cursor: &mut usize) -> io::Result<T> {
		let end = cursor.saturating_add(size_of::<T>());
		if end > record.len() {
			return Err(invalid_data("truncated getattrlistbulk attribute"));
		}
		let ptr = record[*cursor..end].as_ptr();
		*cursor = end;
		Ok(unsafe { std::ptr::read_unaligned(ptr.cast::<T>()) })
	}

	fn checked_attr_offset(base: usize, offset: i32) -> io::Result<usize> {
		if offset < 0 {
			return Err(invalid_data("negative getattrlistbulk attribute offset"));
		}
		base
			.checked_add(offset as usize)
			.ok_or_else(|| invalid_data("overflowing getattrlistbulk attribute offset"))
	}

	fn trim_nul(bytes: &[u8]) -> &[u8] {
		let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
		&bytes[..end]
	}

	const fn file_type_from_vtype(value: u32) -> Option<FileType> {
		match value {
			VREG => Some(FileType::File),
			VDIR => Some(FileType::Dir),
			VLNK => Some(FileType::Symlink),
			_ => None,
		}
	}

	fn is_unsupported_dir_scan(err: &io::Error) -> bool {
		matches!(err.raw_os_error(), Some(libc::ENOTSUP | libc::EINVAL))
	}

	fn is_skippable_entry_error(err: &io::Error) -> bool {
		matches!(err.kind(), io::ErrorKind::NotFound | io::ErrorKind::PermissionDenied)
	}

	fn invalid_data(message: &'static str) -> io::Error {
		io::Error::new(io::ErrorKind::InvalidData, message)
	}
}

pub(crate) mod platform {
	use std::{
		ffi::{CString, OsStr},
		io,
		mem::{size_of, zeroed},
		os::unix::ffi::OsStrExt,
		path::Path,
	};

	use crate::entry::{RawDirEntry, ReadDirControl, ReadDirError, WalkError, FileType};
	use crate::policy::WalkDetail;
	use super::mtime_millis;

	pub const CHEAP_SIZE_HINTS: bool = false;

	const BUFFER_SIZE: usize = 256 * 1024;
	const LINUX_DIRENT64_NAME_OFFSET: usize = 19;
	const STATX_TYPE: u32 = 0x0001;
	const STATX_SIZE: u32 = 0x0200;
	const STATX_MTIME: u32 = 0x0040;
	const STATX_BASIC_STATS: u32 = 0x07ff;

	#[repr(C)]
	#[derive(Clone, Copy)]
	struct StatxTimestamp {
		tv_sec:     i64,
		tv_nsec:    u32,
		__reserved: i32,
	}

	#[repr(C)]
	#[derive(Clone, Copy)]
	struct Statx {
		stx_mask:             u32,
		stx_blksize:          u32,
		stx_attributes:       u64,
		stx_nlink:            u32,
		stx_uid:              u32,
		stx_gid:              u32,
		stx_mode:             u16,
		__spare0:             [u16; 1],
		stx_ino:              u64,
		stx_size:             u64,
		stx_blocks:           u64,
		stx_attributes_mask:  u64,
		stx_atime:            StatxTimestamp,
		stx_btime:            StatxTimestamp,
		stx_ctime:            StatxTimestamp,
		stx_mtime:            StatxTimestamp,
		stx_rdev_major:       u32,
		stx_rdev_minor:       u32,
		stx_dev_major:        u32,
		stx_dev_minor:        u32,
		stx_mnt_id:           u64,
		stx_dio_mem_align:    u32,
		stx_dio_offset_align: u32,
		__spare3:             [u64; 12],
	}

	struct FdGuard(libc::c_int);

	impl Drop for FdGuard {
		fn drop(&mut self) {
			// SAFETY: `FdGuard` owns this file descriptor and closes it exactly once.
			unsafe { libc::close(self.0) };
		}
	}

	struct EntryStat {
		file_type: FileType,
		mtime:     Option<f64>,
		size:      Option<f64>,
	}

	pub fn read_dir_entries<F, E>(
		path: &Path,
		detail: WalkDetail,
		buffer: &mut Vec<u8>,
		mut emit: F,
	) -> std::result::Result<ReadDirControl, ReadDirError<E>>
	where
		F: FnMut(RawDirEntry<'_>) -> std::result::Result<ReadDirControl, WalkError<E>>,
	{
		let fd = open_dir(path)?;
		if buffer.len() != BUFFER_SIZE {
			buffer.resize(BUFFER_SIZE, 0);
		}
		loop {
			let read = unsafe {
				libc::syscall(
					libc::SYS_getdents64,
					fd.0,
					buffer.as_mut_ptr().cast::<libc::c_void>(),
					buffer.len(),
				)
			};
			if read == 0 {
				break;
			}
			if read < 0 {
				let err = io::Error::last_os_error();
				if err.kind() == io::ErrorKind::Interrupted {
					continue;
				}
				return Err(err.into());
			}

			let mut offset = 0usize;
			let read_len = read as usize;
			while offset < read_len {
				if offset + LINUX_DIRENT64_NAME_OFFSET > read_len {
					return Err(invalid_data("truncated getdents64 record").into());
				}
				let reclen = read_u16(&buffer[offset + 16..read_len])? as usize;
				if reclen < LINUX_DIRENT64_NAME_OFFSET || offset + reclen > read_len {
					return Err(invalid_data("invalid getdents64 record length").into());
				}
				let d_type = buffer[offset + 18];
				let name_bytes =
					trim_nul(&buffer[offset + LINUX_DIRENT64_NAME_OFFSET..offset + reclen]);
				offset += reclen;
				if name_bytes.is_empty() {
					continue;
				}

				let dtype_file_type = file_type_from_dtype(d_type);
				let stat = if detail == WalkDetail::Full || dtype_file_type.is_none() {
					match stat_entry(fd.0, name_bytes, detail) {
						Ok(Some(stat)) => Some(stat),
						Ok(None) => continue,
						Err(err) if is_skippable_entry_error(&err) => continue,
						Err(err) => return Err(err.into()),
					}
				} else {
					None
				};
				let file_type = stat
					.as_ref()
					.map_or(dtype_file_type, |stat| Some(stat.file_type));
				let Some(file_type) = file_type else {
					continue;
				};
				let entry = RawDirEntry {
					name: OsStr::from_bytes(name_bytes).into(),
					file_type,
					mtime: stat.as_ref().and_then(|stat| stat.mtime),
					size: stat.as_ref().and_then(|stat| stat.size),
				};
				if emit(entry).map_err(ReadDirError::Walk)? == ReadDirControl::Stop {
					return Ok(ReadDirControl::Stop);
				}
			}
		}
		Ok(ReadDirControl::Continue)
	}

	fn open_dir(path: &Path) -> io::Result<FdGuard> {
		let path = CString::new(path.as_os_str().as_bytes())
			.map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path contains NUL"))?;
		let fd =
			unsafe { libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC) };
		if fd < 0 {
			Err(io::Error::last_os_error())
		} else {
			Ok(FdGuard(fd))
		}
	}

	fn stat_entry(
		dirfd: libc::c_int,
		name: &[u8],
		detail: WalkDetail,
	) -> io::Result<Option<EntryStat>> {
		let name = CString::new(name)
			.map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "entry name contains NUL"))?;
		match statx_entry(dirfd, &name, detail) {
			Ok(value) => Ok(value),
			Err(err) if matches!(err.raw_os_error(), Some(libc::ENOSYS | libc::EINVAL)) => {
				fstatat_entry(dirfd, &name, detail)
			},
			Err(err) => Err(err),
		}
	}

	fn statx_entry(
		dirfd: libc::c_int,
		name: &CString,
		detail: WalkDetail,
	) -> io::Result<Option<EntryStat>> {
		let mut statx = unsafe { zeroed::<Statx>() };
		let mask = if detail == WalkDetail::Full {
			STATX_BASIC_STATS
		} else {
			STATX_TYPE
		};
		let rc = unsafe {
			libc::syscall(
				libc::SYS_statx,
				dirfd,
				name.as_ptr(),
				libc::AT_SYMLINK_NOFOLLOW | libc::AT_NO_AUTOMOUNT,
				mask,
				std::ptr::addr_of_mut!(statx),
			)
		};
		if rc != 0 {
			return Err(io::Error::last_os_error());
		}
		let Some(file_type) = file_type_from_mode(statx.stx_mode as libc::mode_t) else {
			return Ok(None);
		};
		let mtime = if detail == WalkDetail::Full && statx.stx_mask & STATX_MTIME != 0 {
			mtime_millis(statx.stx_mtime.tv_sec, i64::from(statx.stx_mtime.tv_nsec))
		} else {
			None
		};
		let size = if detail == WalkDetail::Full
			&& file_type == FileType::File
			&& statx.stx_mask & STATX_SIZE != 0
		{
			Some(statx.stx_size as f64)
		} else {
			None
		};
		Ok(Some(EntryStat { file_type, mtime, size }))
	}

	fn fstatat_entry(
		dirfd: libc::c_int,
		name: &CString,
		detail: WalkDetail,
	) -> io::Result<Option<EntryStat>> {
		let mut stat = unsafe { zeroed::<libc::stat>() };
		let rc = unsafe {
			libc::fstatat(
				dirfd,
				name.as_ptr(),
				std::ptr::addr_of_mut!(stat),
				libc::AT_SYMLINK_NOFOLLOW,
			)
		};
		if rc != 0 {
			return Err(io::Error::last_os_error());
		}
		let Some(file_type) = file_type_from_mode(stat.st_mode) else {
			return Ok(None);
		};
		let mtime = if detail == WalkDetail::Full {
			mtime_millis(stat.st_mtime, stat.st_mtime_nsec as i64)
		} else {
			None
		};
		let size = if detail == WalkDetail::Full && file_type == FileType::File {
			Some(stat.st_size as f64)
		} else {
			None
		};
		Ok(Some(EntryStat { file_type, mtime, size }))
	}

	fn read_u16(bytes: &[u8]) -> io::Result<u16> {
		if bytes.len() < size_of::<u16>() {
			return Err(invalid_data("truncated u16"));
		}
		Ok(u16::from_ne_bytes(
			bytes[..size_of::<u16>()]
				.try_into()
				.expect("slice length checked"),
		))
	}

	fn trim_nul(bytes: &[u8]) -> &[u8] {
		let end = bytes.iter().position(|b| *b == 0).unwrap_or(bytes.len());
		&bytes[..end]
	}

	const fn file_type_from_dtype(value: u8) -> Option<FileType> {
		match value {
			libc::DT_REG => Some(FileType::File),
			libc::DT_DIR => Some(FileType::Dir),
			libc::DT_LNK => Some(FileType::Symlink),
			_ => None,
		}
	}

	const fn file_type_from_mode(mode: libc::mode_t) -> Option<FileType> {
		match mode & libc::S_IFMT {
			libc::S_IFREG => Some(FileType::File),
			libc::S_IFDIR => Some(FileType::Dir),
			libc::S_IFLNK => Some(FileType::Symlink),
			_ => None,
		}
	}

	fn is_skippable_entry_error(err: &io::Error) -> bool {
		matches!(
			err.kind(),
			io::ErrorKind::NotFound | io::ErrorKind::PermissionDenied | io::ErrorKind::NotADirectory
		)
	}

	fn invalid_data(message: &'static str) -> io::Error {
		io::Error::new(io::ErrorKind::InvalidData, message)
	}
}

#[cfg(target_os = "windows")]
mod platform {
	use std::{
		ffi::OsString,
		io,
		os::windows::ffi::{OsStrExt, OsStringExt},
		path::Path,
	};

	use windows_sys::{
		Wdk::Storage::FileSystem::{
			FILE_ID_FULL_DIR_INFORMATION, FileIdFullDirectoryInformation, NtQueryDirectoryFile,
		},
		Win32::{
			Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE, STATUS_NO_MORE_FILES},
			Storage::FileSystem::{
				CreateFileW, FILE_ATTRIBUTE_DIRECTORY, FILE_ATTRIBUTE_REPARSE_POINT,
				FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT, FILE_LIST_DIRECTORY,
				FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
			},
			System::IO::IO_STATUS_BLOCK,
		},
	};

	use crate::entry::{RawDirEntry, ReadDirControl, ReadDirError, WalkError, FileType};
	use crate::policy::WalkDetail;
	use super::mtime_millis;

	pub const CHEAP_SIZE_HINTS: bool = true;

	const BUFFER_SIZE: usize = 256 * 1024;
	const WINDOWS_TICK: i64 = 10_000_000;
	const UNIX_EPOCH_AS_FILETIME: i64 = 116_444_736_000_000_000;

	struct HandleGuard(HANDLE);

	impl Drop for HandleGuard {
		fn drop(&mut self) {
			// SAFETY: `HandleGuard` owns this handle and closes it exactly once.
			unsafe { CloseHandle(self.0) };
		}
	}

	pub fn read_dir_entries<F, E>(
		path: &Path,
		detail: WalkDetail,
		buffer: &mut Vec<u8>,
		mut emit: F,
	) -> std::result::Result<ReadDirControl, ReadDirError<E>>
	where
		F: FnMut(RawDirEntry<'_>) -> std::result::Result<ReadDirControl, WalkError<E>>,
	{
		let handle = open_dir(path)?;
		if buffer.len() != BUFFER_SIZE {
			buffer.resize(BUFFER_SIZE, 0);
		}
		let mut restart = true;

		loop {
			let mut iosb = IO_STATUS_BLOCK::default();
			let status = unsafe {
				NtQueryDirectoryFile(
					handle.0,
					std::ptr::null_mut(),
					None,
					std::ptr::null(),
					std::ptr::addr_of_mut!(iosb),
					buffer.as_mut_ptr().cast(),
					buffer.len() as u32,
					FileIdFullDirectoryInformation,
					false,
					std::ptr::null(),
					restart,
				)
			};
			restart = false;
			if status == STATUS_NO_MORE_FILES {
				break;
			}
			if status < 0 {
				return Err(io::Error::from_raw_os_error(status).into());
			}

			let mut offset = 0usize;
			loop {
				if offset + std::mem::size_of::<FILE_ID_FULL_DIR_INFORMATION>() > buffer.len() {
					return Err(invalid_data("truncated NtQueryDirectoryFile record").into());
				}
				let info = unsafe {
					std::ptr::read_unaligned(
						buffer[offset..]
							.as_ptr()
							.cast::<FILE_ID_FULL_DIR_INFORMATION>(),
					)
				};
				let name_offset = offset + std::mem::offset_of!(FILE_ID_FULL_DIR_INFORMATION, FileName);
				let name_len = info.FileNameLength as usize;
				if !name_len.is_multiple_of(2) || name_offset + name_len > buffer.len() {
					return Err(invalid_data("invalid NtQueryDirectoryFile name length").into());
				}
				let name_units: Vec<u16> = buffer[name_offset..name_offset + name_len]
					.as_chunks::<2>()
					.0
					.iter()
					.map(|chunk| u16::from_ne_bytes([chunk[0], chunk[1]]))
					.collect();
				let name = OsString::from_wide(&name_units);
				let file_type = file_type_from_attributes(info.FileAttributes);
				let size = if detail == WalkDetail::Full && file_type == FileType::File {
					Some(info.EndOfFile.max(0) as f64)
				} else {
					None
				};
				let mtime = if detail == WalkDetail::Full {
					mtime_from_filetime(info.LastWriteTime)
				} else {
					None
				};
				let entry = RawDirEntry { name: name.into(), file_type, mtime, size };
				if emit(entry).map_err(ReadDirError::Walk)? == ReadDirControl::Stop {
					return Ok(ReadDirControl::Stop);
				}
				if info.NextEntryOffset == 0 {
					break;
				}
				offset = offset.saturating_add(info.NextEntryOffset as usize);
			}
		}
		Ok(ReadDirControl::Continue)
	}

	fn open_dir(path: &Path) -> io::Result<HandleGuard> {
		let mut path: Vec<u16> = path.as_os_str().encode_wide().collect();
		path.push(0);
		let handle = unsafe {
			CreateFileW(
				path.as_ptr(),
				FILE_LIST_DIRECTORY,
				FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
				std::ptr::null(),
				OPEN_EXISTING,
				FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
				std::ptr::null_mut(),
			)
		};
		if handle == INVALID_HANDLE_VALUE {
			Err(io::Error::last_os_error())
		} else {
			Ok(HandleGuard(handle))
		}
	}

	const fn file_type_from_attributes(attributes: u32) -> FileType {
		if attributes & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
			FileType::Symlink
		} else if attributes & FILE_ATTRIBUTE_DIRECTORY != 0 {
			FileType::Dir
		} else {
			FileType::File
		}
	}

	fn mtime_from_filetime(filetime: i64) -> Option<f64> {
		let ticks = filetime.checked_sub(UNIX_EPOCH_AS_FILETIME)?;
		let seconds = ticks / WINDOWS_TICK;
		let nanos = (ticks % WINDOWS_TICK) * 100;
		mtime_millis(seconds, nanos)
	}

	fn invalid_data(message: &'static str) -> io::Error {
		io::Error::new(io::ErrorKind::InvalidData, message)
	}
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
mod platform {
	use std::{borrow::Cow, io, path::Path};

	use crate::entry::{RawDirEntry, ReadDirControl, ReadDirError, WalkDetail, WalkError, FileType};

	pub const CHEAP_SIZE_HINTS: bool = false;

	pub fn read_dir_entries<F, E>(
		path: &Path,
		detail: WalkDetail,
		_buffer: &mut Vec<u8>,
		mut emit: F,
	) -> std::result::Result<ReadDirControl, ReadDirError<E>>
	where
		F: FnMut(RawDirEntry<'_>) -> std::result::Result<ReadDirControl, WalkError<E>>,
	{
		let read_dir = std::fs::read_dir(path)?;
		for entry in read_dir {
			let entry = entry?;
			let file_type_res = entry.file_type();
			let file_type = match file_type_res {
				Ok(ft) => ft,
				Err(err) if is_skippable_entry_error(&err) => continue,
				Err(err) => return Err(err.into()),
			};
			let custom_file_type = if file_type.is_symlink() {
				Some(FileType::Symlink)
			} else if file_type.is_dir() {
				Some(FileType::Dir)
			} else if file_type.is_file() {
				Some(FileType::File)
			} else {
				None
			};
			let Some(custom_file_type) = custom_file_type else {
				continue;
			};

			let mut mtime = None;
			let mut size = None;
			if detail == WalkDetail::Full {
				match std::fs::symlink_metadata(entry.path()) {
					Ok(metadata) => {
						if custom_file_type == FileType::File {
							size = Some(metadata.len() as f64);
						}
						mtime = metadata
							.modified()
							.ok()
							.and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
							.map(|duration| duration.as_millis() as f64);
					},
					Err(err) if is_skippable_entry_error(&err) => continue,
					Err(err) => return Err(err.into()),
				}
			}

			let raw_entry = RawDirEntry {
				name: Cow::Owned(entry.file_name()),
				file_type: custom_file_type,
				mtime,
				size,
			};

			if emit(raw_entry).map_err(ReadDirError::Walk)? == ReadDirControl::Stop {
				return Ok(ReadDirControl::Stop);
			}
		}
		Ok(ReadDirControl::Continue)
	}

	fn is_skippable_entry_error(err: &io::Error) -> bool {
		matches!(err.kind(), io::ErrorKind::NotFound | io::ErrorKind::PermissionDenied)
	}
}
