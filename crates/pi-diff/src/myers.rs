//! Core Myers O(ND) diff algorithm and change-building helpers.
//!
//! Replicates jsdiff's default (non-`oneChangePerToken`, no timeout / `maxEditLength`)
//! execution path so the resulting run structure is identical.

use std::{collections::HashMap, hash::Hash, rc::Rc};

/// A run of tokens sharing one edit classification, in forward order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
	/// Number of tokens in this run.
	pub count:   u32,
	/// True when this run exists only in the new input.
	pub added:   bool,
	/// True when this run exists only in the old input.
	pub removed: bool,
}

struct Component {
	count:   usize,
	added:   bool,
	removed: bool,
	prev:    Option<Rc<Self>>,
}

/// Frontier state for one diagonal: furthest old-position reached plus the
/// component chain that got there.
struct PathState {
	old_pos: isize,
	last:    Option<Rc<Component>>,
}

/// Extend `path` along its diagonal while tokens match, recording the common
/// run. Returns the new-token position (mirrors jsdiff `extractCommon`).
fn extract_common(path: &mut PathState, new: &[u32], old: &[u32], diagonal: isize) -> isize {
	let new_len = new.len() as isize;
	let old_len = old.len() as isize;
	let mut old_pos = path.old_pos;
	let mut new_pos = old_pos - diagonal;
	let mut common = 0usize;
	while new_pos + 1 < new_len
		&& old_pos + 1 < old_len
		&& old[(old_pos + 1) as usize] == new[(new_pos + 1) as usize]
	{
		new_pos += 1;
		old_pos += 1;
		common += 1;
	}
	if common > 0 {
		path.last = Some(Rc::new(Component {
			count:   common,
			added:   false,
			removed: false,
			prev:    path.last.take(),
		}));
	}
	path.old_pos = old_pos;
	new_pos
}

/// Branch from `path` with one added or removed token (mirrors jsdiff
/// `addToPath`, which merges into the previous component when the edit kind
/// repeats).
fn add_to_path(path: &PathState, added: bool, removed: bool, old_pos_inc: isize) -> PathState {
	match &path.last {
		Some(last) if last.added == added && last.removed == removed => PathState {
			old_pos: path.old_pos + old_pos_inc,
			last:    Some(Rc::new(Component {
				count: last.count + 1,
				added,
				removed,
				prev: last.prev.clone(),
			})),
		},
		_ => PathState {
			old_pos: path.old_pos + old_pos_inc,
			last:    Some(Rc::new(Component { count: 1, added, removed, prev: path.last.clone() })),
		},
	}
}

/// Convert the winning component chain into forward-ordered runs.
fn build_runs(last: Option<Rc<Component>>) -> Vec<Run> {
	let mut runs = Vec::new();
	let mut cursor = last.as_deref();
	while let Some(component) = cursor {
		runs.push(Run {
			count:   component.count as u32,
			added:   component.added,
			removed: component.removed,
		});
		cursor = component.prev.as_deref();
	}
	runs.reverse();
	runs
}

/// Myers O(ND) diff over interned token ids, replicating jsdiff's default
/// (non-`oneChangePerToken`, no timeout / `maxEditLength`) execution path so
/// the resulting run structure is identical.
pub fn myers_diff(old: &[u32], new: &[u32]) -> Vec<Run> {
	let old_len = old.len() as isize;
	let new_len = new.len() as isize;
	let max_edit = old_len + new_len;
	let offset = max_edit + 1;
	let mut best: Vec<Option<PathState>> = Vec::new();
	best.resize_with((2 * max_edit + 3) as usize, || None);

	// Seed edit length 0: the content may start with common tokens.
	let mut seed = PathState { old_pos: -1, last: None };
	let seed_new_pos = extract_common(&mut seed, new, old, 0);
	if seed.old_pos + 1 >= old_len && seed_new_pos + 1 >= new_len {
		return build_runs(seed.last);
	}
	best[offset as usize] = Some(seed);

	let mut min_diagonal = isize::MIN;
	let mut max_diagonal = isize::MAX;
	let mut edit_length: isize = 1;
	while edit_length <= max_edit {
		let mut diagonal = min_diagonal.max(-edit_length);
		while diagonal <= max_diagonal.min(edit_length) {
			let idx = (diagonal + offset) as usize;
			let remove_path = best[idx - 1].take();
			let add_path_old_pos = best[idx + 1].as_ref().map(|path| path.old_pos);
			let can_add = add_path_old_pos.is_some_and(|old_pos| {
				let add_new_pos = old_pos - diagonal;
				add_new_pos >= 0 && add_new_pos < new_len
			});
			let can_remove = remove_path
				.as_ref()
				.is_some_and(|path| path.old_pos + 1 < old_len);
			if !can_add && !can_remove {
				best[idx] = None;
				diagonal += 2;
				continue;
			}

			// Branch from the prior path whose old-text position is furthest
			// along, preferring the insertion path on ties (jsdiff order).
			let mut base_path = if !can_remove
				|| (can_add
					&& remove_path.as_ref().is_some_and(|path| {
						add_path_old_pos.is_some_and(|add_old| path.old_pos < add_old)
					})) {
				add_to_path(
					best[idx + 1]
						.as_ref()
						.expect("canAdd implies a live addPath"),
					true,
					false,
					0,
				)
			} else {
				add_to_path(
					remove_path
						.as_ref()
						.expect("canRemove implies a live removePath"),
					false,
					true,
					1,
				)
			};
			let new_pos = extract_common(&mut base_path, new, old, diagonal);
			if base_path.old_pos + 1 >= old_len && new_pos + 1 >= new_len {
				return build_runs(base_path.last);
			}
			if base_path.old_pos + 1 >= old_len {
				max_diagonal = max_diagonal.min(diagonal - 1);
			}
			if new_pos + 1 >= new_len {
				min_diagonal = min_diagonal.max(diagonal + 1);
			}
			best[idx] = Some(base_path);
			diagonal += 2;
		}
		edit_length += 1;
	}
	unreachable!("Myers diff terminates within oldLen + newLen edits")
}

/// Intern tokens as dense ids under exact equality.
pub fn intern<T: Eq + Hash + Copy>(old: &[T], new: &[T]) -> (Vec<u32>, Vec<u32>) {
	fn assign<T: Eq + Hash + Copy>(ids: &mut HashMap<T, u32>, token: T) -> u32 {
		let next = ids.len() as u32;
		*ids.entry(token).or_insert(next)
	}
	let mut ids = HashMap::with_capacity(old.len() + new.len());
	let old_ids = old
		.iter()
		.copied()
		.map(|token| assign(&mut ids, token))
		.collect();
	let new_ids = new
		.iter()
		.copied()
		.map(|token| assign(&mut ids, token))
		.collect();
	(old_ids, new_ids)
}

/// A joined change value and its edit classification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Change<V> {
	/// Joined token value for this run.
	pub value:   V,
	/// Number of tokens in this run.
	pub count:   u32,
	/// True when this run exists only in the new input.
	pub added:   bool,
	/// True when this run exists only in the old input.
	pub removed: bool,
}

/// Map runs back to values, taking common-run text from `new_tokens`.
pub fn build_changes<'a, T: ?Sized, V>(
	runs: &[Run],
	old_tokens: &[&'a T],
	new_tokens: &[&'a T],
	join: impl Fn(&[&'a T]) -> V,
) -> Vec<Change<V>> {
	let mut old_pos = 0usize;
	let mut new_pos = 0usize;
	runs
		.iter()
		.map(|run| {
			let count = run.count as usize;
			let value = if run.removed {
				let value = join(&old_tokens[old_pos..old_pos + count]);
				old_pos += count;
				value
			} else {
				let value = join(&new_tokens[new_pos..new_pos + count]);
				new_pos += count;
				if !run.added {
					old_pos += count;
				}
				value
			};
			Change { value, count: run.count, added: run.added, removed: run.removed }
		})
		.collect()
}
