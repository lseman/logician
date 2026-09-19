#![feature(alloc_error_hook)]
//! Snapcompact bitmap PNG frame renderer.
//!
//! This crate contains the full snapcompact rendering pipeline:
//! - `cancel` — cooperative task cancellation
//! - `prof` — circular-buffer profiler for work scheduling
//! - `js` — N-API JavaScript string utilities (scratch arena for borrowed strings)
//! - `task` — blocking work scheduling on libuv's thread pool
//! - `snapcompact` — rasterization and PNG encoding

mod cancel;
mod crash_handler;
#[allow(
	dead_code,
	reason = "get_work_profile is the N-API export wired in pi-natives' cdylib; its query surface is dead in this standalone crate but kept whole to match the pi-natives twin"
)]
mod prof;
pub mod js;
pub mod task;
pub mod snapcompact;

#[cfg(test)]
mod testing;

pub use snapcompact::*;
