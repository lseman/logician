//! Native N-API bindings for pi-ast, pi-edit, and filesystem search/discovery,
//! forked from oh-my-pi's `pi-natives` crate. Only the AST search/rewrite,
//! edit-engine, grep, glob, fuzzy-find, diff, and token-counting surface is
//! ported here (plus the small internal helpers they depend on); the
//! snapcompact frame renderer comes from the standalone `snapcompact` crate,
//! whose N-API exports register into this module on load. The rest of
//! upstream's binding surface (clipboard, audio, PDF, shell, VCS, ...) is not
//! included.
#![feature(alloc_error_hook)]
#![allow(
	dead_code,
	reason = "glob_util/iofs/js/task/crash_handler are ported wholesale from oh-my-pi, \
	where they're also shared by modules (glob, grep, fd, pdf, html, ...) this fork doesn't include"
)]

pub mod ast;
mod cancel;
mod crash_handler;
pub mod diff;
pub mod edit;
mod fd;
mod glob;
mod grep;
mod minimize;
mod glob_util;
mod iofs;
mod js;
mod task;
pub mod tokens;
mod utils;
mod utok;
#[cfg(test)]
mod testing;

// The standalone `snapcompact` crate self-registers its N-API exports at
// load time (via constructors); nothing else in Rust references it, so the
// cdylib link would drop its objects without this anchor keeping the crate
// alive.
#[allow(dead_code)]
const _SNAPCOMPACT_ANCHOR: fn(
	String,
	snapcompact::SnapcompactRenderOptions,
) -> snapcompact::task::Promise<napi::bindgen_prelude::Latin1String> =
	snapcompact::render_snapcompact_png;

pub use pi_ast::language;
