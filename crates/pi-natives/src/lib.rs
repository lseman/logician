//! Native N-API bindings for pi-ast, pi-edit, and filesystem search/discovery,
//! forked from oh-my-pi's `pi-natives` crate. Only the AST search/rewrite,
//! edit-engine, grep, glob, fuzzy-find, diff, snapcompact, and token-counting
//! surface is ported here (plus the small internal helpers they depend on);
//! the rest of upstream's binding surface (clipboard, audio, PDF, shell,
//! VCS, ...) is not included.
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
pub mod fd;
pub mod glob;
pub mod grep;
mod glob_util;
mod iofs;
mod js;
mod prof;
#[path = "../../snapcompact/src/snapcompact.rs"]
pub mod snapcompact;
mod task;
pub mod tokens;
mod utils;
mod utok;
#[cfg(test)]
mod testing;

pub use pi_ast::language;
