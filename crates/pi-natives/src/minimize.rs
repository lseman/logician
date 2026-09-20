//! N-API bindings for the bash-output minimizer (`pi-minimize` crate,
//! vendored from oh-my-pi's `pi-shell` minimizer).
//!
//! Logician executes shell commands with the platform shell (not an embedded
//! one), so the whole-buffer `apply` entry point is exposed directly: the JS
//! bash tool hands over the command, its captured output, and the exit code,
//! and receives back a minimized replacement — or `None` when the output
//! passes through unchanged.

use napi_derive::napi;

use pi_minimize::{MinimizerConfig, MinimizerOptions};

/// N-API opt-in handle for the minimizer.
#[napi(object)]
#[derive(Debug, Clone, Default)]
pub struct MinimizeOptions {
	/// Master switch. Absent / false = disabled.
	pub enabled: Option<bool>,
	/// Opt-in allowlist of program names (e.g. `"git"`). When empty or
	/// absent, all built-in filters are active.
	pub only: Option<Vec<String>>,
	/// Program names explicitly excluded from minimization.
	pub except: Option<Vec<String>>,
	/// Maximum captured bytes per command before the engine falls back to
	/// the raw, un-minimized output. Default 4 MiB.
	pub max_capture_bytes: Option<u32>,
	/// Source-outline level for `cat <source-file>` minimization. Accepts
	/// `"default"` (current behavior) or `"aggressive"` (strip function
	/// bodies).
	pub source_outline_level: Option<String>,
}

/// Telemetry for a single minimization.
///
/// Surfaced only when the minimizer actually rewrote the command's output.
#[napi(object)]
pub struct MinimizeResult {
	/// Dispatch label produced by the minimizer (e.g. `"git"`,
	/// `"pipeline:gradle"`, `"pipeline+builtin"`).
	pub filter: String,
	/// The minimized replacement text.
	pub text: String,
	/// The full original capture, before minimization.
	pub original_text: String,
	/// Captured byte length before minimization.
	pub input_bytes: u32,
	/// Byte length of the minimized text the consumer received.
	pub output_bytes: u32,
}

/// Minimize a captured shell command output.
///
/// Returns `None` when minimization is disabled, no filter matches the
/// command, or the output passes through unchanged (pipes, compounds,
/// unknown programs, too-large buffers). The command string is the exact
/// line handed to the shell; `captured` is the merged stdout/stderr buffer.
#[napi]
pub fn minimize_bash_output(
	command: String,
	captured: String,
	exit_code: i32,
	options: Option<MinimizeOptions>,
) -> Option<MinimizeResult> {
	let options = options.unwrap_or_default();
	let config = MinimizerConfig::from_options(&MinimizerOptions {
		enabled: options.enabled,
		settings_path: None,
		settings_hash: None,
		only: options.only,
		except: options.except,
		max_capture_bytes: options.max_capture_bytes,
		source_outline_level: options.source_outline_level,
		legacy_filters: None,
	});
	let output = pi_minimize::apply(&command, &captured, exit_code, &config);
	if !output.changed {
		return None;
	}
	Some(MinimizeResult {
		filter: output.filter.to_string(),
		text: output.text,
		original_text: output.original_text.unwrap_or_default(),
		input_bytes: output.input_bytes.try_into().unwrap_or(u32::MAX),
		output_bytes: output.output_bytes.try_into().unwrap_or(u32::MAX),
	})
}
