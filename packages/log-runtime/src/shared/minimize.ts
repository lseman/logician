// ── Bash output minimizer ─────────────────────────────────────────────────
// In-process output compressor for the bash tool. The engine is the
// rtk-ai/rtk-derived filter set vendored from oh-my-pi (pi-minimize crate):
// it rewrites chatty, highly structured output (cargo, git, test runners,
// package managers) into a compact summary before it reaches the model.
//
// Enabled by default. Kill switch: LOGICIAN_MINIMIZER=0|off|false. When the
// engine rewrites output, the caller persists the full original as an
// artifact and swaps the minimized text into the tool result.

import type { MinimizeResult } from "@logician/log-natives";
import { loadNative, type NativeModule } from "./native-addon.ts";

let nativeBroken = false;

function envDisabled(): boolean {
	const value = process.env.LOGICIAN_MINIMIZER;
	if (value === undefined) return false;
	const normalized = value.trim().toLowerCase();
	return normalized === "0" || normalized === "off" || normalized === "false";
}

/**
 * Minimize a captured bash command output.
 *
 * Resolves to `null` when minimization is disabled, the native addon is
 * unavailable (unbuilt — fails open, raw output passes through), no filter
 * matches the command, or the output passes through unchanged. Never throws:
 * the bash tool must degrade to its existing truncation behavior.
 */
export async function minimizeBashOutput(
	command: string,
	captured: string,
	exitCode: number,
): Promise<MinimizeResult | null> {
	if (nativeBroken || envDisabled() || captured.length === 0) return null;
	let native: NativeModule;
	try {
		native = await loadNative();
	} catch {
		nativeBroken = true;
		return null;
	}
	try {
		return (
			native.minimizeBashOutput(command, captured, exitCode, {
				enabled: true,
			}) ?? null
		);
	} catch {
		return null;
	}
}
