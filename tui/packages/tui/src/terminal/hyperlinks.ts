// ── OSC 8 terminal hyperlinks ─────────────────────────────────────────────────
// Ported from repos/pi's terminal-image.ts capability detection, stripped down
// to the hyperlink-only subset (no image-protocol/cell-dimension concerns).

import { execSync } from "node:child_process";

/**
 * Checks whether the attached tmux client forwards OSC 8 hyperlinks to the
 * outer terminal. tmux only re-emits them when its `client_termfeatures` lists
 * `hyperlinks`, and strips them otherwise. On any error, falls back to `false`.
 */
function probeTmuxHyperlinks(): boolean {
	try {
		const termfeatures = execSync(
			"tmux display-message -p '#{client_termfeatures}'",
			{ encoding: "utf8", timeout: 250, stdio: ["ignore", "pipe", "ignore"] },
		);
		return termfeatures
			.split(",")
			.map(feature => feature.trim())
			.includes("hyperlinks");
	} catch {
		return false;
	}
}

/**
 * Detects whether the current terminal supports OSC 8 hyperlinks, based on
 * environment variables identifying the terminal emulator. Unknown terminals
 * default to `false` — OSC 8 renders as invisible escape bytes (the URL
 * disappears from the output entirely) on terminals that don't support it, so
 * being wrong in the "supports it" direction is worse than in the other.
 */
export function detectHyperlinkSupport(
	tmuxForwardsHyperlink: () => boolean = probeTmuxHyperlinks,
): boolean {
	const termProgram = process.env.TERM_PROGRAM?.toLowerCase() || "";
	const term = process.env.TERM?.toLowerCase() || "";

	// tmux only forwards OSC 8 when its client_termfeatures says so.
	if (process.env.TMUX || term.startsWith("tmux")) {
		return tmuxForwardsHyperlink();
	}

	// screen does not forward OSC 8 hyperlinks.
	if (term.startsWith("screen")) return false;

	if (process.env.KITTY_WINDOW_ID || termProgram === "kitty") return true;
	if (
		termProgram === "ghostty" ||
		term.includes("ghostty") ||
		process.env.GHOSTTY_RESOURCES_DIR
	)
		return true;
	if (process.env.WEZTERM_PANE || termProgram === "wezterm") return true;
	if (
		termProgram === "warpterminal" ||
		process.env.WARP_SESSION_ID ||
		process.env.WARP_TERMINAL_SESSION_UUID
	)
		return true;
	if (process.env.ITERM_SESSION_ID || termProgram === "iterm.app") return true;
	if (process.env.WT_SESSION) return true;
	if (termProgram === "vscode") return true;
	if (termProgram === "alacritty") return true;

	// Windows consoles and unrecognized terminals: be conservative.
	return false;
}

let cachedSupport: boolean | null = null;

/** Cached hyperlink-support check — env vars don't change mid-session. */
export function supportsHyperlinks(): boolean {
	if (cachedSupport === null) cachedSupport = detectHyperlinkSupport();
	return cachedSupport;
}

/** Test-only: clears the cached capability check. */
export function resetHyperlinkSupportCache(): void {
	cachedSupport = null;
}

/**
 * Wraps `text` in an OSC 8 hyperlink escape sequence pointing at `url`. The
 * text is rendered as clickable in terminals that support OSC 8 (Kitty,
 * WezTerm, iTerm2, Ghostty, Windows Terminal, VSCode, and others). Terminals
 * that don't support it typically pass the escape bytes through invisibly,
 * displaying only `text` — so callers should still show `url` some other way
 * (e.g. `text (url)`) when `supportsHyperlinks()` is false.
 */
export function hyperlink(text: string, url: string): string {
	return `\x1b]8;;${url}\x1b\\${text}\x1b]8;;\x1b\\`;
}
