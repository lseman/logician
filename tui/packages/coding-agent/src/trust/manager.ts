// ── Trust manager — resolves trust decisions for project directories ──────────
// Combines the trust store (persisted decisions) with the checker (resource
// detection) and provides a unified API for trust resolution.

import { TrustStore } from "./store.ts";
import { formatTrustPrompt, getTrustRequiringPaths } from "./checker.ts";

export type DefaultProjectTrust = "ask" | "always" | "never";

export interface TrustOptions {
	cwd: string;
	trustStore?: TrustStore;
	defaultProjectTrust?: DefaultProjectTrust;
	hasUI: boolean;
	onSelectTrust?: (prompt: string, paths: string[]) => Promise<"trust" | "trust-parent" | "session-only" | "deny" | "deny-session">;
}

export interface TrustResult {
	trusted: boolean;
	remember: boolean;
}

/** Information needed to render the trust prompt overlay in the TUI. */
export interface TrustInfo {
	/** The directory being asked about. */
	cwd: string;
	/** Trust-requiring resource paths found under cwd. */
	paths: string[];
	/** Whether a decision already exists (overlay not needed). */
	preDecided: boolean;
	/** The pre-decided result if available. */
	preDecidedResult?: TrustResult;
	/** Whether to show the overlay at all (only when defaultProjectTrust is 'ask'). */
	needsDecision: boolean;
}

// ── Apply a trust choice and persist the result ──────────────────────────────

export function applyTrustChoice(
	store: TrustStore,
	choice: "trust" | "trust-parent" | "session-only" | "deny" | "deny-session",
	cwd: string,
): TrustResult {
	switch (choice) {
		case "trust":
			store.set(cwd, true);
			return { trusted: true, remember: true };
		case "trust-parent": {
			const parent = cwd.replace(/\/+$/, "").split("/").slice(0, -1).join("/");
			if (parent) {
				store.setMany([
					{ path: parent, decision: true },
					{ path: cwd, decision: null },
				]);
			}
			return { trusted: true, remember: true };
		}
		case "session-only":
			return { trusted: true, remember: false };
		case "deny":
			store.set(cwd, false);
			return { trusted: false, remember: true };
		case "deny-session":
			return { trusted: false, remember: false };
	}
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Resolve a trust decision. If a persisted decision or default policy
 * exists, returns immediately. Otherwise, if hasUI is true and
 * onSelectTrust is provided, prompts the user.
 */
export async function resolveTrust(options: TrustOptions): Promise<TrustResult> {
	const { cwd, defaultProjectTrust = "ask", hasUI, onSelectTrust } = options;
	const store = options.trustStore ?? new TrustStore();

	// Check persisted decision (walks up tree)
	const decision = store.get(cwd);
	if (decision.decision !== null) {
		return { trusted: decision.decision, remember: true };
	}

	// Apply default policy
	switch (defaultProjectTrust) {
		case "always":
			return { trusted: true, remember: false };
		case "never":
			return { trusted: false, remember: true };
		case "ask":
			break;
	}

	// No UI available — default to not trusting
	if (!hasUI) {
		return { trusted: false, remember: false };
	}

	// Ask the user
	const paths = getTrustRequiringPaths(cwd);
	const prompt = formatTrustPrompt(cwd, paths);

	let choice: "trust" | "trust-parent" | "session-only" | "deny" | "deny-session";
	if (onSelectTrust) {
		choice = await onSelectTrust(prompt, paths);
	} else {
		// Fallback: just trust for the session
		choice = "session-only";
	}

	return applyTrustChoice(store, choice, cwd);
}

/**
 * Detect trust-requiring resources and return info without prompting.
 * The TUI uses this to decide whether to show the trust overlay.
 */
export function resolveTrustInfo(
	cwd: string,
	defaultProjectTrust: DefaultProjectTrust = "ask",
): TrustInfo {
	// Check persisted decision (walks up tree)
	const store = new TrustStore();
	const decision = store.get(cwd);
	if (decision.decision !== null) {
		return {
			cwd,
			paths: getTrustRequiringPaths(cwd),
			preDecided: true,
			preDecidedResult: { trusted: decision.decision, remember: true },
			needsDecision: false,
		};
	}

	// Apply default policy
	if (defaultProjectTrust === "always") {
		return {
			cwd,
			paths: getTrustRequiringPaths(cwd),
			preDecided: true,
			preDecidedResult: { trusted: true, remember: false },
			needsDecision: false,
		};
	}
	if (defaultProjectTrust === "never") {
		return {
			cwd,
			paths: getTrustRequiringPaths(cwd),
			preDecided: true,
			preDecidedResult: { trusted: false, remember: true },
			needsDecision: false,
		};
	}

	// defaultProjectTrust === "ask" — needs user decision
	return {
		cwd,
		paths: getTrustRequiringPaths(cwd),
		preDecided: false,
		needsDecision: true,
	};
}
