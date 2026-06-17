// ── Trust manager — resolves trust decisions for project directories ──────────
// Combines the trust store (persisted decisions) with the checker (resource
// detection) and provides a unified API for trust resolution.

import { TrustStore } from "./store.ts";
import { formatTrustPrompt, getTrustRequiringPaths, hasTrustRequiringProjectResources } from "./checker.ts";

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

export async function resolveTrust(options: TrustOptions): Promise<TrustResult> {
	const { cwd, defaultProjectTrust = "ask", hasUI, onSelectTrust } = options;
	const store = options.trustStore ?? new TrustStore();

	// If explicit CLI override or no trust-requiring resources, skip prompt
	if (!hasTrustRequiringProjectResources(cwd)) {
		return { trusted: true, remember: false };
	}

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

	// Save the decision
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
