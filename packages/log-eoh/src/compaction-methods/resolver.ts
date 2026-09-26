// ── Compaction Method Resolver ───────────────────────────────────────────────
// Routes compaction requests to the correct strategy based on:
// 1. User-configured method order (from settings)
// 2. Model/provider capabilities (server compaction availability)
// 3. Available engines (snapcompact, shake, etc.)

import type { CompactionMethod } from "./index.ts";
import {
	DEFAULT_COMPACTION_METHOD_ORDER,
	isCompactionMethod,
} from "./index.ts";

/** Runtime capabilities of the current session. */
export interface CompactionCapabilities {
	/** Whether the provider supports server-native compaction. */
	serverCompactionAvailable: boolean;
	/** Whether snapcompact engine is available. */
	snapcompactAvailable: boolean;
	/** Whether an LLM is available for soft/handoff compaction. */
	llmAvailable: boolean;
}

/** Default capabilities when nothing is known. */
const DEFAULT_CAPABILITIES: CompactionCapabilities = {
	serverCompactionAvailable: false,
	snapcompactAvailable: false,
	llmAvailable: true,
};

/** Minimal compaction settings interface (compatible with AgentBridgeOptions). */
export interface CompactionSettings {
	/** Whether compaction is enabled. */
	enabled?: boolean;
	/** Ordered list of methods to try (first match wins). */
	methodOrder?: CompactionMethod[];
	/** Single method override (shorthand for [method]). */
	method?: CompactionMethod;
	/** Compaction strategy default; only "snapcompact" is selectable here today. */
	mode?: "snapcompact";
	/** Reserve tokens budget for compaction. */
	reserveTokens?: number;
	/** Keep this many recent tokens in context after compaction. */
	keepRecentTokens?: number;
}

/** Resolve which compaction method should be used given settings and capabilities. */
export function resolveCompactionMethod(
	settings: CompactionSettings,
	capabilities: CompactionCapabilities = DEFAULT_CAPABILITIES,
): CompactionMethod | null {
	// Handle mode field from AgentBridgeOptions (backward compat)
	if (settings.mode === "snapcompact") return "snapcompact";

	// Build the effective method order
	const configuredOrder = settings.method
		? [settings.method]
		: settings.methodOrder ?? DEFAULT_COMPACTION_METHOD_ORDER;

	// Filter out methods that aren't available
	for (const method of configuredOrder) {
		if (!isCompactionMethod(method)) continue;

		// Check if this method is available given current capabilities
		if (isMethodAvailable(method, capabilities)) {
			return method;
		}
	}

	return null; // No available method
}

/** Check if a specific compaction method is available given current capabilities. */
export function isMethodAvailable(
	method: CompactionMethod,
	capabilities: CompactionCapabilities = DEFAULT_CAPABILITIES,
): boolean {
	switch (method) {
		case "remote":
			return capabilities.serverCompactionAvailable;
		case "snapcompact":
			return capabilities.snapcompactAvailable;
		case "handoff":
		case "soft":
			return capabilities.llmAvailable;
		case "shake":
			// Shake is always available (no external dependency)
			return true;
	}
}

/** Get the fallback chain for a given method configuration. */
export function getFallbackChain(
	settings: CompactionSettings,
	capabilities: CompactionCapabilities = DEFAULT_CAPABILITIES,
): CompactionMethod[] {
	const configuredOrder = settings.method
		? [settings.method]
		: settings.methodOrder ?? DEFAULT_COMPACTION_METHOD_ORDER;

	return configuredOrder.filter(m => isCompactionMethod(m) && isMethodAvailable(m, capabilities));
}

/** Check if server-native compaction should be used. */
export function shouldUseServerCompaction(
	settings: CompactionSettings,
	capabilities: CompactionCapabilities = DEFAULT_CAPABILITIES,
): boolean {
	const method = resolveCompactionMethod(settings, capabilities);
	return method === "remote";
}
