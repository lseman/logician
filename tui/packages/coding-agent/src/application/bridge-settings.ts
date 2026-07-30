import { readFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import type { AgentHarness } from "@logician/agent-core";

export interface UserSettings {
	compaction?: {
		enabled?: boolean;
		reserveTokens?: number;
		keepRecentTokens?: number;
	};
	maxParallelAgents?: number;
	subagents?: {
		maxParallelAgents?: number;
	};
	[key: string]: unknown;
}

export function loadUserSettings(): UserSettings {
	const settingsPath = path.join(os.homedir(), ".logician", "settings.json");
	let raw: string;
	try {
		raw = readFileSync(settingsPath, "utf8");
	} catch (error: unknown) {
		if ((error as NodeJS.ErrnoException)?.code !== "ENOENT") {
			console.error("[settings] failed to read settings.json:", error);
		}
		return {};
	}
	try {
		const parsed = JSON.parse(raw) as Record<string, unknown>;
		return typeof parsed === "object" && parsed !== null
			? (parsed as UserSettings)
			: {};
	} catch (error: unknown) {
		console.error("[settings] settings.json is not valid JSON:", error);
		return {};
	}
}

export function applyCompactionSettings(
	harness: AgentHarness,
	settings: UserSettings,
): void {
	const compaction = settings.compaction;
	if (!compaction) return;

	const compactionSettings: {
		reserveTokens?: number;
		keepRecentTokens?: number;
	} = {};
	if (compaction.reserveTokens !== undefined && compaction.reserveTokens > 0) {
		compactionSettings.reserveTokens = compaction.reserveTokens;
	}
	if (
		compaction.keepRecentTokens !== undefined &&
		compaction.keepRecentTokens > 0
	) {
		compactionSettings.keepRecentTokens = compaction.keepRecentTokens;
	}
	if (Object.keys(compactionSettings).length > 0) {
		harness.setAutoCompactionSettings(compactionSettings);
	}
	if (compaction.enabled === true) {
		harness.enableAutoCompaction(true);
	}
}
