import { mkdirSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import type { WebSearchConfig } from "@logician/log-core";
import { DEFAULT_SEARXNG_URL } from "../../capabilities/tools/default-tools.ts";

/** Parse a numeric environment setting, returning undefined when absent/invalid. */
export function envNumber(name: string): number | undefined {
	const raw = process.env[name];
	if (!raw) return undefined;
	const value = Number(raw);
	return Number.isFinite(value) ? value : undefined;
}

export interface BridgeEndpoint {
	baseUrl: string;
	model: string;
}

export function createHookTranscriptPath(
	cwd: string,
	sessionId: string,
): string {
	const safeCwd = cwd
		.replace(/[^a-zA-Z0-9._-]+/g, "_")
		.replace(/^_+|_+$/g, "")
		.slice(0, 96);
	const dir = path.join(
		os.homedir(),
		".logician",
		"tui",
		"sessions",
		safeCwd || "workspace",
	);
	const transcriptPath = path.join(dir, `${sessionId}.jsonl`);
	try {
		mkdirSync(dir, { recursive: true });
		writeFileSync(
			transcriptPath,
			`${JSON.stringify({
				type: "session",
				timestamp: new Date().toISOString(),
				session_id: sessionId,
				cwd,
			})}\n`,
			"utf8",
		);
	} catch (_error: unknown) {
		return "";
	}
	return transcriptPath;
}

export function eventLogPathFor(transcriptPath: string): string | undefined {
	if (!transcriptPath) return undefined;
	return transcriptPath.replace(/\.jsonl$/, ".events.jsonl");
}

export function buildPluginRuntimeEnv(
	endpoint: BridgeEndpoint,
): NodeJS.ProcessEnv {
	const model = endpoint.model?.trim() || "";
	const baseUrl = endpoint.baseUrl?.trim().replace(/\/+$/, "");
	const env: NodeJS.ProcessEnv = {};
	if (!baseUrl) return env;

	env.CLAUDE_MEM_MODEL = model;
	env.CLAUDE_MEM_OPENROUTER_MODEL = model;
	env.CLAUDE_MEM_TIER_ROUTING_ENABLED = "false";
	env.CLAUDE_MEM_TIER_SIMPLE_MODEL = "";
	env.CLAUDE_MEM_TIER_SUMMARY_MODEL = "";
	env.CLAUDE_MEM_TIER_FAST_MODEL = "";
	env.CLAUDE_MEM_TIER_SMART_MODEL = "";
	env.CLAUDE_MEM_PROVIDER = "openrouter";
	env.CLAUDE_MEM_OPENROUTER_BASE_URL = baseUrl;
	env.OPENROUTER_BASE_URL = baseUrl;
	env.CLAUDE_MEM_OPENROUTER_API_KEY =
		process.env.CLAUDE_MEM_OPENROUTER_API_KEY ||
		process.env.OPENROUTER_API_KEY ||
		"logician-local";
	env.OPENROUTER_API_KEY = env.CLAUDE_MEM_OPENROUTER_API_KEY;
	return env;
}

export function resolveWebSearchConfig(): WebSearchConfig {
	return {
		baseUrl: process.env.LOGICIAN_SEARXNG_URL?.trim() || DEFAULT_SEARXNG_URL,
		maxResults: envNumber("LOGICIAN_SEARXNG_MAX_RESULTS"),
	};
}
