// ── Claude Code plugin hook executor ─────────────────────────────────────────
// Loads, parses, matches, and executes plugin hooks. Handles shell/HTTP/prompt
// command types, JSON response parsing, and result merging.

import { spawn } from "node:child_process";
import { existsSync, promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { stripJsonComments } from "../../../tools/json-utils.ts";

// ── Types ─────────────────────────────────────────────────────────────────────

export type HookEventType =
	| "SessionStart"
	| "SessionEnd"
	| "Stop"
	| "UserPromptSubmit"
	| "PreToolUse"
	| "PostToolUse"
	| "PostToolUseFailure"
	| "PreCompact"
	| "PostCompact";

export interface HookCommand {
	type: "command" | "prompt" | "agent" | "http";
	command?: string;
	prompt?: string;
	agent?: string;
	url?: string;
	headers?: Record<string, string>;
	timeout?: number;
}

export interface HookDefinition {
	matcher?: string;
	hooks: HookCommand[];
}

export interface LoadedHook {
	pluginId: string;
	pluginName: string;
	pluginDir: string;
	eventType: HookEventType;
	definition: HookDefinition;
}

export interface HookContextMessage {
	plugin_id: string;
	plugin_name: string;
	matcher: string;
	content: string;
}

export interface HookExecutionResult {
	additional_contexts: string[];
	context_messages: HookContextMessage[];
	initial_user_message: string | null;
	watch_paths: string[];
	raw_output: string;
	decision?: "block" | "approve";
	reason?: string;
	permission_decision?: "allow" | "deny" | "ask";
	permission_reason?: string;
}

// ── Core functions ────────────────────────────────────────────────────────────

/** Load hooks from a plugin's manifest and hooks directory. */
export async function loadPluginHooks(
	pluginDir: string,
	pluginId: string,
): Promise<LoadedHook[]> {
	const merged: Record<string, HookDefinition[]> = {};
	const manifest = await readPluginManifest(pluginDir);
	await mergeManifestHooks(merged, pluginDir, manifest.hooks);
	const pluginName = await pluginNameFor(pluginDir, pluginId);
	const out: LoadedHook[] = [];
	for (const [eventType, defs] of Object.entries(merged)) {
		for (const definition of defs) {
			out.push({
				pluginId,
				pluginName,
				pluginDir,
				eventType: eventType as HookEventType,
				definition,
			});
		}
	}
	return out;
}

export async function readPluginManifest(
	pluginDir: string,
): Promise<Record<string, unknown>> {
	return readJson(path.join(pluginDir, ".claude-plugin", "plugin.json"));
}

export async function mergeManifestHooks(
	merged: Record<string, HookDefinition[]>,
	pluginDir: string,
	hooks: unknown,
): Promise<void> {
	if (typeof hooks === "string") {
		const hookPath = path.resolve(pluginDir, hooks);
		const hookJson = await readJson(hookPath);
		mergeHooks(merged, parseHooksDict(hookJson.hooks || hookJson));
		return;
	}
	mergeHooks(merged, parseHooksDict(hooks));
	try {
		const hookJson = await readJson(
			path.join(pluginDir, "hooks", "hooks.json"),
		);
		mergeHooks(merged, parseHooksDict(hookJson.hooks || hookJson));
	} catch (_e: unknown) {
		// No hooks directory — that's fine.
	}
}

export function parseHooksDict(
	data: unknown,
): Record<string, HookDefinition[]> {
	if (!data || typeof data !== "object" || Array.isArray(data)) return {};
	const out: Record<string, HookDefinition[]> = {};
	for (const [eventName, entries] of Object.entries(
		data as Record<string, unknown>,
	)) {
		if (!Array.isArray(entries)) continue;
		const defs: HookDefinition[] = [];
		for (const entry of entries) {
			if (!entry || typeof entry !== "object") continue;
			const raw = entry as Record<string, unknown>;
			const hooks = Array.isArray(raw.hooks)
				? raw.hooks
						.filter(
							(item): item is Record<string, unknown> =>
								Boolean(item) &&
								typeof item === "object" &&
								!Array.isArray(item),
						)
						.map(item => ({
							type: hookType(String(item.type || "command")),
							command: stringOrUndefined(item.command),
							prompt: stringOrUndefined(item.prompt),
							agent: stringOrUndefined(item.agent),
							url: stringOrUndefined(item.url),
							headers: isRecord(item.headers)
								? Object.fromEntries(
										Object.entries(item.headers).map(([k, v]) => [
											k,
											String(v),
										]),
									)
								: undefined,
							timeout:
								typeof item.timeout === "number" ? item.timeout : undefined,
						}))
				: [];
			if (hooks.length)
				defs.push({ matcher: stringOrUndefined(raw.matcher), hooks });
		}
		if (defs.length) out[eventName] = defs;
	}
	return out;
}

export async function executeLoadedHook(
	hook: LoadedHook,
	source: string,
	hookInput: string,
): Promise<HookExecutionResult> {
	const aggregate = emptyHookResult();
	for (const command of hook.definition.hooks) {
		const result = await executeCommand(command, hook, source, hookInput);
		if (result) mergeHookResult(aggregate, result);
	}
	return aggregate;
}

export async function executeCommand(
	command: HookCommand,
	hook: LoadedHook,
	source: string,
	hookInput: string,
): Promise<HookExecutionResult | null> {
	if (command.type === "prompt") {
		return command.prompt
			? { ...emptyHookResult(), additional_contexts: [command.prompt] }
			: null;
	}
	if (command.type === "http" && command.url) {
		const controller = new AbortController();
		const timer = setTimeout(
			() => controller.abort(),
			Math.max(1, command.timeout || 30) * 1000,
		);
		try {
			const response = await fetch(command.url, {
				headers: command.headers,
				signal: controller.signal,
			});
			return parseHookResponse(await response.text());
		} finally {
			clearTimeout(timer);
		}
	}
	if (command.type === "agent") return null;
	if (!command.command) return null;
	const timeout = Math.max(
		1,
		command.timeout ||
			(source === "startup"
				? envNumber("LOGICIAN_STARTUP_HOOK_TIMEOUT_MS", 30000) / 1000
				: 30),
	);
	const { stdout, stderr, code } = await runShellCommand(command.command, {
		cwd: hook.pluginDir,
		env: {
			...process.env,
			CLAUDE_PLUGIN_ROOT: hook.pluginDir,
		},
		input: hookInput,
		timeoutMs: timeout * 1000,
	});
	const parsed = parseHookResponse((stdout || stderr || "").trim());
	// Claude Code convention: exit code 2 = blocking error; stderr is the
	// reason fed back to the model.
	if (code === 2) {
		parsed.permission_decision = "deny";
		parsed.permission_reason =
			parsed.permission_reason || stderr.trim() || "Blocked by hook.";
	}
	return parsed;
}

export function parseHookResponse(rawOutput: string): HookExecutionResult {
	const result = { ...emptyHookResult(), raw_output: rawOutput };
	const trimmedOutput = rawOutput.trim();
	if (!trimmedOutput) return result;
	try {
		const data = JSON.parse(trimmedOutput);
		if (!isRecord(data)) return result;
		applyHookResponseObject(result, data);
		return result;
	} catch (_e: unknown) {
		// Some hooks emit JSONL control records.
	}

	const lines = trimmedOutput.split(/\r?\n/);
	const plainLines: string[] = [];
	let parsedJsonLine = false;
	for (const line of lines) {
		const trimmed = line.trim();
		if (!trimmed || !/^[{[]/.test(trimmed)) {
			plainLines.push(line);
			continue;
		}
		try {
			const data = JSON.parse(trimmed);
			if (!isRecord(data)) {
				plainLines.push(line);
				continue;
			}
			parsedJsonLine = true;
			applyHookResponseObject(result, data);
		} catch (_e: unknown) {
			plainLines.push(line);
		}
	}

	if (parsedJsonLine) {
		const plainText = plainLines.join("\n").trim();
		if (plainText) result.additional_contexts.push(plainText);
		return result;
	}

	result.additional_contexts.push(trimmedOutput);
	return result;
}

export function applyHookResponseObject(
	result: HookExecutionResult,
	data: Record<string, unknown>,
): void {
	const hookSpecific = data.hookSpecificOutput;
	if (isRecord(hookSpecific)) {
		const eventName = hookSpecific.hookEventName;
		pushContextValues(result, hookSpecific.additionalContext);
		pushContextValues(result, hookSpecific.additional_context);
		if (eventName === "SessionStart") {
			if (typeof hookSpecific.initialUserMessage === "string")
				result.initial_user_message = hookSpecific.initialUserMessage;
			if (Array.isArray(hookSpecific.watchPaths))
				result.watch_paths.push(
					...hookSpecific.watchPaths.filter(
						(p: unknown): p is string => typeof p === "string",
					),
				);
		}
		if (eventName === "PreToolUse") {
			const decision = hookSpecific.permissionDecision;
			if (decision === "allow" || decision === "deny" || decision === "ask") {
				result.permission_decision = decision;
				if (typeof hookSpecific.permissionDecisionReason === "string")
					result.permission_reason = hookSpecific.permissionDecisionReason;
			}
		}
		const decision = hookSpecific.decision;
		if (decision === "block" || decision === "approve") {
			result.decision = decision;
			if (typeof hookSpecific.reason === "string") {
				result.reason = hookSpecific.reason;
				if (decision === "block")
					result.additional_contexts.push(hookSpecific.reason);
			}
		}
		return;
	}

	pushContextValues(result, data.additional_context);
	pushContextValues(result, data.additionalContext);
	if (typeof data.initial_user_message === "string")
		result.initial_user_message = data.initial_user_message;
	if (typeof data.initialUserMessage === "string")
		result.initial_user_message = data.initialUserMessage;
	pushWatchPaths(result, data.watch_paths);
	pushWatchPaths(result, data.watchPaths);

	if (data.decision === "block" || data.decision === "approve") {
		result.decision = data.decision;
		if (typeof data.reason === "string") {
			result.reason = data.reason;
			if (data.decision === "block")
				result.additional_contexts.push(data.reason);
		}
	}
	if (
		(data.decision === "deny" ||
			data.permissionDecision === "deny" ||
			data.permission_decision === "deny") &&
		(typeof data.reason === "string" ||
			typeof data.permissionDecisionReason === "string" ||
			typeof data.permission_reason === "string")
	) {
		const reason = String(
			data.reason || data.permissionDecisionReason || data.permission_reason,
		);
		result.permission_decision = "deny";
		result.permission_reason = result.permission_reason || reason;
	}
}

function pushContextValues(result: HookExecutionResult, value: unknown): void {
	if (typeof value === "string") {
		const trimmed = value.trim();
		if (trimmed) result.additional_contexts.push(trimmed);
		return;
	}
	if (Array.isArray(value)) {
		for (const item of value) pushContextValues(result, item);
	}
}

function pushWatchPaths(result: HookExecutionResult, value: unknown): void {
	if (!Array.isArray(value)) return;
	result.watch_paths.push(
		...value.filter((item): item is string => typeof item === "string"),
	);
}

export function buildHookInput(
	eventType: HookEventType,
	payload: Record<string, unknown>,
): string {
	const data: Record<string, unknown> = {
		session_id: payload.session_id || "",
		transcript_path: payload.transcript_path || "",
		cwd: payload.cwd || process.cwd(),
		hook_event_name: eventType,
	};
	if (eventType === "SessionStart")
		data.source = String(payload.source || "startup").toLowerCase();
	if (eventType === "SessionEnd")
		data.reason = String(payload.reason || "other");
	if (eventType === "UserPromptSubmit" && payload.prompt !== undefined)
		data.prompt = payload.prompt;
	if (
		(eventType === "PreToolUse" ||
			eventType === "PostToolUse" ||
			eventType === "PostToolUseFailure") &&
		payload.tool_name !== undefined
	)
		data.tool_name = payload.tool_name;
	if (
		(eventType === "PreToolUse" ||
			eventType === "PostToolUse" ||
			eventType === "PostToolUseFailure") &&
		payload.tool_input !== undefined
	)
		data.tool_input = payload.tool_input;
	if (eventType === "PostToolUse" && payload.tool_response !== undefined)
		data.tool_response = payload.tool_response;
	if (eventType === "PostToolUseFailure" && payload.tool_error !== undefined)
		data.tool_error = payload.tool_error;
	if (eventType === "Stop")
		data.stop_hook_active = Boolean(payload.stop_hook_active);
	return JSON.stringify(data);
}

export function parseHookEventType(value: string): HookEventType | null {
	const clean = value.trim().toLowerCase();
	const events: HookEventType[] = [
		"SessionStart",
		"SessionEnd",
		"Stop",
		"UserPromptSubmit",
		"PreToolUse",
		"PostToolUse",
		"PostToolUseFailure",
		"PreCompact",
		"PostCompact",
	];
	return events.find(event => event.toLowerCase() === clean) || null;
}

export function mergeHooks(
	target: Record<string, HookDefinition[]>,
	source: Record<string, HookDefinition[]>,
): void {
	for (const [event, defs] of Object.entries(source))
		target[event] = [...(target[event] || []), ...defs];
}

export function mergeHookResult(
	target: HookExecutionResult,
	source: HookExecutionResult | null,
): void {
	if (!source) return;
	target.additional_contexts.push(...source.additional_contexts);
	target.context_messages.push(...source.context_messages);
	if (!target.initial_user_message && source.initial_user_message)
		target.initial_user_message = source.initial_user_message;
	target.watch_paths.push(...source.watch_paths);
	if (source.raw_output) target.raw_output = source.raw_output;
	if (source.decision === "block") {
		target.decision = "block";
		target.reason = source.reason;
	} else if (!target.decision && source.decision) {
		target.decision = source.decision;
		target.reason = source.reason;
	}
	if (source.permission_decision) {
		const rank = { deny: 2, ask: 1, allow: 0 } as const;
		const current = target.permission_decision;
		if (!current || rank[source.permission_decision] > rank[current]) {
			target.permission_decision = source.permission_decision;
			target.permission_reason = source.permission_reason;
		}
	}
}

export function emptyHookResult(): HookExecutionResult {
	return {
		additional_contexts: [],
		context_messages: [],
		initial_user_message: null,
		watch_paths: [],
		raw_output: "",
	};
}

export function withHookMetadata(
	hook: LoadedHook,
	result: HookExecutionResult | null,
): HookExecutionResult | null {
	if (!result) return null;
	if (!result.context_messages.length && result.additional_contexts.length) {
		result.context_messages = result.additional_contexts.map(content => ({
			plugin_id: hook.pluginId,
			plugin_name: hook.pluginName,
			matcher: hook.definition.matcher || "",
			content,
		}));
	}
	return result;
}

export function matcherMatches(
	pattern: string | undefined,
	source: string,
): boolean {
	if (!pattern) return true;
	const clean = pattern.trim();
	const sourceClean = source.trim();
	if (clean === "*" || !sourceClean) return true;
	const sourceParts = sourceClean.split("|").map(s => s.trim());
	try {
		const regex = new RegExp(clean, "i");
		if (regex.test(sourceClean)) return true;
		if (sourceParts.some(part => regex.test(part))) return true;
	} catch (_e: unknown) {
		// Fall back to legacy substring matching.
	}
	const lowerSource = sourceClean.toLowerCase();
	return clean
		.split("|")
		.map(p => p.trim())
		.some(
			p =>
				p === "*" ||
				lowerSource
					.split("|")
					.map(s => s.trim())
					.includes(p.toLowerCase()) ||
				lowerSource.includes(p.toLowerCase()),
		);
}

// ── Shell execution ──────────────────────────────────────────────────────────

export function runShellCommand(
	command: string,
	options: {
		cwd: string;
		env: NodeJS.ProcessEnv;
		input: string;
		timeoutMs: number;
	},
): Promise<{ stdout: string; stderr: string; code: number | null }> {
	return new Promise((resolve, reject) => {
		const child = spawn(command, {
			cwd: options.cwd,
			env: options.env,
			shell: true,
			stdio: ["pipe", "pipe", "pipe"],
		});
		let stdout = "";
		let stderr = "";
		const timer = setTimeout(
			() => {
				child.kill("SIGTERM");
				reject(new Error("hook command timed out"));
			},
			Math.max(1, options.timeoutMs),
		);

		child.stdout.setEncoding("utf8");
		child.stderr.setEncoding("utf8");
		child.stdout.on("data", chunk => {
			stdout += chunk;
			if (stdout.length > 1024 * 1024) stdout = stdout.slice(-1024 * 1024);
		});
		child.stderr.on("data", chunk => {
			stderr += chunk;
			if (stderr.length > 1024 * 1024) stderr = stderr.slice(-1024 * 1024);
		});
		child.on("error", error => {
			clearTimeout(timer);
			reject(error);
		});
		child.on("close", code => {
			clearTimeout(timer);
			resolve({ stdout, stderr, code });
		});
		child.stdin.end(options.input);
	});
}

// ── Shared utilities ─────────────────────────────────────────────────────────

export function resolvePluginsDir(): string {
	const override = (process.env.CLAUDE_CODE_PLUGIN_CACHE_DIR || "").trim();
	if (override)
		return path.resolve(os.homedir(), override.replace(/^~(?=$|\/)/, ""));
	const openclaude = path.join(
		os.homedir(),
		".claude",
		"openclaude",
		"plugins",
	);
	if (existsSync(openclaude)) return openclaude;
	return path.join(os.homedir(), ".claude", "plugins");
}

export function normalizeInstall(
	install: Record<string, unknown>,
): Record<string, unknown> {
	return {
		scope: (install.scope as string) || "user",
		installPath: (install.installPath as string) || "",
		version: (install.version as string) || "unknown",
		installedAt: (install.installedAt as string) || "",
		lastUpdated: (install.lastUpdated as string) || "",
		gitCommitSha: (install.gitCommitSha as string) || "",
		enabled: (install.enabled as boolean) !== false,
		projectPath: (install.projectPath as string) || "",
		dependencies: Array.isArray(install.dependencies)
			? install.dependencies
			: [],
	};
}

export async function readJson(file: string): Promise<Record<string, unknown>> {
	try {
		const content = await fs.readFile(file, "utf8");
		const stripped = stripJsonComments(content);
		return JSON.parse(stripped);
	} catch (_e: unknown) {
		return {};
	}
}

export async function isDir(p: string): Promise<boolean> {
	if (!p) return false;
	try {
		return (await fs.stat(p)).isDirectory();
	} catch (_e: unknown) {
		return false;
	}
}

export async function childDirNames(dir: string): Promise<string[]> {
	try {
		return (await fs.readdir(dir, { withFileTypes: true }))
			.filter(d => d.isDirectory())
			.map(d => d.name)
			.sort();
	} catch (_e: unknown) {
		return [];
	}
}

export async function markdownNames(dir: string): Promise<string[]> {
	const out: string[] = [];
	async function walk(base: string): Promise<void> {
		try {
			for (const entry of await fs.readdir(base, { withFileTypes: true })) {
				const full = path.join(base, entry.name);
				if (entry.isDirectory()) await walk(full);
				else if (entry.name.endsWith(".md")) out.push(path.relative(dir, full));
			}
		} catch {
			/* no-op */
		}
	}
	await walk(dir);
	return out.sort();
}

export async function findPluginManifest(root: string): Promise<string | null> {
	const direct = path.join(root, ".claude-plugin", "plugin.json");
	if (await fileExists(direct)) return direct;
	try {
		for (const entry of await fs.readdir(root, { withFileTypes: true })) {
			if (!entry.isDirectory() || entry.name.startsWith(".")) continue;
			const nested = path.join(
				root,
				entry.name,
				".claude-plugin",
				"plugin.json",
			);
			if (await fileExists(nested)) return nested;
		}
	} catch {
		/* no-op */
	}
	return null;
}

export async function fileExists(file: string): Promise<boolean> {
	try {
		await fs.access(file);
		return true;
	} catch (_e: unknown) {
		return false;
	}
}

export async function pluginNameFor(
	pluginDir: string,
	pluginId: string,
): Promise<string> {
	const manifest = await readPluginManifest(pluginDir);
	return typeof manifest.name === "string" && manifest.name
		? manifest.name
		: pluginId.split("@")[0];
}

export async function gitHead(cwd: string): Promise<string> {
	const { execFile } = await import("node:child_process");
	const { promisify } = await import("node:util");
	const execFileAsync = promisify(execFile);
	try {
		const { stdout } = await execFileAsync("git", ["rev-parse", "HEAD"], {
			cwd,
			timeout: 10_000,
		});
		const sha = String(stdout).trim();
		return /^[a-f0-9]{40}$/i.test(sha) ? sha : "";
	} catch (_e: unknown) {
		return "";
	}
}

export async function copyDir(src: string, dst: string): Promise<void> {
	await fs.mkdir(dst, { recursive: true });
	for (const entry of await fs.readdir(src, { withFileTypes: true })) {
		const from = path.join(src, entry.name);
		const to = path.join(dst, entry.name);
		if (entry.isDirectory()) await copyDir(from, to);
		else if (entry.isSymbolicLink()) {
			const target = await fs.readlink(from);
			if (!path.isAbsolute(target))
				await fs.symlink(target, to).catch(() => undefined);
		} else if (entry.isFile()) await fs.copyFile(from, to);
	}
}

export function hookType(value: string): HookCommand["type"] {
	return value === "prompt" || value === "agent" || value === "http"
		? value
		: "command";
}

export function stringOrUndefined(value: unknown): string | undefined {
	return typeof value === "string" ? value : undefined;
}

export function isRecord(value: unknown): value is Record<string, unknown> {
	return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function sanitize(value: string): string {
	return (
		value.replace(/[^a-zA-Z0-9\-_]/g, "-").replace(/^-+|-+$/g, "") || "unknown"
	);
}

export function nowIso(): string {
	return new Date().toISOString();
}

export function envNumber(name: string, fallback: number): number {
	const raw = Number(process.env[name] || "");
	return Number.isFinite(raw) && raw > 0 ? raw : fallback;
}
