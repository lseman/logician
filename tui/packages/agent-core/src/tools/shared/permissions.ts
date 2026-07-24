// ── Permission system ────────────────────────────────────────────────────────
// Gate tool execution behind modes and allow/deny rules, Claude-Code style.
//
// Rule syntax (string):
//   "bash"            — every call of the bash tool
//   "bash(git *)"     — bash calls whose command matches the glob "git *"
//   "edit_file(src/*)"— edit_file calls whose primary arg matches "src/*"
//
// The "primary arg" used for the parenthesised pattern is the tool's most
// command-like argument: `command` (bash/git), else `path`/`file_path`, else
// the JSON-serialized args.
//
// Evaluation order: deny rules → allow rules (config + session "always") →
// mode policy. Modes:
//   acceptAll   — allow everything (legacy behavior, the default)
//   acceptEdits — read-only + file-edit tools allowed; the rest ask
//   ask         — read-only tools allowed; everything else asks
//   plan        — read-only tools allowed; everything else denied

import type { Tool, ToolCall } from "../../core/types.ts";

export type PermissionMode = "acceptAll" | "acceptEdits" | "ask" | "plan";

export interface PermissionRules {
	allow?: string[];
	deny?: string[];
}

export interface PermissionVerdict {
	decision: "allow" | "deny" | "ask";
	source: "rule" | "mode";
	reason?: string;
}

// Tools auto-approved in acceptEdits mode on top of read-only ones.
const EDIT_TOOLS = new Set(["edit_file", "write_file"]);

interface ParsedRule {
	tool: string;
	pattern?: RegExp;
	raw: string;
}

function parseRule(raw: string): ParsedRule | null {
	const match = /^\s*([A-Za-z0-9_-]+)\s*(?:\((.*)\))?\s*$/.exec(raw);
	if (!match) return null;
	const [, tool, glob] = match;
	return {
		tool,
		pattern: glob !== undefined ? globToRegExp(glob) : undefined,
		raw,
	};
}

function globToRegExp(glob: string): RegExp {
	const escaped = glob
		.trim()
		.replace(/[.+^${}()|[\]\\]/g, "\\$&")
		.replace(/\*/g, ".*")
		.replace(/\?/g, ".");
	return new RegExp(`^${escaped}`, "s");
}

function primaryArgStrings(args: Record<string, unknown>): string[] {
	if (Array.isArray(args.commands)) {
		const commands = args.commands.flatMap((entry) => {
			if (!entry || typeof entry !== "object") return [];
			const command = (entry as Record<string, unknown>).command;
			return typeof command === "string" ? [command] : [];
		});
		if (commands.length > 0) return commands;
	}
	for (const key of ["command", "path", "file_path", "url", "query"]) {
		if (typeof args[key] === "string") return [args[key] as string];
	}
	try {
		return [JSON.stringify(args)];
	} catch {
		return [""];
	}
}

/** The most command-like argument of a call, for display and compatibility. */
export function primaryArgString(args: Record<string, unknown>): string {
	return primaryArgStrings(args).join("\n");
}

export class PermissionManager {
	private mode: PermissionMode;
	private allowRules: ParsedRule[];
	private denyRules: ParsedRule[];
	// "always allow" decisions made interactively this session.
	private sessionAllow: ParsedRule[] = [];

	constructor(opts?: { mode?: PermissionMode; rules?: PermissionRules }) {
		this.mode = opts?.mode ?? "acceptAll";
		this.allowRules = (opts?.rules?.allow ?? [])
			.map(parseRule)
			.filter((r): r is ParsedRule => r !== null);
		this.denyRules = (opts?.rules?.deny ?? [])
			.map(parseRule)
			.filter((r): r is ParsedRule => r !== null);
	}

	getMode(): PermissionMode {
		return this.mode;
	}

	setMode(mode: PermissionMode): void {
		this.mode = mode;
	}

	/** Persist an interactive "always allow" for the rest of the session. */
	addSessionAllow(toolName: string): void {
		const rule = parseRule(toolName);
		if (rule) this.sessionAllow.push(rule);
	}

	evaluate(
		call: ToolCall,
		args: Record<string, unknown>,
		tool?: Tool,
	): PermissionVerdict {
		const commandArgs = primaryArgStrings(args);
		const denied = commandArgs
			.map((arg) => this.matchRules(this.denyRules, call.name, arg))
			.find((rule) => rule !== undefined);
		if (denied) {
			return {
				decision: "deny",
				source: "rule",
				reason: `denied by rule "${denied.raw}"`,
			};
		}
		const allowRules = [...this.allowRules, ...this.sessionAllow];
		const allowed = commandArgs.every(
			(arg) => this.matchRules(allowRules, call.name, arg) !== undefined,
		);
		if (allowed) return { decision: "allow", source: "rule" };

		const readOnly = tool?.readOnly === true;
		switch (this.mode) {
			case "acceptAll":
				return { decision: "allow", source: "mode" };
			case "acceptEdits":
				return readOnly || EDIT_TOOLS.has(call.name)
					? { decision: "allow", source: "mode" }
					: { decision: "ask", source: "mode" };
			case "ask":
				return readOnly
					? { decision: "allow", source: "mode" }
					: { decision: "ask", source: "mode" };
			case "plan":
				return readOnly
					? { decision: "allow", source: "mode" }
					: {
							decision: "deny",
							source: "mode",
							reason:
								"Plan mode is active: present a plan instead of executing. " +
								"Only read-only tools are available.",
						};
		}
	}

	private matchRules(
		rules: ParsedRule[],
		toolName: string,
		arg: string,
	): ParsedRule | undefined {
		return rules.find(
			(rule) =>
				rule.tool === toolName &&
				(rule.pattern === undefined || rule.pattern.test(arg)),
		);
	}
}
