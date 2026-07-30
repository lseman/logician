// ── Slash command definitions ─────────────────────────────────────────────────
// ~30 commands with rich context: categories, argument hints, examples.

export type SlashDispatch = "local" | "bridge" | "state" | "quit";
export type SlashCommandSource = "builtin" | "extension" | "skill";
export type SlashCommandCategory =
	| "help"
	| "session"
	| "agent"
	| "context"
	| "skills"
	| "reasoning"
	| "display"
	| "permissions"
	| "shortcuts"
	| "loop"
	| "misc";

/** Category ordering for grouped popup display. */
export const CATEGORY_ORDER: Readonly<SlashCommandCategory[]> = [
	"help",
	"session",
	"agent",
	"context",
	"skills",
	"reasoning",
	"display",
	"permissions",
	"shortcuts",
	"loop",
	"misc",
];

export interface SlashCommandDef {
	command: string;
	usage?: string;
	description: string;
	dispatch: SlashDispatch;
	acceptsArgs: boolean;
	/** One-line argument hint for the popup (e.g. "<number>", "[mode]"). */
	argHint?: string;
	/** Command category for grouped popup display. */
	category?: SlashCommandCategory;
	/** Usage examples (shown in popup when command selected). */
	examples?: string[];
	/** Source attribution (builtin / extension / skill). */
	source?: SlashCommandSource;
	handler?: (args: string) => string | undefined;
	bridgeHandler?: (args: string) => void;
}

// ── Command spec factory ──────────────────────────────────────────────────────

function cmd(
	command: string,
	description: string,
	dispatch: SlashDispatch = "bridge",
	acceptsArgs = false,
	extra?: Partial<SlashCommandDef>,
	handler?: (args: string) => string | undefined,
	bridgeHandler?: (args: string) => void,
): SlashCommandDef {
	return {
		command,
		usage: command,
		description,
		dispatch,
		acceptsArgs,
		category: "misc",
		source: "builtin",
		...extra,
		handler,
		bridgeHandler,
	};
}

// ── Full command list ─────────────────────────────────────────────────────────

export function createSlashCommands(
	_bridge: {
		sendSlash: (raw: string) => void;
		cancel: () => void;
		reset: () => void;
	},
	localHandlers: Record<string, (...args: unknown[]) => unknown>,
): SlashCommandDef[] {
	const commands: SlashCommandDef[] = [
		// ── Help & info ──────────────────────────────────────────────────────
		cmd(
			"/help",
			"Show all available commands",
			"local",
			true,
			{
				category: "help",
				argHint: "[topic]",
				examples: ["/help", "/help session"],
			},
			(args) => formatSlashHelp(commands, args),
		),
		cmd("/?", "Alias for /help", "local", false, { category: "help" }, () =>
			formatSlashHelp(commands, ""),
		),

		// ── Session management ───────────────────────────────────────────────
		cmd("/new", "Start a new session", "bridge", false, {
			category: "session",
		}),
		cmd("/sessions", "List previous sessions", "local", true, {
			category: "session",
			argHint: "[filter]",
			examples: ["/sessions", "/sessions 2024"],
		}),
		cmd("/save", "Save current session", "local", false, {
			category: "session",
		}),

		cmd("/rename", "Rename current session", "local", true, {
			category: "session",
			argHint: "<name>",
			examples: ["/rename My analysis"],
		}),
		cmd(
			"/name",
			"Set short human name on current session",
			"local",
			true,
			{
				category: "session",
				argHint: "<name>",
				examples: ["/name my-debug-run"],
			},
			(...args) =>
				String(localHandlers.nameSession?.(...args) ?? "No active session."),
		),
		cmd(
			"/bookmark",
			"Add a label/bookmark to current position",
			"local",
			true,
			{
				category: "session",
				argHint: "<label> [note]",
				examples: ["/bookmark breakthrough found the root cause"],
			},
			(...args) =>
				String(localHandlers.bookmark?.(...args) ?? "No active session."),
		),
		cmd(
			"/bookmarks",
			"List bookmarks in current session",
			"local",
			false,
			{ category: "session" },
			() => String(localHandlers.listBookmarks?.() ?? "No bookmarks."),
		),
		cmd(
			"/session",
			"Open the interactive session manager",
			"local",
			false,
			{ category: "session", examples: ["/session"] },
			() => {
				localHandlers.openSessionManager?.();
				return undefined;
			},
		),

		// ── Agent control ────────────────────────────────────────────────────
		cmd("/status", "Show runtime state snapshot", "state", false, {
			category: "agent",
		}),
		cmd("/steer-now", "Process queued steering immediately", "bridge", false, {
			category: "agent",
			examples: ["/steer-now"],
		}),
		cmd("/queue", "Show queued steering and follow-up messages", "bridge", false, {
			category: "agent",
		}),
		cmd("/queue-drop", "Remove one queued message by number", "bridge", true, {
			category: "agent",
			argHint: "<number>",
			examples: ["/queue-drop 2"],
		}),
		cmd("/queue-clear", "Remove all queued messages", "bridge", false, {
			category: "agent",
		}),
		cmd("/reload", "Reload config and agents", "bridge", false, {
			category: "agent",
		}),

		// ── Context & memory ─────────────────────────────────────────────────
		cmd(
			"/context",
			"Show session/data context",
			"local",
			false,
			{ category: "context" },
			() => {
				return (
					(localHandlers.getContext?.() as string | undefined) ||
					"No context available."
				);
			},
		),
		cmd("/compact", "Summarize older conversation history", "bridge", false, {
			category: "context",
		}),
		cmd("/fork", "Fork the conversation into a branch", "local", false, {
			category: "context",
			examples: ["/fork"],
		}),
		cmd(
			"/branch-summary",
			"Summarize active branch into parent",
			"local",
			false,
			{ category: "context" },
		),
		cmd("/discard-branch", "Discard the active branch", "local", false, {
			category: "context",
		}),

		// ── Skills ───────────────────────────────────────────────────────────
		cmd("/plugins", "Manage installed plugins", "local", true, {
			category: "skills",
			argHint: "[list|install|remove]",
			examples: ["/plugins", "/plugins install my-ext"],
		}),
		cmd("/mcp", "Manage MCP servers", "local", true, {
			category: "skills",
			argHint: "[list|add|remove]",
			examples: ["/mcp", "/mcp add http://localhost:3000"],
		}),
		cmd("/theme", "Select a color theme", "local", true, {
			category: "display",
			argHint: "<theme>",
			examples: ["/theme dark", "/theme github-dark", "/theme light"],
		}),
		cmd(
			"/model",
			"Open the model selector",
			"local",
			false,
			{ category: "display", examples: ["/model"] },
			() => {
				localHandlers.openModelSelector?.();
				return undefined;
			},
		),

		// ── Reasoning ────────────────────────────────────────────────────────
		cmd("/reasoner", "Select reasoning mode", "local", true, {
			category: "reasoning",
			argHint: "<mode>",
			examples: ["/reasoner none", "/reasoner tot", "/reasoner reflexion"],
		}),
		cmd(
			"/eoh",
			"Evolve a self-evaluating Python heuristic",
			"local",
			true,
			{
				category: "reasoning",
				argHint: "<heuristic.py> [generations] | status | stop | best | reset",
				examples: [
					"/eoh heuristic.py",
					"/eoh heuristic.py 10",
					"/eoh status",
					"/eoh stop",
				],
			},
			(args: string) =>
				String(localHandlers.eoh?.(args) ?? "EoH unavailable."),
		),

		// ── Display ──────────────────────────────────────────────────────────
		cmd(
			"/thinking",
			"Set thinking level",
			"local",
			true,
			{
				category: "display",
				argHint: "<level>",
				examples: ["/thinking off", "/thinking high", "/thinking xhigh"],
			},
			(args: string) => {
				const level = args.trim().toLowerCase();
				const valid = ["off", "low", "medium", "high", "xhigh"];
				if (valid.includes(level)) {
					localHandlers.setThinking?.(level);
					return `Thinking level: ${level}`;
				}
				return `Valid levels: ${valid.join(", ")}`;
			},
		),
		cmd(
			"/mode",
			"Cycle thinking display mode",
			"local",
			false,
			{ category: "display", examples: ["/mode"] },
			() => {
				localHandlers.cycleThinking?.();
				return "Thinking mode cycled.";
			},
		),
		cmd(
			"/thinking-steps",
			"Set thinking display mode",
			"local",
			true,
			{
				category: "display",
				argHint: "<mode>",
				examples: ["/thinking-steps collapsed", "/thinking-steps expanded"],
			},
			(args: string) => {
				const mode = args.trim().toLowerCase();
				if (["collapsed", "summary", "expanded"].includes(mode)) {
					localHandlers.setThinkingMode?.(mode);
					return `Thinking display: ${mode}`;
				}
				return "Valid modes: collapsed, summary, expanded";
			},
		),
		cmd(
			"/clear",
			"Clear visible transcript only",
			"local",
			false,
			{ category: "display", examples: ["/clear"] },
			() => {
				localHandlers.clear?.();
				return "Transcript cleared.";
			},
		),
		cmd(
			"/ask-preview",
			"Preview the interactive ask-user popup",
			"local",
			false,
			{ category: "display", examples: ["/ask-preview"] },
			() => {
				localHandlers.askPreview?.();
				return undefined;
			},
		),

		// ── Permissions ──────────────────────────────────────────────────────
		cmd(
			"/permissions",
			"Set permission mode",
			"local",
			true,
			{
				category: "permissions",
				argHint: "<mode>",
				examples: ["/permissions acceptAll", "/permissions ask"],
			},
			(args: string) => {
				const valid = ["acceptAll", "acceptEdits", "ask", "plan"];
				const mode = valid.find(
					(m) => m.toLowerCase() === args.trim().toLowerCase(),
				);
				if (mode) {
					localHandlers.setPermissionMode?.(mode);
					return `Permission mode: ${mode}`;
				}
				return `Valid modes: ${valid.join(", ")} (current: ${
					localHandlers.getPermissionMode?.() ?? "acceptAll"
				})`;
			},
		),
		cmd(
			"/plan",
			"Toggle plan mode",
			"local",
			false,
			{ category: "permissions", examples: ["/plan"] },
			() => String(localHandlers.togglePlanMode?.() ?? "Plan mode unavailable"),
		),
		cmd(
			"/rewind",
			"Rewind to previous checkpoint",
			"local",
			false,
			{ category: "permissions", examples: ["/rewind"] },
			() => String(localHandlers.rewind?.() ?? "Nothing to rewind."),
		),

		// ── Shortcuts ────────────────────────────────────────────────────────
		cmd("/q", "Quick quit", "quit", false, {
			category: "shortcuts",
			examples: ["/q"],
		}),
		cmd("/quit", "Exit TUI", "quit", false, { category: "shortcuts" }),
		cmd("/exit", "Alias for /quit", "quit", false, { category: "shortcuts" }),

		// ── Loop ─────────────────────────────────────────────────────────────
		cmd("/loop", "Run a prompt repeatedly", "local", true, {
			category: "loop",
			argHint: "<count|duration> <prompt>",
			examples: ["/loop 5m check deploy", "/loop 10 build"],
		}),
		cmd(
			"/goal",
			"Set a completion condition; agent loops until met",
			"local",
			true,
			{
				category: "loop",
				argHint: "<condition | clear>",
				examples: [
					"/goal all tests in test/auth pass and lint is clean",
					"/goal implement feature X or stop after 20 turns",
					"/goal",
					"/goal clear",
				],
			},
		),

		// ── Misc ─────────────────────────────────────────────────────────────
		cmd(
			"/notifications",
			"Show recent notification history",
			"local",
			false,
			{ category: "misc", examples: ["/notifications"] },
			() =>
				String(
					localHandlers.notifications?.() ?? "No notifications yet.",
				),
		),
		cmd(
			"/version",
			"Show TUI and bridge version",
			"local",
			false,
			{ category: "misc" },
			() =>
				String(localHandlers.version?.() ?? "Logician version unavailable."),
		),
		// ── Sandbox ──────────────────────────────────────────────────────
		cmd(
			"/sandbox",
			"Run a command in a Bubblewrap-isolated sandbox (Linux only)",
			"local",
			true,
			{
				category: "misc",
				argHint: "<command> | profile <none|code|file|dev|full> | status",
				examples: [
					"/sandbox echo hello",
					"/sandbox code echo secure",
					"/sandbox profile code",
					"/sandbox status",
				],
			},
		),
		cmd(
			"/sandbox-cycle",
			"Cycle the default sandbox mode (off/code/full)",
			"local",
			false,
			{ category: "misc", examples: ["/sandbox-cycle"] },
			() =>
				String(
					localHandlers.cycleSandboxMode?.() ?? "Sandbox cycle unavailable.",
				),
		),
		cmd(
			"/execution-policy-cycle",
			"Cycle execution policy (autonomous ↔ minimal)",
			"local",
			false,
			{ category: "misc", examples: ["/execution-policy-cycle"] },
			() =>
				String(
					localHandlers.cycleExecutionProfile?.() ??
						"Execution policy cycle unavailable.",
				),
		),
		cmd(
			"/inference-mode-cycle",
			"Cycle inference mode (thinking/instruct variants)",
			"local",
			false,
			{ category: "misc", examples: ["/inference-mode-cycle"] },
			() =>
				String(
					localHandlers.cycleInferenceMode?.() ??
						"Inference mode cycle unavailable.",
				),
		),

		// ── Settings ─────────────────────────────────────────────────────
		cmd(
			"/settings",
			"View or modify runtime settings",
			"local",
			true,
			{
				category: "misc",
				argHint: "[subcommand <value>]",
				examples: [
					"/settings",
					"/settings thinking high",
					"/settings model claude-sonnet-4",
					"/settings model-cycle",
					"/settings temp 0.7",
					"/settings max-tokens 8192",
					"/settings max-iterations 20",
					"/settings loop-detection on",
					"/settings guards on",
					"/settings compaction on",
					"/settings permissions ask",
				],
			},
			(args) => {
				return String(
					localHandlers.settings?.(args) ?? "Settings are unavailable.",
				);
			},
		),

		// ── RTK Proxy ────────────────────────────────────────────────────
		cmd(
			"/rtk",
			"Toggle RTK CLI proxy (compresses bash output 60-90%)",
			"local",
			false,
			{
				category: "misc",
				examples: ["/rtk"],
			},
			() => {
				const state = localHandlers.toggleRtkProxy?.();
				return state ? "RTK proxy: on" : "RTK proxy: off";
			},
		),
	];
	return commands;
}

export function formatSlashHelp(
	commands: SlashCommandDef[],
	topic = "",
): string {
	const normalized = topic.trim().toLowerCase().replace(/^\//, "");
	const matches = normalized
		? commands.filter(
				(command) =>
					command.category === normalized ||
					command.command.slice(1).toLowerCase().includes(normalized) ||
					command.description.toLowerCase().includes(normalized),
			)
		: commands;
	if (matches.length === 0) {
		return `No commands match "${topic.trim()}". Use /help to list everything.`;
	}
	const groups = groupByCategory(matches);
	const lines = [
		`Available commands (${matches.length})`,
		"Type / to browse, ↑/↓ to select, Tab to complete, Enter to run.",
	];
	for (const category of CATEGORY_ORDER) {
		const entries = groups.get(category);
		if (!entries?.length) continue;
		lines.push("", `${category.toUpperCase()}`);
		for (const command of entries) {
			const usage = `${command.command}${command.argHint ? ` ${command.argHint}` : ""}`;
			lines.push(`  ${usage.padEnd(32)} ${command.description}`);
		}
	}
	return lines.join("\n");
}

// ── Category grouping helper ──────────────────────────────────────────────────

/** Group commands by category, preserving CATEGORY_ORDER. Only non-empty categories included. */
export function groupByCategory(
	commands: SlashCommandDef[],
): Map<SlashCommandCategory, SlashCommandDef[]> {
	const groups = new Map<SlashCommandCategory, SlashCommandDef[]>();
	for (const cat of CATEGORY_ORDER) groups.set(cat, []);
	for (const cmd of commands) {
		const cat = cmd.category ?? "misc";
		groups.get(cat)?.push(cmd);
	}
	// Remove empty categories
	for (const [cat, cmds] of groups) {
		if (cmds.length === 0) groups.delete(cat);
	}
	return groups;
}

// ── Fuzzy filter helper ───────────────────────────────────────────────────────

export function filterSlashCommands(
	commands: SlashCommandDef[],
	query: string,
	limit: number = Number.POSITIVE_INFINITY,
): SlashCommandDef[] {
	if (!query || query.length <= 1) return commands.slice(0, limit);

	const lowerQuery = query.toLowerCase().trim();
	const scored = commands
		.map((cmd, idx) => {
			const cmdName = cmd.command.toLowerCase();
			const desc = cmd.description.toLowerCase();
			let score = -1;

			// Exact match on command name
			if (cmdName === lowerQuery) score = 3000 - idx;
			// Prefix match
			else if (cmdName.startsWith(lowerQuery))
				score = 2500 - (cmdName.length - lowerQuery.length) - idx;
			// Contains match
			else if (cmdName.includes(lowerQuery))
				score = 2000 - cmdName.indexOf(lowerQuery) * 8 - idx;
			// Subsequence match
			else if (subsequenceMatch(lowerQuery, cmdName)) score = 1500 - idx;
			// Description match
			else if (desc.includes(lowerQuery)) score = 800 - idx;
			// Word match in description
			else if (desc.split(/\s+/).some((w) => w.startsWith(lowerQuery)))
				score = 1000 - idx;

			return score >= 0 ? { cmd, score } : null;
		})
		.filter(Boolean) as { cmd: SlashCommandDef; score: number }[];

	scored.sort((a, b) => b.score - a.score);
	return scored.map((s) => s.cmd).slice(0, limit);
}

function subsequenceMatch(query: string, text: string): boolean {
	let qi = 0;
	for (let ti = 0; ti < text.length && qi < query.length; ti++) {
		if (text[ti] === query[qi]) qi++;
	}
	return qi === query.length;
}
