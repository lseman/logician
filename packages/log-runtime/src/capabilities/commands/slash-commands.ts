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
	/** Literal first-level subcommands offered by inline autocomplete. */
	subcommands?: string[];
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
			args => formatSlashHelp(commands, args),
		),
		cmd("/?", "Alias for /help", "local", false, { category: "help" }, () =>
			formatSlashHelp(commands, ""),
		),

		// ── Session management ───────────────────────────────────────────────
		cmd("/new", "Start a new session", "bridge", false, {
			category: "session",
		}),
		cmd(
			"/sessions",
			"List or clean sessions in the current folder",
			"local",
			true,
			{
				category: "session",
				argHint: "[clean]",
				examples: ["/sessions", "/sessions clean"],
			},
			args => {
				if (!args.trim()) return undefined;
				return String(
					localHandlers.sessions?.(args) ?? "Session cleanup unavailable.",
				);
			},
		),
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
		cmd(
			"/queue",
			"Open the interactive message queue manager",
			"local",
			false,
			{ category: "agent", examples: ["/queue"] },
			() => {
				localHandlers.openQueueManager?.();
				return undefined;
			},
		),
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
		cmd(
			"/memory",
			"List or search stored session memories",
			"local",
			true,
			{
				category: "context",
				argHint: "[list | search <query>]",
				examples: ["/memory", "/memory list", "/memory search auth"],
				subcommands: [
					"list",
					"search",
					"obs",
					"stats",
					"tiers",
					"auto-tier",
					"forget",
					"clean",
					"consolidate",
					"context",
					"retention",
				],
			},
			args => String(localHandlers.memory?.(args as any) ?? ""),
		),
		cmd(
			"/obs",
			"List or search agent observations",
			"local",
			true,
			{
				category: "context",
				argHint:
					"[list [type] [limit] | search <query> [limit] | stats | sessions | by-session <sid> [limit]]",
				examples: [
					"/obs",
					"/obs list",
					"/obs list file_read 20",
					"/obs search error 10",
					"/obs stats",
					"/obs sessions",
					"/obs by-session sess-abc123 50",
				],
				subcommands: [
					"list",
					"search",
					"stats",
					"sessions",
					"by-session",
					"clean",
				],
			},
			args => String(localHandlers.obs?.(args as any) ?? ""),
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
			subcommands: ["list", "install", "remove"],
		}),
		cmd("/mcp", "Manage MCP servers", "local", true, {
			category: "skills",
			argHint: "[list|add|remove]",
			examples: ["/mcp", "/mcp add http://localhost:3000"],
			subcommands: ["list", "add", "remove"],
		}),
		cmd("/theme", "Select a color theme", "local", true, {
			category: "display",
			argHint: "<theme>",
			examples: ["/theme dark", "/theme github-dark", "/theme light"],
			subcommands: ["list", "dark", "github-dark", "light"],
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
			subcommands: [
				"list",
				"none",
				"tot",
				"reflexion",
				"ssr",
				"auto-cot",
				"best-of-n",
				"self-consistency",
				"got",
				"cover",
			],
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
				subcommands: ["status", "stop", "best", "reset"],
			},
			(args: string) => String(localHandlers.eoh?.(args) ?? "EoH unavailable."),
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
				subcommands: ["off", "low", "medium", "high", "xhigh"],
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
				subcommands: ["collapsed", "summary", "expanded"],
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
				subcommands: ["acceptAll", "acceptEdits", "ask", "plan"],
			},
			(args: string) => {
				const valid = ["acceptAll", "acceptEdits", "ask", "plan"];
				const mode = valid.find(
					m => m.toLowerCase() === args.trim().toLowerCase(),
				);
				if (mode) {
					localHandlers.setPermissionMode?.(mode);
					return `Permission mode: ${mode}`;
				}
				return `Valid modes: ${valid.join(", ")} (current: ${
					localHandlers.getPermissionMode?.() ?? "acceptEdits"
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
		cmd(
			"/autoresearch",
			"Start, stop, clear, or export an autonomous experiment loop",
			"local",
			true,
			{
				category: "loop",
				argHint: "[off|clear|export|<goal>]",
				examples: [
					"/autoresearch optimize unit test runtime, monitor correctness",
					"/autoresearch export",
					"/autoresearch off",
					"/autoresearch clear",
				],
				subcommands: ["off", "clear", "export"],
			},
		),

		// ── Agent ────────────────────────────────────────────────────────────
		cmd(
			"/spawn",
			"Spawn a subagent to run a task autonomously",
			"local",
			true,
			{
				category: "agent",
				argHint: "<task description>",
				examples: ["/spawn Review auth changes and run tests", "/spawn"],
			},
			args => {
				const task =
					args.trim() || "Investigate the codebase and report findings";
				localHandlers.spawnAgentDirectly?.(task);
				return undefined;
			},
		),
		cmd(
			"/spawn-test",
			"Spawn 2 fixed sample tasks via spawn_agents to test rendering",
			"local",
			false,
			{
				category: "agent",
				examples: ["/spawn-test"],
			},
			() => {
				const prompt = `Call spawn_agents once with exactly these two tasks, then report their results:

- Task 0: agent="explorer" — task="List the files in the current directory"
- Task 1: agent="general" — task="Say hello and explain what spawn_agents is in one sentence"`;
				localHandlers.sendSpawnPrompt?.(prompt);
				return undefined;
			},
		),

		// ── Misc ─────────────────────────────────────────────────────────────
		cmd(
			"/notifications",
			"Show recent notification history",
			"local",
			false,
			{ category: "misc", examples: ["/notifications"] },
			() => String(localHandlers.notifications?.() ?? "No notifications yet."),
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
				subcommands: [
					"status",
					"profile",
					"none",
					"code",
					"file",
					"dev",
					"full",
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
				subcommands: [
					"thinking",
					"model",
					"model-cycle",
					"temp",
					"max-tokens",
					"max-iterations",
					"loop-detection",
					"guards",
					"compaction",
					"permissions",
				],
			},
			args => {
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
		cmd(
			"/legroom",
			"Toggle Legroom SDK context compression",
			"local",
			false,
			{
				category: "misc",
				examples: ["/legroom"],
			},
			() => {
				const state = localHandlers.toggleLegroom?.();
				return state ? "Legroom SDK: on" : "Legroom SDK: off";
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
				command =>
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

	const lowerQuery = query.toLowerCase().trim().replace(/^\/+/, "");
	if (!lowerQuery) return commands.slice(0, limit);
	const scored = commands
		.map((cmd, idx) => {
			const cmdName = cmd.command.toLowerCase().replace(/^\/+/, "");
			const desc = cmd.description.toLowerCase();
			const category = (cmd.category ?? "misc").toLowerCase();
			const commandScore = fuzzyFieldScore(lowerQuery, cmdName, 10_000);
			const descriptionScore = fuzzyFieldScore(lowerQuery, desc, 3_000);
			const categoryScore = fuzzyFieldScore(lowerQuery, category, 1_500);
			const score = Math.max(commandScore, descriptionScore, categoryScore);

			return score >= 0 ? { cmd, score, idx } : null;
		})
		.filter(Boolean) as { cmd: SlashCommandDef; score: number; idx: number }[];

	scored.sort((a, b) => b.score - a.score || a.idx - b.idx);
	return scored.map(s => s.cmd).slice(0, limit);
}

/** Score a field while rewarding compact runs and word-boundary matches. */
function fuzzyFieldScore(query: string, text: string, base: number): number {
	if (text === query) return base + 4_000;
	if (text.startsWith(query))
		return base + 3_000 - (text.length - query.length);

	const containedAt = text.indexOf(query);
	if (containedAt >= 0) {
		const boundaryBonus =
			containedAt === 0 || /[\s_/-]/.test(text[containedAt - 1] ?? "")
				? 800
				: 0;
		return base + 1_800 + boundaryBonus - containedAt * 4;
	}

	let queryIndex = 0;
	let first = -1;
	let previous = -2;
	let gaps = 0;
	let consecutive = 0;
	let boundaries = 0;
	for (
		let textIndex = 0;
		textIndex < text.length && queryIndex < query.length;
		textIndex++
	) {
		if (text[textIndex] !== query[queryIndex]) continue;
		if (first < 0) first = textIndex;
		if (textIndex === previous + 1) consecutive++;
		else if (previous >= 0) gaps += textIndex - previous - 1;
		if (textIndex === 0 || /[\s_/-]/.test(text[textIndex - 1] ?? ""))
			boundaries++;
		previous = textIndex;
		queryIndex++;
	}
	if (queryIndex !== query.length) return -1;
	return (
		base + 700 + boundaries * 100 + consecutive * 35 - gaps * 12 - first * 3
	);
}
