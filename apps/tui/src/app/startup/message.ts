import { clampLineToWidth, visibleWidth } from "../../terminal/core.ts";
import { theme } from "../../terminal/theme.ts";

// ── Startup splash ───────────────────────────────────────────────────────────
// A bordered welcome card rendered verbatim into the transcript (see the
// "[Banner]\n" branch in rendering/transcript/display.ts). Left panel: a
// gradient wordmark + the active model. Right panel: quick-start keys and a
// compact rundown of what loaded. Anything that doesn't fit the card cleanly
// (plugin startup notices, the initial hook message) is printed below it.

const RESET = "\x1b[0m";
const BOLD = "\x1b[1m";
const ITALIC = "\x1b[3m";

const BANNER_PREFIX = "[Banner]\n";

export interface StartupMessageOptions {
	configPath?: string;
	project: string;
	themeName: string;
}

// Pink → violet → blue → cyan. Sampled across the wordmark and the mark.
const GRADIENT: Array<[number, number, number]> = [
	[0xe0, 0x6b, 0xd6],
	[0xa8, 0x55, 0xf7],
	[0x3b, 0x82, 0xf6],
	[0x22, 0xd3, 0xee],
];

function lerp(a: number, b: number, t: number): number {
	return Math.round(a + (b - a) * t);
}

function gradientAt(t: number): [number, number, number] {
	const clamped = Math.min(1, Math.max(0, t));
	const span = GRADIENT.length - 1;
	const scaled = clamped * span;
	const i = Math.min(span - 1, Math.floor(scaled));
	const frac = scaled - i;
	const [r1, g1, b1] = GRADIENT[i];
	const [r2, g2, b2] = GRADIENT[i + 1];
	return [lerp(r1, r2, frac), lerp(g1, g2, frac), lerp(b1, b2, frac)];
}

/** Colour `text` with the gradient swept from `t0` to `t1` across its chars. */
function gradientText(text: string, t0: number, t1: number): string {
	const chars = [...text];
	if (chars.length === 0) return "";
	let out = "";
	for (let i = 0; i < chars.length; i++) {
		const t =
			chars.length === 1 ? t0 : t0 + ((t1 - t0) * i) / (chars.length - 1);
		const [r, g, b] = gradientAt(t);
		out += `${theme.rgbRaw(r, g, b)}${chars[i]}`;
	}
	return `${out}${RESET}`;
}

const TIPS = [
	"Type / to browse commands, or @ to pull a file into context.",
	"Prefix a line with ! to run bash, or $ to run Python.",
	"Press Esc to interrupt the current turn.",
	"Drop a path with @ and it's read straight into the conversation.",
];

// The Logician mark: a bold "L" whose spine and foot double as the axes of a
// graph, with a logarithmic curve rising through it — an ASCII echo of
// logo/logician-logo.svg. The curve segments are drawn per row so the gradient
// can sweep along it from origin to the flattening tail.
const MARK_SPINE = "┃";
const MARK_FOOT = "┗━━━━━━━━━━";
const MARK_CURVE = ["      ╭──", "   ╭──╯", " ╭─╯", "╭╯"];

export function formatStartupMessage(
	state: Record<string, unknown>,
	options: StartupMessageOptions,
): string {
	const pluginCount = Number(state.startup_plugins_loaded || 0);
	const hookCount = Number(state.startup_hooks_loaded || 0);
	const mcpServerCount = Number(state.mcp_servers_loaded || 0);
	const mcpToolCount = Number(state.mcp_tools_loaded || 0);
	const skills = normalizeSkills(state.loaded_skills);
	const contexts = stringList(state.startup_hook_contexts);
	const hookMessages = Array.isArray(state.startup_hook_messages)
		? state.startup_hook_messages
				.map(normalizeStartupHookMessage)
				.filter(item => item.content)
		: [];
	const initialMessage = String(
		state.startup_hook_initial_message || "",
	).trim();
	const errors = stringList(state.startup_hook_errors);
	const mcpErrors = stringList(state.mcp_errors);
	const model = String(state.model || "unknown");
	const agent = String(state.agent_name || "logician");
	const mcpState = state.mcp_loading
		? "MCP loading"
		: state.mcp_deferred
			? "MCP deferred"
			: `MCP ${mcpServerCount}/${mcpToolCount}`;
	const searchEnabled =
		state.web_search_enabled === true ||
		(Array.isArray(state.tools) && state.tools.includes("web_search"));
	const searchState = searchEnabled ? "on" : "off";

	// ── Card geometry ───────────────────────────────────────────────────────
	const termCols =
		typeof process.stdout.columns === "number" && process.stdout.columns > 20
			? process.stdout.columns
			: 80;
	const W = Math.max(54, Math.min(78, termCols - 6));
	const LEFT = 16; // left panel visible width
	const RIGHT = W - LEFT - 7; // right panel visible width
	const leftDashes = LEFT + 2;
	const rightDashes = W - LEFT - 5;

	const bd = theme.fgRaw("borderMuted");
	const accent = theme.fgRaw("accent");
	const muted = theme.fgRaw("muted");
	const dim = theme.fgRaw("dim");
	const err = theme.fgRaw("error");

	const fit = (s: string, n: number): string => {
		const clipped = clampLineToWidth(s, n);
		return clipped + " ".repeat(Math.max(0, n - visibleWidth(clipped)));
	};
	const center = (s: string, n: number): string => {
		const clipped = clampLineToWidth(s, n);
		const pad = Math.max(0, n - visibleWidth(clipped));
		const l = Math.floor(pad / 2);
		return " ".repeat(l) + clipped + " ".repeat(pad - l);
	};
	const ellip = (s: string, n: number): string =>
		visibleWidth(s) <= n ? s : `${clampLineToWidth(s, Math.max(1, n - 1))}…`;

	const row = (left: string, right: string): string =>
		`${bd}│${RESET} ${fit(left, LEFT)} ${bd}│${RESET} ${fit(right, RIGHT)} ${bd}│${RESET}`;
	// The left column is one tall cell — section rules only cross the right side,
	// so a separator row still carries whatever left-cell line falls on it.
	const sep = (left: string): string =>
		`${bd}│${RESET} ${fit(left, LEFT)} ${bd}├${"─".repeat(rightDashes)}┤${RESET}`;
	const bottom = (): string =>
		`${bd}╰${"─".repeat(leftDashes)}┴${"─".repeat(rightDashes)}╯${RESET}`;
	const top = (): string => {
		const titleSeg = `─ ${gradientText("logician", 0, 0.45)}${bd} `;
		const fill = Math.max(0, leftDashes - (2 + "logician".length + 1));
		return `${bd}╭${titleSeg}${"─".repeat(fill)}┬${"─".repeat(rightDashes)}╮${RESET}`;
	};

	// ── Left panel ──────────────────────────────────────────────────────────
	// Keep a constant indent so the L's spine stays a straight vertical column
	// (centring each row independently would make it wobble).
	const lStroke = `${theme.fgRaw("text")}${BOLD}`;
	const markIndent = "  ";
	const mark = [
		...MARK_CURVE.map((seg, i) => {
			// Sweep the curve gradient from its flat tail (top) down to the origin.
			const t = 1 - (i / MARK_CURVE.length) * 0.55;
			const [r, g, b] = gradientAt(t);
			return fit(
				`${markIndent}${lStroke}${MARK_SPINE}${RESET}${theme.rgbRaw(r, g, b)}${seg}${RESET}`,
				LEFT,
			);
		}),
		fit(`${markIndent}${lStroke}${MARK_FOOT}${RESET}`, LEFT),
	];
	const leftContent = [
		...mark,
		center(`${accent}${BOLD}${ellip(agent, LEFT)}${RESET}`, LEFT),
		center(`${dim}${ellip(model, LEFT)}${RESET}`, LEFT),
	];

	// ── Right panel groups ──────────────────────────────────────────────────
	const key = (k: string, label: string): string =>
		`${accent}${k}${RESET}  ${muted}${label}${RESET}`;
	const hdr = (s: string): string => `${accent}${BOLD}${s}${RESET}`;
	const line = (s: string): string => `${muted}${s}${RESET}`;

	const startGroup = [
		hdr("start here"),
		key("/", "commands"),
		key("!", "run bash"),
		key("$", "run python"),
		key("@", "mention a file"),
	];

	const sessionGroup = [
		hdr("session"),
		line(options.project),
		line(`${options.themeName} theme`),
	];

	const loadedGroup = [
		hdr("loaded"),
		line(
			`${pluginCount} plugins · ${skills.length} skills · ${hookCount} hooks`,
		),
		line(`${mcpState} · web search ${searchState}`),
	];
	if (skills.length) {
		loadedGroup.push(
			`${dim}${ellip(skills.map(s => `/${s.slashName}`).join(" "), RIGHT)}${RESET}`,
		);
	}

	const groups: string[][] = [startGroup, sessionGroup, loadedGroup];
	if (errors.length || mcpErrors.length) {
		groups.push([
			`${err}${BOLD}notices${RESET}`,
			...[...errors, ...mcpErrors].map(e => `${err}• ${e}${RESET}`),
		]);
	}

	const rightRows: Array<string | null> = [];
	groups.forEach((group, i) => {
		if (i > 0) rightRows.push(null);
		rightRows.push(...group);
	});

	// ── Assemble card ───────────────────────────────────────────────────────
	// Center the left cell's content vertically over the whole interior (it
	// flows through separator rows too, since the left column is one cell).
	const leftTop = Math.max(
		0,
		Math.floor((rightRows.length - leftContent.length) / 2),
	);
	const out: string[] = [top()];
	rightRows.forEach((rr, idx) => {
		const li = idx - leftTop;
		const left = li >= 0 && li < leftContent.length ? leftContent[li] : "";
		out.push(rr === null ? sep(left) : row(left, rr));
	});
	out.push(bottom());

	// ── Below the card ──────────────────────────────────────────────────────
	if (initialMessage) {
		out.push(
			"",
			...wrapPlain(initialMessage, W).map(l => `${muted}${l}${RESET}`),
		);
	}
	const messageBlocks = hookMessages.length
		? hookMessages
		: contexts.map((content, i) => ({
				title: `Startup hook ${i + 1}`,
				content,
			}));
	for (const block of messageBlocks) {
		out.push(
			"",
			`${accent}${BOLD}${block.title}${RESET}`,
			...wrapPlain(block.content, W).map(l => `${muted}${l}${RESET}`),
		);
	}

	const tip = TIPS[Math.floor(Math.random() * TIPS.length)];
	out.push("", `${theme.fgRaw("warning")}${ITALIC}✦ ${tip}${RESET}`);

	return BANNER_PREFIX + out.join("\n");
}

/** Naive word wrap for plain (unstyled) text. */
function wrapPlain(text: string, width: number): string[] {
	const result: string[] = [];
	for (const rawLine of text.split("\n")) {
		if (rawLine.trim() === "") {
			result.push("");
			continue;
		}
		let current = "";
		for (const word of rawLine.split(/\s+/)) {
			if (current === "") {
				current = word;
			} else if (visibleWidth(`${current} ${word}`) <= width) {
				current += ` ${word}`;
			} else {
				result.push(current);
				current = word;
			}
		}
		if (current) result.push(current);
	}
	return result;
}

function stringList(value: unknown): string[] {
	return Array.isArray(value)
		? value.map(item => String(item || "").trim()).filter(Boolean)
		: [];
}

function normalizeSkills(value: unknown): Array<{
	slashName: string;
	description: string;
}> {
	if (!Array.isArray(value)) return [];
	return value
		.map(item => {
			if (!item || typeof item !== "object") return null;
			const skill = item as Record<string, unknown>;
			const slashName = String(skill.slash_name || skill.name || "").trim();
			const description = String(skill.description || "").trim();
			return slashName ? { slashName, description } : null;
		})
		.filter(
			(item): item is { slashName: string; description: string } =>
				item !== null,
		);
}

function normalizeStartupHookMessage(item: unknown): {
	title: string;
	content: string;
} {
	if (!item || typeof item !== "object") {
		return { title: "Startup hook", content: String(item || "").trim() };
	}
	const raw = item as Record<string, unknown>;
	const pluginName = String(raw.plugin_name || "").trim();
	const pluginId = String(raw.plugin_id || "").trim();
	const matcher = String(raw.matcher || "").trim();
	const label = pluginName || pluginId || "Startup hook";
	const suffix =
		pluginName && pluginId && pluginName !== pluginId ? ` (${pluginId})` : "";
	const matcherText = matcher && matcher !== "*" ? ` · ${matcher}` : "";
	return {
		title: `${label}${suffix}${matcherText}`,
		content: String(raw.content || "").trim(),
	};
}
