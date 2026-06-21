// ── Status bar (compact single-line footer) ────────────────────────────────────
// Example: ⏸ ready | Qwen | think:off | dir logician | ⎇ main *18 +1 ?4 | ◫ 49.4%/150k | cache in: 46M | reasoner: none
//
// Sections (separated by |):
//   phase | model | thinking | dir/git | context | cache | reasoner

import { type Component, visibleWidth } from "../layers/core/tui-core.ts";
import { theme } from "../layers/theme/theme.ts";

const RESET = "\x1b[0m";
const DIM = "\x1b[2m";

interface StatusInfo {
	thinkingLevel: string;
	cacheEnabled: boolean;
	cacheSize?: number;
	turnCount: number;
	messageCount: number;
	phase: string;
	model: string;
	cwd: string;
	branch: string;
	gitModified?: number;
	gitStaged?: number;
	gitUntracked?: number;
	contextTokens: number;
	contextMaxTokens?: number;
	contextCompacted: boolean;
	reasoner: string;
	sessionTitle?: string;
}

const DEFAULT_INFO: StatusInfo = {
	thinkingLevel: "off",
	cacheEnabled: true,
	cacheSize: 0,
	turnCount: 0,
	messageCount: 0,
	phase: "ready",
	model: "local",
	cwd: process.cwd(),
	branch: "",
	gitModified: 0,
	gitStaged: 0,
	gitUntracked: 0,
	contextTokens: 0,
	contextMaxTokens: undefined,
	contextCompacted: false,
	reasoner: "none",
	sessionTitle: "",
};

export class StatusBar implements Component {
	private info: StatusInfo = { ...DEFAULT_INFO };
	private tick = 0;
	private timer: ReturnType<typeof setInterval> | null = null;
	private cachedLine: string | null = null;
	private cachedWidth = -1;
	private onInvalidate: (() => void) | null = null;

	setOnInvalidate(cb: () => void): void {
		this.onInvalidate = cb;
	}

	update(info: Partial<StatusInfo>): void {
		Object.assign(this.info, info);
		this._invalidate();
	}

	setTick(tick: number): void {
		this.tick = tick;
		this._invalidate();
	}

	invalidate(): void {
		this._invalidate();
	}

	_invalidate(): void {
		this.cachedLine = null;
		this.onInvalidate?.();
	}

	// Start animation timer for streaming phases
	startAnimation(): void {
		if (this.timer) return;
		this.timer = setInterval(() => {
			this.tick = (this.tick + 1) % 8;
			this._invalidate();
		}, 150);
	}

	stopAnimation(): void {
		if (this.timer) {
			clearInterval(this.timer);
			this.timer = null;
		}
		this.tick = 0;
		this._invalidate();
	}

	render(width: number): string[] {
		if (width === this.cachedWidth && this.cachedLine !== null) {
			return [this.cachedLine];
		}

		this.cachedWidth = width;
		const line = this.renderCompact(width);
		this.cachedLine = line;
		return [line];
	}

	// ── Compact single-line render ──────────────────────────────────────────

	private renderCompact(width: number): string {
		const parts: string[] = [];

		// 1. Phase indicator
		const phase = this.formatPhase();
		if (phase) parts.push(phase);

		// 2. Model
		const model = this.formatModel();
		if (model) parts.push(model);

		// 3. Thinking
		const think = this.formatThinking();
		if (think) parts.push(think);

		// 4. Directory + Git
		const dirGit = this.formatDirWithGit();
		if (dirGit) parts.push(dirGit);

		// 5. Context
		const ctx = this.formatContext();
		if (ctx) parts.push(ctx);

		// 6. Cache
		const cache = this.formatCache();
		if (cache) parts.push(cache);

		// 7. Reasoner
		const reasoner = this.formatReasoner();
		if (reasoner) parts.push(reasoner);

		// Join with |
		let line = parts.join(` ${DIM}|${RESET} `);

		// Truncate if too wide
		if (visibleWidth(line) > width) {
			line = this.truncateVisible(line, width);
		}

		return line;
	}

	private formatModel(): string {
		return theme.fg("text", this.info.model || "local");
	}

	private formatPhase(): string {
		const phase = this.info.phase || "ready";
		const phaseLabels: Record<string, string> = {
			ready: "⏸ ready",
			thinking: "🧠 thinking",
			tool: "🔧 tool",
			streaming: "⚡ streaming",
			compacting: "📦 compacting",
			branching: "🌿 branching",
			error: "❌ error",
		};
		const label = phaseLabels[phase] || `⏸ ${phase}`;
		const color =
			this.info.phase === "error"
				? theme.fg("error", "")
				: this.info.phase === "streaming"
					? theme.fg("accent", "")
					: this.info.phase === "thinking"
						? theme.fg("phaseThinking", "")
						: this.info.phase === "tool"
							? theme.fg("phaseTool", "")
							: theme.fg("success", "");
		return `${color}${label}${RESET}`;
	}

	private formatThinking(): string {
		const lvl = this.info.thinkingLevel;
		if (lvl === "off") {
			return `${DIM}think:${RESET} ${theme.fg("levelOff", "off")}`;
		}
		const levelColors: Record<string, string> = {
			low: theme.fg("levelLow", ""),
			medium: theme.fg("levelMedium", ""),
			high: theme.fg("levelHigh", ""),
			xhigh: theme.fg("levelXhigh", ""),
		};
		const color = levelColors[lvl] ?? theme.fg("accent", "");
		return `${DIM}think:${RESET} ${color}${lvl.toUpperCase()}`;
	}

	private formatDir(): string {
		const home = process.env.HOME || "";
		let cwd = this.info.cwd || process.cwd();
		if (home && cwd.startsWith(home)) {
			cwd = `~${cwd.slice(home.length)}`;
		}
		// Just the last directory component
		const parts = cwd.split("/").filter(Boolean);
		const name = parts[parts.length - 1] || ".";
		return `${DIM}dir${RESET} ${name}`;
	}

	private formatGit(): string {
		if (!this.info.branch) return "";

		const branch = theme.fg("success", this.info.branch);
		let indicators = "";

		if (this.info.gitModified) {
			indicators += `${theme.fg("warning", `*${this.info.gitModified}`)}`;
		}
		if (this.info.gitStaged) {
			indicators += ` ${theme.fg("success", `+${this.info.gitStaged}`)}`;
		}
		if (this.info.gitUntracked) {
			indicators += ` ${theme.fg("error", `?${this.info.gitUntracked}`)}`;
		}

		if (!indicators) return `${DIM}⎇${RESET} ${branch}`;

		return `${DIM}⎇${RESET} ${branch} ${indicators}`;
	}

	private formatDirWithGit(): string {
		const dir = this.formatDir();
		const git = this.formatGit();
		if (dir && git) {
			return `${dir} ${DIM}|${RESET} ${git}`;
		}
		return dir || git || "";
	}

	private formatContext(): string {
		const tokens = Math.max(0, Math.round(this.info.contextTokens || 0));
		const maxTokens = this.info.contextMaxTokens;
		if (!maxTokens || maxTokens === 0) return "";

		const ratio = Math.min(1, tokens / maxTokens);
		const pct = (ratio * 100).toFixed(1);

		// Color based on ratio
		const color =
			ratio >= 0.9
				? theme.fg("contextCritical", "")
				: ratio >= 0.75
					? theme.fg("contextWarning", "")
					: theme.fg("contextGood", "");

		const maxStr = formatTokenCountClean(maxTokens);

		return `${DIM}◫${RESET} ${color}${pct}%/${maxStr}${RESET} AC`;
	}

	private formatCache(): string {
		const size = this.info.cacheSize || 0;
		if (size === 0) {
			return this.info.cacheEnabled
				? `${DIM}cache in:${RESET} ${theme.fg("success", "on")}`
				: `${DIM}cache in:${RESET} ${theme.fg("dim", "off")}`;
		}
		const sizeStr = formatCacheSize(size);
		return `${DIM}cache in:${RESET} ${theme.fg("accent", sizeStr)}`;
	}

	private formatReasoner(): string {
		const reasoner = this.info.reasoner || "none";
		return `${DIM}reasoner:${RESET} ${theme.fg("muted", reasoner)}`;
	}

	// ── Helpers ──────────────────────────────────────────────────────────────

	private truncateVisible(text: string, width: number): string {
		if (visibleWidth(text) <= width) return text;
		const ellipsis = "…";
		let out = "";
		let inEscape = false;
		let visible = 0;
		const target = Math.max(0, width - visibleWidth(ellipsis));
		for (let i = 0; i < text.length && visible < target; i++) {
			const ch = text[i];
			if (ch === "\x1b" && text[i + 1] === "[") {
				inEscape = true;
				out += ch;
				continue;
			}
			if (inEscape) {
				out += ch;
				if (ch === "m") inEscape = false;
				continue;
			}
			const chWidth = visibleWidth(ch);
			if (chWidth > 0) {
				out += ch;
				visible += chWidth;
			}
		}
		return out + ellipsis;
	}
}

function formatTokenCountClean(tokens: number): string {
	if (tokens >= 1_000_000) {
		const v = tokens / 1_000_000;
		return v % 1 === 0 ? `${Math.round(v)}M` : `${v.toFixed(1)}M`;
	}
	if (tokens >= 1000) {
		const v = tokens / 1000;
		return v % 1 === 0 ? `${Math.round(v)}k` : `${v.toFixed(1)}k`;
	}
	return String(tokens);
}

function formatCacheSize(bytes: number): string {
	if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(0)}M`;
	if (bytes >= 1000) return `${(bytes / 1000).toFixed(0)}K`;
	return `${bytes}B`;
}
