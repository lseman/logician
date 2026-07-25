// ── Status bar (compact single-line footer) ────────────────────────────────────
// Example: ⏸ ready | Qwen | think:off | dir logician | ⎇ main *18 +1 ?4 | ◫ 49.4%/150k | cache read: 12.4k | reasoner: none | mcp: 3
//
// Sections (separated by |):
//   phase | model | thinking | dir/git | context | cache | reasoner | mcp

import { type Component, visibleWidth, RESET, DIM } from "../layers/core/tui-core.ts";
import { theme } from "../layers/theme/theme.ts";

/** Max chars for a free-text label before it's ellipsis-truncated. */
const LABEL_TRUNCATE_LENGTH = 24;

interface StatusInfo {
	thinkingLevel: string;
	inferenceMode: string;
	cacheReadTokens?: number;
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
	goalCondition?: string;
	goalTurnCount?: number;
	goalElapsed?: number;
	mcpServerCount?: number;
	sandboxMode?: "none" | "code" | "file" | "dev" | "full";
}

const DEFAULT_INFO: StatusInfo = {
	thinkingLevel: "off",
	inferenceMode: "instruct-general",
	cacheReadTokens: undefined,
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
	mcpServerCount: 0,
	sandboxMode: "code",
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
		const separator = ` ${DIM}│${RESET} `;
		const phase = this.formatPhase();
		const model = this.formatModel();
		const context = this.formatContext();
		const parts = [phase, model, context].filter(Boolean);
		const fits = (candidate: string[]): boolean =>
			visibleWidth(candidate.join(separator)) <= width;
		const insertIfFits = (part: string, index = parts.length): void => {
			if (!part) return;
			const candidate = [...parts];
			candidate.splice(index, 0, part);
			if (fits(candidate)) parts.splice(index, 0, part);
		};

		// Add detail by usefulness. Narrow terminals retain phase/model/context.
		insertIfFits(this.formatDirWithGit(), 2);
		insertIfFits(this.formatSession(), 2);
		insertIfFits(this.formatThinking());
		insertIfFits(this.formatCache());
		insertIfFits(this.formatGoal());
		insertIfFits(this.formatInferenceMode());
		if (this.info.reasoner !== "none") insertIfFits(this.formatReasoner());
		if (this.info.mcpServerCount) insertIfFits(this.formatMcp());
		insertIfFits(this.formatSandbox());

		let line = parts.join(separator);
		if (visibleWidth(line) > width) {
			const compact = [phase, context].filter(Boolean).join(separator);
			line = visibleWidth(compact) <= width
				? compact
				: this.truncateVisible(phase, width);
		}
		return line + RESET;
	}

	private formatModel(): string {
		return theme.fg("text", this.info.model || "local");
	}

	private formatSession(): string {
		const title = this.info.sessionTitle?.trim();
		if (!title || title === "New Session") return "";
		const compact = title.length > LABEL_TRUNCATE_LENGTH
			? `${title.slice(0, LABEL_TRUNCATE_LENGTH - 1)}…`
			: title;
		return `${DIM}◇${RESET} ${theme.fg("muted", compact)}`;
	}

	private formatPhase(): string {
		const phase = this.info.phase || "ready";
		const spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"][
			this.tick % 8
		];
		const phaseLabels: Record<string, string> = {
			ready: "● READY",
			thinking: `${spinner} THINKING`,
			tool: `${spinner} TOOL`,
			verifying: `${spinner} VERIFYING`,
			streaming: `${spinner} STREAMING`,
			waiting: "◆ WAITING",
			approval: "◆ APPROVAL",
			failed: "× FAILED",
			compacting: `${spinner} COMPACTING`,
			branching: `${spinner} BRANCHING`,
			error: "× ERROR",
		};
		const label = phaseLabels[phase] || `● ${phase.toUpperCase()}`;
		const color =
			this.info.phase === "error" || this.info.phase === "failed"
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

	private formatInferenceMode(): string {
		const mode = this.info.inferenceMode;
		const modeLabels: Record<string, string> = {
			"thinking-general": "THINK GEN",
			"thinking-coding": "THINK CODE",
			"instruct-general": "INSTRUCT",
			"instruct-reasoning": "REASON",
		};
		const label = modeLabels[mode] ?? mode.toUpperCase();
		return `${DIM}mode:${RESET} ${theme.fg("accent", label)}`;
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
			return `${dir} ${DIM}│${RESET} ${git}`;
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
		const cells = 5;
		const filled = Math.min(cells, Math.max(0, Math.round(ratio * cells)));
		const meter = filled > 0 ? `${"▰".repeat(filled)}` : "";

		return `${DIM}ctx${RESET} ${color}${meter} ${pct}%${RESET}${DIM}/${maxStr}${RESET}`;
	}

	private formatCache(): string {
		const tokens = this.info.cacheReadTokens;
		const value =
			tokens === undefined
				? theme.fg("dim", "unknown")
				: theme.fg("accent", formatTokenCountClean(tokens));
		return `${DIM}cache read:${RESET} ${value}`;
	}

	private formatReasoner(): string {
		const reasoner = this.info.reasoner || "none";
		return `${DIM}reasoner:${RESET} ${theme.fg("muted", reasoner)}`;
	}

	private formatGoal(): string {
		const cond = this.info.goalCondition;
		if (!cond) return "";
		const turns = this.info.goalTurnCount || 0;
		const elapsed = this.info.goalElapsed || 0;
		const mins = Math.floor(elapsed / 60);
		const secs = elapsed % 60;
		const timeStr = mins > 0 ? `${mins}m${secs}s` : `${secs}s`;
		const truncated = cond.length > LABEL_TRUNCATE_LENGTH
			? cond.slice(0, LABEL_TRUNCATE_LENGTH) + "…"
			: cond;
		return `${theme.fg("accent", `◎ ${truncated}`)} ${DIM}(${turns} turns, ${timeStr})${RESET}`;
	}

	private formatMcp(): string {
		const count = this.info.mcpServerCount || 0;
		return `${DIM}mcp${RESET} ${theme.fg("accent", `${count}`)}${RESET}`;
	}

	private formatSandbox(): string {
		const mode = this.info.sandboxMode ?? "code";
		if (mode === "none") {
			return `${DIM}sandbox:${RESET} ${theme.fg("levelOff", "off")}`;
		}
		return `${DIM}sandbox:${RESET} ${theme.fg("accent", mode)}`;
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
