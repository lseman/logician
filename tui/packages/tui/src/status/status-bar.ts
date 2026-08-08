// ── Status bar (compact single-line footer) ────────────────────────────────────
// Example: ⏸ ready | Qwen | think:off | dir logician | ⎇ main *18 +1 ?4 | ◫ 49.4%/150k | cache read: 12.4k | reasoner: none | mcp: 3
//
// Sections (separated by |):
//   phase | model | thinking | dir/git | context | cache | reasoner | mcp

import { type Component, DIM, RESET, visibleWidth } from "../terminal/core.ts";
import { theme } from "../terminal/theme.ts";

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
	mcpLoading?: boolean;
	sandboxMode?: "none" | "code" | "file" | "dev" | "full";
	permissionMode?: string;
	executionProfile?: "autonomous" | "minimal";
	promptTokens?: number;
	completionTokens?: number;
	rtkProxyEnabled?: boolean;
	memoryEnabled?: boolean;
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
	mcpLoading: false,
	sandboxMode: "code",
	permissionMode: "acceptAll",
	executionProfile: "autonomous",
	promptTokens: undefined,
	completionTokens: undefined,
	rtkProxyEnabled: false,
};

export class StatusBar implements Component {
	private info: StatusInfo = { ...DEFAULT_INFO };
	private tick = 0;
	private _timer: ReturnType<typeof setInterval> | null = null;
	private cachedLine: string | null = null;
	private cachedWidth = -1;
	private onInvalidate: (() => void) | null = null;

	/** @internal Exposed for tests. */
	get timer(): ReturnType<typeof setInterval> | null { return this._timer; }
	/** Non-phase parts that fit at cachedWidth, from the last full layout pass.
	 * A tick-only update reuses this instead of rerunning the fit probing. */
	private cachedParts: string[] = [];
	private tickOnlyDirty = false;

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
		this.tickOnlyDirty = false;
		this.onInvalidate?.();
	}

	// Start animation timer for streaming phases
	startAnimation(): void {
		if (this._timer) return;
		this._timer = setInterval(() => {
			this.tick = (this.tick + 1) % 8;
			// A spinner-only change never affects layout (every frame glyph is one
			// column wide, so which parts fit and the truncation fallback can't
			// flip), so skip the full renderCompact() rebuild — including its
			// dozen-odd visibleWidth() fit-probe calls — and just splice the new
			// phase segment into the already-composed line on the next render().
			if (!this.tickOnlyDirty) {
				this.tickOnlyDirty = true;
				this.onInvalidate?.();
			}
		}, 150);
	}

	stopAnimation(): void {
		if (this._timer) {
			clearInterval(this._timer);
			this._timer = null;
		}
		this.tick = 0;
		this._invalidate();
	}

	render(width: number): string[] {
		if (width === this.cachedWidth && this.cachedLine !== null) {
			if (this.tickOnlyDirty) {
				// Spinner-only change: every frame glyph is one column wide, so
				// which parts fit and whether the truncation fallback triggers
				// can't flip between ticks — only formatPhase()'s output does.
				// Recompute just that and re-splice, skipping the dozen-odd
				// visibleWidth() fit-probe calls renderCompact() would otherwise
				// redo for parts that provably haven't changed.
				this.cachedLine = this.composeLine(width, this.cachedParts);
				this.tickOnlyDirty = false;
			}
			return [this.cachedLine];
		}

		this.cachedWidth = width;
		this.cachedParts = this.layoutParts(width);
		this.tickOnlyDirty = false;
		const line = this.composeLine(width, this.cachedParts);
		this.cachedLine = line;
		return [line];
	}

	// ── Compact single-line render ──────────────────────────────────────────

	/** Which non-phase parts fit at this width, and where phase/context sit. */
	private layoutParts(width: number): string[] {
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
		insertIfFits(this.formatThinking());
		// An explicitly selected reasoner is active behavior, not telemetry. Give
		// it priority over cache/token details so the user's choice stays visible.
		if (this.info.reasoner !== "none") insertIfFits(this.formatReasoner());
		insertIfFits(this.formatCache());
		insertIfFits(this.formatTokenFlow());
		insertIfFits(this.formatGoal());
		insertIfFits(this.formatInferenceMode());
		if (this.info.mcpServerCount || this.info.mcpLoading)
			insertIfFits(this.formatMcp());
		insertIfFits(this.formatSandbox());
		insertIfFits(this.formatExecutionProfile());
		insertIfFits(this.formatPermissionMode());
		insertIfFits(this.formatRtk());
		insertIfFits(this.formatMemory());
		return parts;
	}

	/** Re-derives phase/context fresh (tick-sensitive) and joins with the
	 * already-decided part list. Cheap: no width-fit probing. */
	private composeLine(width: number, parts: string[]): string {
		const separator = ` ${DIM}│${RESET} `;
		const phase = this.formatPhase();
		const context = this.formatContext();
		// parts[0] is always phase when present (layoutParts starts the array
		// with it); refresh it in place since formatPhase() picked up the tick.
		const refreshed = parts.length > 0 ? [phase, ...parts.slice(1)] : parts;

		let line = refreshed.join(separator);
		if (visibleWidth(line) > width) {
			const compact = [phase, context].filter(Boolean).join(separator);
			line =
				visibleWidth(compact) <= width
					? compact
					: this.truncateVisible(phase, width);
		}
		return line + RESET;
	}

	private formatModel(): string {
		return theme.fg("text", this.info.model || "local");
	}

	private formatPhase(): string {
		const phase = this.info.phase || "ready";
		const spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"][this.tick % 8];
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
			cancelling: `${spinner} CANCELLING`,
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
			none: "PROVIDER",
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

	private formatTokenFlow(): string {
		const inTok = this.info.promptTokens;
		const outTok = this.info.completionTokens;
		if (inTok === undefined && outTok === undefined) return "";

		const inStr = inTok !== undefined ? formatTokenCountClean(inTok) : "–";
		const outStr = outTok !== undefined ? formatTokenCountClean(outTok) : "–";

		return `${DIM}↑${RESET} ${theme.fg("accent", inStr)}${DIM} │ ${RESET}${DIM}↓${RESET} ${theme.fg("accent", outStr)}`;
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
		const truncated =
			cond.length > LABEL_TRUNCATE_LENGTH
				? `${cond.slice(0, LABEL_TRUNCATE_LENGTH)}…`
				: cond;
		return `${theme.fg("accent", `◎ ${truncated}`)} ${DIM}(${turns} turns, ${timeStr})${RESET}`;
	}

	private formatMcp(): string {
		if (this.info.mcpLoading) {
			return `${DIM}mcp${RESET} ${theme.fg("warning", "loading…")}`;
		}
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

	private formatPermissionMode(): string {
		const mode = this.info.permissionMode ?? "acceptAll";
		if (mode === "acceptAll") {
			return `${theme.fg("success", "act")}`;
		}
		return `${theme.fg("warning", "plan")}`;
	}

	private formatRtk(): string {
		if (!this.info.rtkProxyEnabled) return "";
		return `${DIM}rtk${RESET} ${theme.fg("accent", "on")}`;
	}

	private formatMemory(): string {
		if (!this.info.memoryEnabled) return "";
		return `${DIM}memory${RESET} ${theme.fg("accent", "on")}`;
	}

	private formatExecutionProfile(): string {
		const profile = this.info.executionProfile ?? "autonomous";
		return profile === "minimal"
			? `${DIM}exec:${RESET} ${theme.fg("warning", "minimal")}`
			: `${DIM}exec:${RESET} ${theme.fg("success", "auto")}`;
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
