// ── Status bar (compact single-line footer) ────────────────────────────────────
// Example: ⏸ ready | Qwen | think:off | dir logician | ⎇ main *18 +1 ?4 | ◫ 49.4%/150k | cache read: 12.4k | reasoner: none | mcp: 3
//
// Sections (separated by |):
//   phase | model | thinking | dir/git | context | cache | reasoner | mcp

import { clampLineToWidth, type InkTextComponent, type InkTextRow, visibleWidth } from "../terminal/core.ts";
import { RESET, semanticMarkupToInkRow, theme } from "../rendering/transcript/semantic-markup.ts";

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
	sandboxMode: "code",
	permissionMode: "acceptAll",
	executionProfile: "autonomous",
	promptTokens: undefined,
	completionTokens: undefined,
	rtkProxyEnabled: false,
};

export class StatusBar implements InkTextComponent {
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

	getInkTextRows(width: number): InkTextRow[] {
		if (width === this.cachedWidth && this.cachedLine !== null) {
			return [semanticMarkupToInkRow(this.cachedLine)];
		}

		this.cachedWidth = width;
		const line = this.renderCompact(width);
		this.cachedLine = line;
		return [semanticMarkupToInkRow(line)];
	}

	// ── Compact single-line render ──────────────────────────────────────────

	private renderCompact(width: number): string {
		const separator = ` ${theme.fg("separator", "│")} `;
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
		// An explicitly selected reasoner is active behavior, not telemetry. Give
		// it priority over cache/token details so the user's choice stays visible.
		if (this.info.reasoner !== "none") insertIfFits(this.formatReasoner());
		insertIfFits(this.formatCache());
		insertIfFits(this.formatTokenFlow());
		insertIfFits(this.formatGoal());
		insertIfFits(this.formatInferenceMode());
		if (this.info.mcpServerCount) insertIfFits(this.formatMcp());
		insertIfFits(this.formatSandbox());
		insertIfFits(this.formatExecutionProfile());
		insertIfFits(this.formatPermissionMode());
		insertIfFits(this.formatRtk());
	insertIfFits(this.formatMemory());

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

	private label(text: string): string {
		return theme.fg("muted", text);
	}

	private value(text: string): string {
		return theme.fg("text", text);
	}

	private formatSession(): string {
		const title = this.info.sessionTitle?.trim();
		if (!title || title === "New Session") return "";
		const compact = title.length > LABEL_TRUNCATE_LENGTH
			? `${title.slice(0, LABEL_TRUNCATE_LENGTH - 1)}…`
			: title;
		return `${this.label("◇")} ${this.value(compact)}`;
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
			cancelling: `${spinner} CANCELLING`,
			error: "× ERROR",
		};
		const label = phaseLabels[phase] || `● ${phase.toUpperCase()}`;
		const color =
			this.info.phase === "error" || this.info.phase === "failed"
				? theme.fgRaw("error")
				: this.info.phase === "streaming"
					? theme.fgRaw("accent")
					: this.info.phase === "thinking"
						? theme.fgRaw("phaseThinking")
						: this.info.phase === "tool"
							? theme.fgRaw("phaseTool")
							: theme.fgRaw("success");
		return `${color}${label}${RESET}`;
	}

	private formatThinking(): string {
		const lvl = this.info.thinkingLevel;
		if (lvl === "off") {
			return `${this.label("think:")} ${theme.fg("levelOff", "off")}`;
		}
		const levelColors: Record<string, string> = {
			low: theme.fgRaw("levelLow"),
			medium: theme.fgRaw("levelMedium"),
			high: theme.fgRaw("levelHigh"),
			xhigh: theme.fgRaw("levelXhigh"),
		};
		const color = levelColors[lvl] ?? theme.fgRaw("accent");
		return `${this.label("think:")} ${color}${lvl.toUpperCase()}${RESET}`;
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
		return `${this.label("mode:")} ${this.value(label)}`;
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
		return `${this.label("dir")} ${this.value(name)}`;
	}

	private formatGit(): string {
		if (!this.info.branch) return "";

		const branch = this.value(this.info.branch);
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

		if (!indicators) return `${this.label("⎇")} ${branch}`;

		return `${this.label("⎇")} ${branch} ${indicators}`;
	}

	private formatDirWithGit(): string {
		const dir = this.formatDir();
		const git = this.formatGit();
		if (dir && git) {
			return `${dir} ${theme.fg("separator", "│")} ${git}`;
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
				? theme.fgRaw("contextCritical")
				: ratio >= 0.75
					? theme.fgRaw("contextWarning")
					: theme.fgRaw("contextGood");

		const maxStr = formatTokenCountClean(maxTokens);
		const cells = 5;
		const filled = Math.min(cells, Math.max(0, Math.round(ratio * cells)));
		const meter = filled > 0 ? `${"▰".repeat(filled)}` : "";

		return `${this.label("ctx")} ${color}${meter} ${pct}%${RESET}${this.label(`/${maxStr}`)}`;
	}

	private formatCache(): string {
		const tokens = this.info.cacheReadTokens;
		const value =
			tokens === undefined
				? theme.fg("dim", "unknown")
				: theme.fg("text", formatTokenCountClean(tokens));
		return `${this.label("cache read:")} ${value}`;
	}

	private formatTokenFlow(): string {
		const inTok = this.info.promptTokens;
		const outTok = this.info.completionTokens;
		if (inTok === undefined && outTok === undefined) return "";

		const inStr = inTok !== undefined ? formatTokenCountClean(inTok) : "–";
		const outStr = outTok !== undefined ? formatTokenCountClean(outTok) : "–";

		return `${this.label("↑")} ${this.value(inStr)} ${theme.fg("separator", "│")} ${this.label("↓")} ${this.value(outStr)}`;
	}

	private formatReasoner(): string {
		const reasoner = this.info.reasoner || "none";
		return `${this.label("reasoner:")} ${this.value(reasoner)}`;
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
		return `${theme.fg("accent", "◎")} ${this.value(truncated)} ${this.label(`(${turns} turns, ${timeStr})`)}`;
	}

	private formatMcp(): string {
		const count = this.info.mcpServerCount || 0;
		return `${this.label("mcp:")} ${this.value(`${count}`)}`;
	}

	private formatSandbox(): string {
		const mode = this.info.sandboxMode ?? "code";
		if (mode === "none") {
			return `${this.label("sandbox:")} ${theme.fg("levelOff", "off")}`;
		}
		return `${this.label("sandbox:")} ${this.value(mode)}`;
	}

	private formatPermissionMode(): string {
		const mode = this.info.permissionMode ?? "acceptAll";
		if (mode === "acceptAll") {
			return `${this.label("perm:")} ${theme.fg("success", "act")}`;
		}
		return `${this.label("perm:")} ${theme.fg("warning", "plan")}`;
	}

	private formatRtk(): string {
		if (!this.info.rtkProxyEnabled) return "";
		return `${this.label("rtk:")} ${theme.fg("success", "on")}`;
	}

	private formatMemory(): string {
		if (!this.info.memoryEnabled) return "";
		return `${this.label("memory:")} ${theme.fg("success", "on")}`;
	}

	private formatExecutionProfile(): string {
		const profile = this.info.executionProfile ?? "autonomous";
		return profile === "minimal"
			? `${this.label("exec:")} ${theme.fg("warning", "minimal")}`
			: `${this.label("exec:")} ${theme.fg("success", "auto")}`;
	}

	// ── Helpers ──────────────────────────────────────────────────────────────

	private truncateVisible(text: string, width: number): string {
		if (visibleWidth(text) <= width) return text;
		const ellipsis = "…";
		const target = Math.max(0, width - visibleWidth(ellipsis));
		return clampLineToWidth(text, target) + ellipsis;
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
