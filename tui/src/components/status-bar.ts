// ── Status bar component ───────────────────────────────────────────────────────
// Top status bar with animated phase indicator, thinking level, cache, counts.

import { type Component, visibleWidth } from "../tui-core.ts";

interface StatusInfo {
	thinkingLevel: string;
	cacheEnabled: boolean;
	turnCount: number;
	messageCount: number;
	phase: string;
	model: string;
	cwd: string;
	branch: string;
	contextTokens: number;
	contextMaxTokens?: number;
	contextCompacted: boolean;
	reasoner: string;
	sessionTitle?: string;
}

const DEFAULT_INFO: StatusState = {
	thinkingLevel: "medium",
	cacheEnabled: true,
	turnCount: 0,
	messageCount: 0,
	phase: "ready",
	model: "local",
	cwd: process.cwd(),
	branch: "",
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
	private cachedLines: string[] | null = null;
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
		this.cachedLines = null;
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
		if (width === this.cachedWidth && this.cachedLines !== null) {
			return this.cachedLines;
		}

		this.cachedWidth = width;
		const safeWidth = Math.max(1, width - 1);

		const brand = `${BOLD}\x1b[38;5;159mlogician\x1b[0m`;
		const phaseDisplay = this.renderPhase();
		const left = `${brand} ${DIM}·${RESET} ${phaseDisplay}`;

		const levelColors: Record<string, string> = {
			off: "\x1b[38;5;244m",
			low: "\x1b[38;5;111m",
			medium: "\x1b[38;5;141m",
			high: "\x1b[38;5;220m",
			xhigh: "\x1b[38;5;203m",
		};
		const levelColor = levelColors[this.info.thinkingLevel] ?? "\x1b[38;5;159m";
		const thinking = `${levelColor}${BOLD}${this.info.thinkingLevel.toUpperCase()}${RESET}`;
		const cache = this.info.cacheEnabled
			? "\x1b[38;5;40mcache on\x1b[0m"
			: "\x1b[38;5;244mcache off\x1b[0m";
		const context = this.renderContext();
		const counts = `${DIM}${this.info.turnCount} turns · ${this.info.messageCount} msgs${RESET}`;
		const right = `${thinking} ${DIM}·${RESET} ${cache} ${DIM}·${RESET} ${context} ${DIM}·${RESET} ${counts}`;
		const top = this.joinLeftRight(left, right, safeWidth);

		const location = this.formatLocation();
		const model = this.info.model || "local";
		const reasoner = this.info.reasoner || "none";
		const reasonerColor =
			reasoner === "none" ? "\x1b[38;5;244m" : "\x1b[38;5;159m";
		const reasonerDisplay = `${reasonerColor}reasoner:${reasoner}${RESET}`;
		const sessionInfo = this.info.sessionTitle
			? `${DIM}${this.truncateVisible(this.info.sessionTitle, 30)}${RESET}`
			: "";
		const bottomCenter = sessionInfo
			? `${DIM}[session]${RESET} ${sessionInfo}`
			: "";
		const bottomLeftFull = bottomCenter
			? `${DIM}${location}${RESET} ${DIM}·${RESET} ${bottomCenter}`
			: `${DIM}${location}${RESET}`;
		const bottomRight = `${reasonerDisplay} ${DIM}·${RESET} ${DIM}${model}${RESET}`;
		const bottom = this.joinLeftRight(bottomLeftFull, bottomRight, safeWidth);

		this.cachedLines = [top, bottom];
		return this.cachedLines;
	}

	private joinLeftRight(left: string, right: string, width: number): string {
		const leftWidth = visibleWidth(left);
		const rightWidth = visibleWidth(right);

		if (leftWidth + 2 + rightWidth <= width) {
			return left + " ".repeat(width - leftWidth - rightWidth) + right;
		}

		const availableLeft = Math.max(1, width - rightWidth - 2);
		if (availableLeft > 8) {
			const clippedLeft = this.truncateVisible(left, availableLeft);
			return (
				clippedLeft +
				" ".repeat(
					Math.max(1, width - visibleWidth(clippedLeft) - rightWidth),
				) +
				right
			);
		}

		return this.truncateVisible(left, width);
	}

	private formatLocation(): string {
		const home = process.env.HOME || process.env.USERPROFILE || "";
		let cwd = this.info.cwd || process.cwd();
		if (home && cwd.startsWith(home)) {
			cwd = `~${cwd.slice(home.length)}`;
		}
		if (this.info.branch) {
			cwd += ` (${this.info.branch})`;
		}
		return cwd;
	}

	private renderContext(): string {
		const tokens = Math.max(0, Math.round(this.info.contextTokens || 0));
		const maxTokens = this.info.contextMaxTokens;
		const ratio =
			maxTokens && maxTokens > 0 ? Math.min(1, tokens / maxTokens) : 0;
		const color =
			ratio >= 0.9
				? "\x1b[38;5;203m"
				: ratio >= 0.75
					? "\x1b[38;5;220m"
					: "\x1b[38;5;111m";
		const compacted = this.info.contextCompacted ? " compacted" : "";
		const text = maxTokens
			? `ctx ${formatTokenCount(tokens)}/${formatTokenCount(maxTokens)}${compacted}`
			: `ctx ${formatTokenCount(tokens)}${compacted}`;
		return `${color}${text}${RESET}`;
	}

	private truncateVisible(text: string, width: number): string {
		if (visibleWidth(text) <= width)
			return text + " ".repeat(Math.max(0, width - visibleWidth(text)));
		const ellipsis = "…";
		let out = "";
		let inEscape = false;
		let visible = 0;
		const target = Math.max(0, width - 1);
		for (let i = 0; i < text.length; i++) {
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
			if (visible + chWidth > target) break;
			out += ch;
			visible += chWidth;
		}
		return out + ellipsis + RESET;
	}

	private renderPhase(): string {
		const phaseColors: Record<string, string> = {
			ready: "\x1b[38;5;40m",
			thinking: "\x1b[38;5;220m",
			tool: "\x1b[38;5;141m",
			error: "\x1b[38;5;203m",
			streaming: "\x1b[38;5;111m",
			compacting: "\x1b[38;5;208m",
			branching: "\x1b[38;5;177m",
		};
		const color = phaseColors[this.info.phase] ?? "\x1b[38;5;240m";

		const phase = this.info.phase.toLowerCase();

		// Animated spinner for active phases.
		const SPINNING = [
			"streaming",
			"thinking",
			"tool",
			"compacting",
			"branching",
		];
		if (SPINNING.includes(phase)) {
			const spinners = ["◐", "◓", "◑", "◒"];
			const s = spinners[this.tick % spinners.length];
			return `${color}\x1b[1m${s} ${phase.toUpperCase()}${RESET}\x1b[0m`;
		}

		return `${color}\x1b[1m${phase.toUpperCase()}\x1b[0m`;
	}
}

interface StatusState {
	thinkingLevel: string;
	cacheEnabled: boolean;
	turnCount: number;
	messageCount: number;
	phase: string;
	model: string;
	cwd: string;
	branch: string;
	contextTokens: number;
	contextMaxTokens?: number;
	contextCompacted: boolean;
	reasoner: string;
	sessionTitle?: string;
}

const RESET = "\x1b[0m";
const BOLD = "\x1b[1m";
const DIM = "\x1b[2m";

function formatTokenCount(tokens: number): string {
	if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}m`;
	if (tokens >= 1000) return `${(tokens / 1000).toFixed(1)}k`;
	return String(tokens);
}
