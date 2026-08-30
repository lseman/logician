// ── Ink TUI Widget Factory ──────────────────────────────────────────────────
// Maps status state → typed WidgetData for each WidgetId. The layout/renderer
// layer then positions widgets per config.

import type { BuiltinWidgetId, WidgetData, WidgetId } from "./widget-types.ts";
import type { ThemeColor } from "../theme.ts";
import { DIM, RESET, colorText } from "./primitives.ts";
import { theme } from "../theme.ts";

/* ════════════════════════════════════════════════════════════════════════════
 *  StatusInfo — subset of TuiState fields needed by the factory
 * ════════════════════════════════════════════════════════════════════════════ */

export interface WidgetFactoryStatus {
	thinkingLevel: string;
	inferenceMode: string;
	cacheReadTokens?: number;
	turnCount: number;
	messageCount: number;
	phase: string;
	model: string;
	cwd: string;
	virtualEnv?: string;
	virtualEnvPythonVersion?: string;
	branch: string;
	gitModified?: number;
	gitStaged?: number;
	gitUntracked?: number;
	gitCommit?: string;
	gitAhead?: number;
	gitBehind?: number;
	gitAddedLines?: number;
	gitRemovedLines?: number;
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
	workflowMode?: "act" | "plan";
	executionProfile?: "autonomous" | "minimal";
	promptTokens?: number;
	completionTokens?: number;
	rtkProxyEnabled?: boolean;
	legroomEnabled?: boolean;
	memoriamEnabled?: boolean;
	graphicianEnabled?: boolean;
	fffgrepEnabled?: boolean;
	runPhase?: string;
	runtimeRetry?: string;
	runtimeRepair?: string;
	activeSubagents?: number;
	tick?: number; // 0-7 for spinner animation
}

/* ════════════════════════════════════════════════════════════════════════════
 *  Widget data helpers
 * ════════════════════════════════════════════════════════════════════════════ */

export type { WidgetData, WidgetId };

function empty(id: WidgetId): WidgetData {
	return { id, text: "", empty: true };
}

function withIcon(id: WidgetId, icon: string, text: string): WidgetData {
	return { id, icon, text };
}

function styled(
	id: WidgetId,
	color: string | undefined,
	label: string,
	value: string,
): WidgetData {
	const labelText = label ? `${DIM}${label}${RESET}` : "";
	return { id, text: `${labelText} ${colorText(color, value)}` };
}

function tokenStr(tokens: number): string {
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

/* ════════════════════════════════════════════════════════════════════════════
 *  Widget provider functions
 * ════════════════════════════════════════════════════════════════════════════ */

const PHASE_SPINNERS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];

const PHASE_LABELS: Record<string, { label: string; color: ThemeColor }> = {
	ready: { label: "READY", color: "success" },
	thinking: { label: "THINKING", color: "phaseThinking" },
	tool: { label: "TOOL", color: "phaseTool" },
	verifying: { label: "VERIFYING", color: "phaseThinking" },
	streaming: { label: "STREAMING", color: "phaseStreaming" },
	waiting: { label: "WAITING", color: "accent" },
	approval: { label: "APPROVAL", color: "warning" },
	failed: { label: "FAILED", color: "error" },
	compacting: { label: "COMPACTING", color: "phaseCompacting" },
	branching: { label: "BRANCHING", color: "phaseBranching" },
	cancelling: { label: "CANCELLING", color: "muted" },
	error: { label: "ERROR", color: "error" },
};

function phaseWidget(status: WidgetFactoryStatus): WidgetData {
	const tick = status.tick ?? 0;
	const spinner = PHASE_SPINNERS[tick % 8];
	const raw = status.phase || "ready";
	const info = PHASE_LABELS[raw] ?? {
		label: raw.toUpperCase(),
		color: "muted",
	};
	const withSpinner =
		raw === "thinking" ||
		raw === "tool" ||
		raw === "verifying" ||
		raw === "streaming" ||
		raw === "compacting" ||
		raw === "branching" ||
		raw === "cancelling"
			? `${spinner} ${info.label}`
			: raw === "waiting" || raw === "approval"
				? `◆ ${info.label}`
				: raw === "failed" || raw === "error"
					? `× ${info.label}`
					: `● ${info.label}`;
	const color = theme.fg(info.color) ?? "";
	return withIcon("phase", "", `${color}${withSpinner}${RESET}`);
}

function runtimeStatusWidget(status: WidgetFactoryStatus): WidgetData {
	const parts: string[] = [];
	if (status.runtimeRetry) parts.push(`retry ${status.runtimeRetry}`);
	if (status.runtimeRepair) parts.push(`repair ${status.runtimeRepair}`);
	if (status.activeSubagents) parts.push(`agents ${status.activeSubagents}`);
	if (parts.length === 0) return empty("runtime-status");
	return withIcon("runtime-status", "◈", colorText(theme.fg("accent"), parts.join(" ")));
}

function modelWidget(status: WidgetFactoryStatus): WidgetData {
	if (!status.model) return empty("model");
	const text = colorText(theme.fg("text"), status.model);
	return withIcon("model", "󰚩", text);
}

function thinkingWidget(status: WidgetFactoryStatus): WidgetData {
	const lvl = status.thinkingLevel;
	if (lvl === "off") {
		return styled("thinking", theme.fg("levelOff"), "think:", "off");
	}
	const levelColors: Record<string, string | undefined> = {
		low: theme.fg("levelLow"),
		medium: theme.fg("levelMedium"),
		high: theme.fg("levelHigh"),
		xhigh: theme.fg("levelXhigh"),
	};
	const color = levelColors[lvl] ?? theme.fg("accent");
	return styled("thinking", color, "think:", lvl.toUpperCase());
}

function contextBarWidget(status: WidgetFactoryStatus): WidgetData {
	const tokens = Math.max(0, Math.round(status.contextTokens || 0));
	const maxTokens = status.contextMaxTokens;
	if (!maxTokens || maxTokens === 0) return empty("context-bar");

	const ratio = Math.min(1, tokens / maxTokens);
	const pct = (ratio * 100).toFixed(1);

	const color =
		ratio >= 0.9
			? theme.fg("contextCritical")
			: ratio >= 0.75
				? theme.fg("contextWarning")
				: theme.fg("contextGood");

	const maxStr = tokenStr(maxTokens);
	const cells = 5;
	const filled = Math.min(cells, Math.max(0, Math.round(ratio * cells)));
	const meter = filled > 0 ? "▰".repeat(filled) : "";

	return styled(
		"context-bar",
		color,
		"ctx",
		`${meter} ${pct}%${RESET}${DIM}/${maxStr}${RESET}`,
	);
}

function contextCapacityWidget(status: WidgetFactoryStatus): WidgetData {
	if (!status.contextMaxTokens || status.contextMaxTokens === 0)
		return empty("context-capacity");
	const str = tokenStr(status.contextMaxTokens);
	return styled("context-capacity", theme.fg("dim"), "cap:", str);
}

function tokenFlowWidget(status: WidgetFactoryStatus): WidgetData {
	const hasIn = status.promptTokens !== undefined;
	const hasOut = status.completionTokens !== undefined;
	if (!hasIn && !hasOut) return empty("token-flow");

	const inStr =
		status.promptTokens === undefined ? "–" : tokenStr(status.promptTokens);
	const outStr =
		status.completionTokens === undefined
			? "–"
			: tokenStr(status.completionTokens);
	const color = theme.fg("accent") ?? "";
	return {
		id: "token-flow",
		text: `${DIM}↑${RESET} ${color}${inStr}${RESET}${DIM} │ ${RESET}${DIM}↓${RESET} ${color}${outStr}${RESET}`,
	};
}

function cacheReadWidget(status: WidgetFactoryStatus): WidgetData {
	const unknown = status.cacheReadTokens === undefined;
	const val =
		status.cacheReadTokens === undefined
			? "unknown"
			: tokenStr(status.cacheReadTokens);
	const color = theme.fg(unknown ? "dim" : "accent") ?? "";
	return styled("cache-read", color, "cache read:", val);
}

function cacheWriteWidget(): WidgetData {
	return empty("cache-write");
}

function cacheHitRateWidget(): WidgetData {
	return empty("cache-hit-rate");
}

function locationWidget(status: WidgetFactoryStatus): WidgetData {
	const home = process.env.HOME || "";
	let cwd = status.cwd || process.cwd();
	if (home && cwd.startsWith(home)) {
		cwd = `~${cwd.slice(home.length)}`;
	}
	const parts = cwd.split("/").filter(Boolean);
	const name = parts[parts.length - 1] || ".";
	return withIcon("location", "", colorText(theme.fg("text"), name));
}

function virtualEnvWidget(status: WidgetFactoryStatus): WidgetData {
	if (!status.virtualEnv) return empty("virtual-env");
	const normalized = status.virtualEnv.replace(/[\\/]+$/, "");
	const name = normalized.split(/[\\/]/).pop() || normalized;
	const version = status.virtualEnvPythonVersion
		? ` · py${status.virtualEnvPythonVersion}`
		: "";
	return styled(
		"virtual-env",
		theme.fg("success"),
		"venv:",
		`${name}${version}`,
	);
}

function branchWidget(status: WidgetFactoryStatus): WidgetData {
	if (!status.branch) return empty("branch");
	let text = `⎇ ${colorText(theme.fg("success"), status.branch)}`;
	if (status.gitModified)
		text += ` ${colorText(theme.fg("warning"), `*${status.gitModified}`)}`;
	if (status.gitStaged)
		text += ` ${theme.fg("success") + `+${status.gitStaged}` + RESET}`;
	if (status.gitUntracked)
		text += ` ${theme.fg("error") + `?${status.gitUntracked}` + RESET}`;
	return withIcon("branch", "", text);
}

function commitWidget(status: WidgetFactoryStatus): WidgetData {
	if (!status.gitCommit) return empty("commit");
	return styled("commit", theme.fg("muted"), "commit:", status.gitCommit);
}

function gitDiffAddedWidget(status: WidgetFactoryStatus): WidgetData {
	if (!status.gitAddedLines) return empty("git-diff-added");
	const val = `+${status.gitAddedLines}`;
	return withIcon("git-diff-added", "↗", colorText(theme.fg("success"), val));
}

function gitDiffRemovedWidget(status: WidgetFactoryStatus): WidgetData {
	if (!status.gitRemovedLines) return empty("git-diff-removed");
	return withIcon(
		"git-diff-removed",
		"↘",
		colorText(theme.fg("error"), `-${status.gitRemovedLines}`),
	);
}

function gitStatusWidget(status: WidgetFactoryStatus): WidgetData {
	const hasRepository = Boolean(
		status.branch ||
			status.gitCommit ||
			status.gitAhead ||
			status.gitBehind ||
			status.gitModified ||
			status.gitStaged ||
			status.gitUntracked,
	);
	if (!hasRepository) return empty("git-status");
	const parts: string[] = [];
	if (status.gitAhead) parts.push(`↑${status.gitAhead}`);
	if (status.gitBehind) parts.push(`↓${status.gitBehind}`);
	const dirty =
		(status.gitModified ?? 0) +
		(status.gitStaged ?? 0) +
		(status.gitUntracked ?? 0);
	if (dirty) parts.push(`${dirty} changed`);
	else parts.push("clean");
	return styled(
		"git-status",
		theme.fg(dirty ? "warning" : "success"),
		"git:",
		parts.join(" "),
	);
}

function pullRequestWidget(): WidgetData {
	return empty("pull-request");
}

function pullRequestReviewThreadsWidget(): WidgetData {
	return empty("pull-request-review-threads");
}

function pullRequestCiStatusWidget(): WidgetData {
	return empty("pull-request-ci-status");
}

function reasonerWidget(status: WidgetFactoryStatus): WidgetData {
	return styled(
		"reasoner",
		theme.fg("muted"),
		"reasoner:",
		status.reasoner === "none" ? "off" : status.reasoner,
	);
}

function inferenceModeWidget(status: WidgetFactoryStatus): WidgetData {
	const mode = status.inferenceMode;
	if (!mode) return empty("inference-mode");
	const labels: Record<string, string> = {
		"thinking-general": "THINK GEN",
		"thinking-coding": "THINK CODE",
		"instruct-general": "INSTRUCT",
		"instruct-reasoning": "REASON",
		none: "PROVIDER",
	};
	const label = labels[mode] ?? mode.toUpperCase();
	return styled("inference-mode", theme.fg("accent"), "mode:", label);
}

function sandboxWidget(status: WidgetFactoryStatus): WidgetData {
	const mode = status.sandboxMode ?? "code";
	if (mode === "none") {
		return styled("sandbox", theme.fg("levelOff"), "sandbox:", "off");
	}
	return styled("sandbox", theme.fg("accent"), "sandbox:", mode);
}

function permissionWidget(status: WidgetFactoryStatus): WidgetData {
	const mode = status.workflowMode ?? "act";
	return mode === "plan"
		? withIcon("permission", "", colorText(theme.fg("warning"), "plan"))
		: withIcon("permission", "", colorText(theme.fg("success"), "act"));
}

function mcpWidget(status: WidgetFactoryStatus): WidgetData {
	if (!status.mcpServerCount && !status.mcpLoading) return empty("mcp");
	if (status.mcpLoading) {
		return styled("mcp", theme.fg("warning"), "mcp:", "loading…");
	}
	const count = status.mcpServerCount || 0;
	return styled("mcp", theme.fg("accent"), "mcp:", `${count}`);
}

function rtkWidget(status: WidgetFactoryStatus): WidgetData {
	if (!status.rtkProxyEnabled) return empty("rtk");
	return styled("rtk", theme.fg("accent"), "rtk", "on");
}

function toggleWidget(
	id: WidgetId,
	label: string,
	enabled: boolean,
): WidgetData {
	return styled(
		id,
		theme.fg(enabled ? "success" : "dim"),
		`${label}:`,
		enabled ? "on" : "off",
	);
}

function graphicianWidget(status: WidgetFactoryStatus): WidgetData {
	return toggleWidget("graphician", "gph", status.graphicianEnabled ?? true);
}

function legroomWidget(status: WidgetFactoryStatus): WidgetData {
	return styled(
		"legroom",
		theme.fg((status.legroomEnabled ?? false) ? "success" : "dim"),
		"leg:",
		(status.legroomEnabled ?? false) ? "sdk" : "off",
	);
}

function memoriamWidget(status: WidgetFactoryStatus): WidgetData {
	return styled(
		"memoriam",
		theme.fg((status.memoriamEnabled ?? false) ? "success" : "dim"),
		"mem:",
		(status.memoriamEnabled ?? false) ? "sdk" : "off",
	);
}

function fffgrepWidget(status: WidgetFactoryStatus): WidgetData {
	return toggleWidget("fffgrep", "fff", status.fffgrepEnabled ?? true);
}

function totalCostWidget(): WidgetData {
	return empty("total-cost");
}

function goalWidget(status: WidgetFactoryStatus): WidgetData {
	const cond = status.goalCondition;
	if (!cond) return empty("goal");
	const turns = status.goalTurnCount || 0;
	const elapsed = status.goalElapsed || 0;
	const mins = Math.floor(elapsed / 60);
	const secs = elapsed % 60;
	const timeStr = mins > 0 ? `${mins}m${secs}s` : `${secs}s`;
	const maxLen = 24;
	const truncated = cond.length > maxLen ? `${cond.slice(0, maxLen)}…` : cond;
	return styled(
		"goal",
		theme.fg("accent"),
		"◎",
		`${truncated} (${turns} turns, ${timeStr})`,
	);
}

function executionProfileWidget(status: WidgetFactoryStatus): WidgetData {
	const profile = status.executionProfile ?? "minimal";
	if (profile === "minimal") {
		return styled(
			"execution-profile",
			theme.fg("warning"),
			"exec:",
			"minimal",
		);
	}
	return styled("execution-profile", theme.fg("success"), "exec:", "auto");
}

/* ════════════════════════════════════════════════════════════════════════════
 *  Provider registry — maps WidgetId → provider function
 * ════════════════════════════════════════════════════════════════════════════ */

const PROVIDERS: Record<
	BuiltinWidgetId,
	(status: WidgetFactoryStatus) => WidgetData
> = {
	model: modelWidget,
	thinking: thinkingWidget,
	phase: phaseWidget,
	"runtime-status": runtimeStatusWidget,
	"context-bar": contextBarWidget,
	"context-capacity": contextCapacityWidget,
	"token-flow": tokenFlowWidget,
	"cache-read": cacheReadWidget,
	"cache-write": cacheWriteWidget,
	"cache-hit-rate": cacheHitRateWidget,
	location: locationWidget,
	"virtual-env": virtualEnvWidget,
	branch: branchWidget,
	commit: commitWidget,
	"git-diff-added": gitDiffAddedWidget,
	"git-diff-removed": gitDiffRemovedWidget,
	"git-status": gitStatusWidget,
	"pull-request": pullRequestWidget,
	"pull-request-review-threads": pullRequestReviewThreadsWidget,
	"pull-request-ci-status": pullRequestCiStatusWidget,
	reasoner: reasonerWidget,
	"inference-mode": inferenceModeWidget,
	sandbox: sandboxWidget,
	permission: permissionWidget,
	mcp: mcpWidget,
	rtk: rtkWidget,
	legroom: legroomWidget,
	memoriam: memoriamWidget,
	graphician: graphicianWidget,
	fffgrep: fffgrepWidget,
	goal: goalWidget,
	"execution-profile": executionProfileWidget,
	"total-cost": totalCostWidget,
};

/** Produce all enabled widgets from status state. Returns flat list of
 * WidgetData in the order determined by their config (row + position). */
export function produceWidgets(status: WidgetFactoryStatus): WidgetData[] {
	const results: WidgetData[] = [];
	for (const id of Object.keys(PROVIDERS) as BuiltinWidgetId[]) {
		try {
			const data = PROVIDERS[id](status);
			if (!data.empty) {
				results.push(data);
			}
		} catch {
			// Widget provider crashed — skip it silently
		}
	}
	return results;
}
