import { expect, test } from "bun:test";
import {
	produceWidgets,
	type WidgetFactoryStatus,
} from "../footer/widget-factory.ts";
import { initTheme } from "../terminal/theme.ts";

const completeStatus: WidgetFactoryStatus = {
	thinkingLevel: "high",
	inferenceMode: "thinking-coding",
	cacheReadTokens: 1200,
	turnCount: 3,
	messageCount: 8,
	phase: "thinking",
	model: "test-model",
	cwd: "/workspace/logician",
	virtualEnv: "/workspace/logician/.venv",
	virtualEnvPythonVersion: "3.12",
	branch: "main",
	gitModified: 2,
	gitStaged: 1,
	gitUntracked: 1,
	gitCommit: "abc1234",
	gitAhead: 1,
	gitBehind: 2,
	gitAddedLines: 10,
	gitRemovedLines: 4,
	contextTokens: 42_000,
	contextMaxTokens: 128_000,
	contextCompacted: false,
	reasoner: "reviewer",
	goalCondition: "Modernize the runtime",
	goalTurnCount: 4,
	goalElapsed: 65,
	mcpServerCount: 2,
	sandboxMode: "code",
	workflowMode: "plan",
	executionProfile: "autonomous",
	promptTokens: 900,
	completionTokens: 300,
	rtkProxyEnabled: true,
	legroomEnabled: true,
	memoriamEnabled: true,
	graphicianEnabled: true,
	fffgrepEnabled: true,
	runtimeRetry: "1/3",
	activeSubagents: 2,
};

test("every implemented built-in widget renders under the typed theme contract", () => {
	initTheme("dark");
	const ids = new Set(produceWidgets(completeStatus).map(widget => widget.id));
	for (const expected of [
		"model",
		"thinking",
		"phase",
		"runtime-status",
		"context-bar",
		"context-capacity",
		"token-flow",
		"cache-read",
		"location",
		"virtual-env",
		"branch",
		"commit",
		"git-diff-added",
		"git-diff-removed",
		"git-status",
		"reasoner",
		"inference-mode",
		"sandbox",
		"permission",
		"mcp",
		"rtk",
		"legroom",
		"memoriam",
		"graphician",
		"fffgrep",
		"goal",
		"execution-profile",
	]) {
		expect(ids.has(expected)).toBe(true);
	}
});

test("token flow supports either side independently without unsafe assertions", () => {
	initTheme("dark");
	const widgets = produceWidgets({
		...completeStatus,
		promptTokens: undefined,
		completionTokens: 12,
	});
	const tokenFlow = widgets.find(widget => widget.id === "token-flow");
	expect(tokenFlow?.text).toContain("–");
	expect(tokenFlow?.text).toContain("12");
});
