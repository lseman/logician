// ── Goal evaluation ──────────────────────────────────────────────────────
// Evaluates an active /goal condition against the conversation transcript.
// Calls the LLM directly (bypassing the bridge) and mixes goal logic,
// transcript access, bridge config, and fetch.

import {
	type AgentRuntime,
	type GoalState,
	GoalTracker,
} from "@logician/log-runtime/application";
import type { Transcript } from "@logician/log-runtime/sessions";
import type { TranscriptDisplay } from "../rendering/transcript/display.ts";
import type { TuiHandle } from "../terminal/core.ts";

export interface GoalRunnerCtx {
	bridge: AgentRuntime;
	transcript: Transcript;
	transcriptDisplay: TranscriptDisplay;
	tui: TuiHandle;
	goalManager: GoalTracker;
	goalActive: boolean;
	goalEvaluationPending: boolean;
}

export async function evaluateGoal(
	ctx: GoalRunnerCtx,
	goalState: Readonly<GoalState>,
): Promise<void> {
	if (ctx.goalEvaluationPending) return;
	ctx.goalEvaluationPending = true;
	// Build conversation snapshot from transcript turns
	const turns = ctx.transcript.getTurns();
	const snapshot = turns
		.map(t => {
			const parts: string[] = [];
			if (t.userMessage) parts.push(`User: ${t.userMessage}`);
			if (t.assistantMessage) parts.push(`Assistant: ${t.assistantMessage}`);
			return parts.join("\n");
		})
		.filter(Boolean)
		.join("\n\n");

	const evaluatorPrompt = GoalTracker.buildEvaluatorPrompt(
		goalState.condition,
		snapshot,
	);

	ctx.transcript.handleEvent({
		type: "notice",
		level: "info",
		label: "Goal evaluation",
		text: `Evaluation #${goalState.turnCount}: "${goalState.condition}"`,
	});
	ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
	ctx.tui.requestRender();

	// Call LLM directly for evaluation (like dropper.ts does)
	const { baseUrl, model } = ctx.bridge.getConfig();
	const apiKey =
		process.env.ANTHROPIC_API_KEY ??
		process.env.OPENAI_API_KEY ??
		process.env.LLM_API_KEY ??
		"sk-no-key";

	let response: string;
	try {
		const res = await fetch(
			`${(baseUrl ?? "https://api.openai.com").replace(/\/+$/, "")}/v1/chat/completions`,
			{
				method: "POST",
				headers: {
					"Content-Type": "application/json",
					Authorization: `Bearer ${apiKey}`,
					"x-api-key": apiKey,
				},
				body: JSON.stringify({
					model: model || "gpt-4o",
					messages: [{ role: "system", content: evaluatorPrompt }],
					max_tokens: 256,
					temperature: 0,
				}),
			},
		);

		if (!res.ok) {
			const errText = await res.text().catch(() => "");
			throw new Error(`LLM API error ${res.status}: ${errText.slice(0, 200)}`);
		}

		const data = (await res.json()) as {
			choices: Array<{ message: { content: string } }>;
		};
		response = data.choices?.[0]?.message?.content ?? "";
	} catch (e: unknown) {
		const err = e instanceof Error ? e.message : String(e);
		ctx.goalManager.handleAction({ type: "clear" });
		ctx.goalActive = false;
		ctx.transcript.handleEvent({
			type: "notice",
			level: "error",
			label: "Goal stopped",
			text: `Evaluation failed: ${err}. Goal cancelled.`,
		});
		ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
		ctx.tui.requestRender();
		ctx.goalEvaluationPending = false;
		return;
	}

	const { met, reason } = GoalTracker.parseEvaluatorResponse(response);

	if (met) {
		ctx.goalManager.recordEvaluation(true, reason);
		ctx.goalActive = false;
		ctx.transcript.handleEvent({
			type: "notice",
			level: "success",
			label: "Goal achieved",
			text: `"${goalState.condition}" — ${reason}`,
		});
	} else {
		ctx.goalManager.recordEvaluation(false, reason);
		const stillActive = ctx.goalManager.isActive();
		ctx.goalActive = stillActive;
		ctx.transcript.handleEvent({
			type: "notice",
			level: stillActive ? "warn" : "error",
			label: stillActive ? "Goal continuing" : "Goal stopped",
			text: stillActive
				? `${reason} — continuing...`
				: ctx.goalManager.getState()?.lastReason || reason,
		});
		if (stillActive) {
			const reminder = `Goal reminder: "${goalState.condition}". ${reason}. Continue working toward the goal.`;
			void ctx.bridge.sendMessage(reminder).catch((error: unknown) => {
				ctx.bridge.events.reportError(error);
			});
		}
	}

	ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
	ctx.tui.requestRender();
	ctx.goalEvaluationPending = false;
}
