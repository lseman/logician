// ── Adaptive inference mode ──────────────────────────────────────────────────
// Picks a concrete sampling preset for `inferenceMode: "auto"` from cheap,
// already-tracked loop signals — no durable task ledger required.

import type { InferenceMode } from "../../core/types/types-config.ts";

export interface AdaptiveModeDecision {
	mode: Exclude<InferenceMode, "auto">;
	reason: string;
}

export interface AdaptiveModeSignals {
	/** The user's objective for this run (last meaningful prompt). */
	objective: string;
	/** Whether a non-task_status tool has been called yet this run. */
	performedToolWork: boolean;
	/** Tool calls whose result matched a failure pattern, this run. */
	toolFailures: number;
}

const FAILURE_PATTERN =
	/(?:\bfailed\b|\bfailure\b|\b[1-9]\d* fails?\b|\berror(?:\s*:|$)|exception|traceback|not ok|exit(?:ed)? (?:code )?[1-9])/i;

/** Whether a tool result reads as a failure, for adaptive-mode failure counting. */
export function isToolFailureResult(result: string): boolean {
	return FAILURE_PATTERN.test(result);
}

export function selectAdaptiveMode(
	signals: AdaptiveModeSignals,
): AdaptiveModeDecision {
	const objective = signals.objective.toLowerCase();

	if (signals.toolFailures >= 2) {
		return {
			mode: "thinking-coding",
			reason: "recovery after repeated tool failures",
		};
	}
	if (signals.performedToolWork) {
		return {
			mode: "instruct-coding",
			reason: "tool work underway favors precise code generation",
		};
	}
	if (
		/\b(?:brainstorm|creative|ideas?|name|design alternatives?)\b/.test(
			objective,
		)
	) {
		return {
			mode: "creative",
			reason: "objective requests ideation or alternatives",
		};
	}
	if (
		/\b(?:debug|diagnos|review|analy[sz]|compare|investigat|why)\b/.test(
			objective,
		)
	) {
		return {
			mode: "analytical",
			reason: "objective is primarily diagnostic or analytical",
		};
	}
	if (
		/\b(?:implement|fix|code|refactor|build|add|change|test)\b/.test(
			objective,
		)
	) {
		return {
			mode: "thinking-coding",
			reason: "coding objective still requires orientation",
		};
	}
	return {
		mode: "instruct-general",
		reason: "general objective with no elevated reasoning signal",
	};
}

/** Last meaningful user prompt, ignoring bare continuation nudges. */
export function taskObjectiveFromMessages(
	messages: Array<{ role: string; content: unknown }>,
): string {
	const prompts = messages
		.filter(message => message.role === "user" && typeof message.content === "string")
		.map(message => String(message.content).replace(/\s+/g, " ").trim().slice(0, 1000))
		.filter(Boolean);
	const meaningful = prompts.filter(
		prompt =>
			!/^(?:continue|resume|go on|keep going)[.! ]*$/i.test(prompt) &&
			!/^\[continuation-nudge:/i.test(prompt),
	);
	return meaningful.at(-1) ?? prompts.at(-1) ?? "";
}
