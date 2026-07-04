// Steering advisor: suggest next actions based on loop detector state and history.

import type { LoopDetector } from "./loop-detector.ts";
import type { Message } from "./types.ts";

export interface SteeringHint {
	level: "info" | "warning" | "critical";
	message: string;
	suggestedAction?: string;
}

export class SteeringAdvisor {
	constructor(private loopDetector: LoopDetector) {}

	/** Suggest next action based on detector state. */
	getHints(messages: Message[]): SteeringHint[] {
		const hints: SteeringHint[] = [];

		// Check turn history for patterns
		if (messages.length > 20) {
			const recentTools = this.getRecentToolCalls(messages, 5);
			const toolSet = new Set(recentTools.map((t) => t.name));

			// Narrow tool set + many turns = focus issue
			if (toolSet.size <= 2 && recentTools.length > 4) {
				hints.push({
					level: "warning",
					message: "Focused on narrow set of tools for several turns",
					suggestedAction: "Try a different approach or tool",
				});
			}
		}

		// Check for repetition
		if (messages.length > 10) {
			const last3 = messages.slice(-6, -3);
			const last6 = messages.slice(-3);
			if (last3.length > 0 && last6.length > 0) {
				const text3 = JSON.stringify(last3);
				const text6 = JSON.stringify(last6);
				if (text3 === text6) {
					hints.push({
						level: "critical",
						message: "Last 3 turns identical to previous 3 turns",
						suggestedAction: "Stop and try a completely different approach",
					});
				}
			}
		}

		// Check assistant output trend
		const assistantMessages = messages.filter((m) => m.role === "assistant");
		if (assistantMessages.length >= 3) {
			const recent = assistantMessages.slice(-3);
			const avgLength = recent.reduce((sum, m) => sum + (m.content?.length || 0), 0) / 3;
			if (avgLength < 50) {
				hints.push({
					level: "warning",
					message: "Recent responses very short (avg < 50 chars)",
					suggestedAction: "Model may be struggling — try rephrasing the goal",
				});
			}
		}

		return hints;
	}

	private getRecentToolCalls(
		messages: Message[],
		count: number,
	): Array<{ name: string; result?: string }> {
		const toolCalls: Array<{ name: string; result?: string }> = [];
		for (let i = messages.length - 1; i >= 0 && toolCalls.length < count; i--) {
			const msg = messages[i];
			if (msg.role === "assistant" && msg.tool_calls) {
				for (const tc of msg.tool_calls) {
					toolCalls.unshift({ name: tc.name });
					if (toolCalls.length >= count) break;
				}
			}
		}
		return toolCalls;
	}
}
