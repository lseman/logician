import type { InferenceMode } from "../configuration/inference-modes.ts";
import type { Message, ToolCall } from "../types.ts";

export type TaskPhase =
	| "orient"
	| "investigate"
	| "implement"
	| "verify"
	| "handoff"
	| "blocked";

export interface TaskEvidence {
	kind: "observation" | "change" | "verification" | "failure";
	summary: string;
	tool?: string;
	iteration: number;
}

export interface ExplicitTaskState {
	objective: string;
	phase: TaskPhase;
	hypotheses: string[];
	evidence: TaskEvidence[];
	changedFiles: string[];
	verification: Array<{ command: string; passed: boolean; summary: string }>;
	blockers: string[];
	toolCalls: number;
	toolFailures: number;
}

export interface AdaptiveModeDecision {
	mode: Exclude<InferenceMode, "auto">;
	reason: string;
}

/** Serialize task state exactly as it is injected into provider context. */
export function formatTaskStateContext(state: ExplicitTaskState): string {
	const recent = state.evidence.slice(-6);
	return [
		"<task_state>",
		`objective: ${state.objective}`,
		`phase: ${state.phase}`,
		`progress: ${state.toolCalls} tool calls, ${state.toolFailures} failures`,
		`changed_files: ${state.changedFiles.join(", ") || "none"}`,
		`verification: ${state.verification.map(item => `${item.passed ? "pass" : "fail"} ${item.command}`).join("; ") || "none"}`,
		`blockers: ${state.blockers.join("; ") || "none"}`,
		"recent_evidence:",
		...(recent.length
			? recent.map(
					item =>
						`- [${item.kind}] ${item.tool ? `${item.tool}: ` : ""}${item.summary}`,
				)
			: ["- none"]),
		"Use this state as a concise ledger. Do not repeat it to the user. Advance the phase through evidence-backed work.",
		"</task_state>",
	].join("\n");
}

const WRITE_TOOLS = new Set([
	"edit_file",
	"write_file",
	"write_file_append",
	"apply_patch",
]);
const READ_TOOLS = new Set([
	"read_file",
	"grep",
	"glob",
	"list_files",
	"web_search",
]);
const VERIFY_PATTERN =
	/(?:^|\b)(?:test|check|lint|typecheck|build|compile|pytest|vitest|jest|cargo test)(?:\b|$)/i;
const FAILURE_PATTERN =
	/(?:\bfailed\b|\bfailure\b|\b[1-9]\d* fails?\b|\berror(?:\s*:|$)|exception|traceback|not ok|exit(?:ed)? (?:code )?[1-9])/i;

function compact(value: string, limit = 240): string {
	return value.replace(/\s+/g, " ").trim().slice(0, limit);
}

/** Resolve the durable objective across user turns and internal continuations. */
export function taskObjectiveFromMessages(messages: Message[]): string {
	const prompts = messages
		.filter(
			message => message.role === "user" && typeof message.content === "string",
		)
		.map(message => compact(String(message.content), 1000))
		.filter(Boolean);
	const meaningful = prompts.filter(
		prompt =>
			!/^(?:continue|resume|go on|keep going)[.! ]*$/i.test(prompt) &&
			!/^\[continuation-nudge:/i.test(prompt),
	);
	return meaningful.at(-1) ?? prompts.at(-1) ?? "";
}

function stringArg(
	args: Record<string, unknown>,
	...keys: string[]
): string | undefined {
	for (const key of keys) {
		const value = args[key];
		if (typeof value === "string" && value.trim()) return value.trim();
	}
	return undefined;
}

export class TaskStateController {
	private readonly state: ExplicitTaskState;

	constructor(objective: string) {
		this.state = {
			objective: compact(objective, 1000),
			phase: "orient",
			hypotheses: [],
			evidence: [],
			changedFiles: [],
			verification: [],
			blockers: [],
			toolCalls: 0,
			toolFailures: 0,
		};
	}

	snapshot(): ExplicitTaskState {
		return structuredClone(this.state);
	}

	recordToolBatch(
		calls: ToolCall[],
		results: Message[],
		iteration: number,
	): void {
		for (let index = 0; index < calls.length; index++) {
			const call = calls[index];
			const result = compact(String(results[index]?.content ?? ""));
			let args: Record<string, unknown> = {};
			try {
				args = JSON.parse(call.arguments) as Record<string, unknown>;
			} catch {
				// Malformed arguments are already handled by the tool controller.
			}
			this.state.toolCalls++;
			const command = stringArg(args, "command", "cmd") ?? "";
			const failed = FAILURE_PATTERN.test(result);
			if (failed) {
				this.state.toolFailures++;
				this.pushEvidence(
					"failure",
					result || `${call.name} failed`,
					call.name,
					iteration,
				);
			} else if (WRITE_TOOLS.has(call.name)) {
				const file = stringArg(args, "path", "file", "filePath");
				if (file && !this.state.changedFiles.includes(file))
					this.state.changedFiles.push(file);
				this.pushEvidence(
					"change",
					result || `${call.name} completed`,
					call.name,
					iteration,
				);
			} else {
				this.pushEvidence(
					"observation",
					result || `${call.name} completed`,
					call.name,
					iteration,
				);
			}

			if (call.name === "bash" && VERIFY_PATTERN.test(command)) {
				this.state.verification.push({
					command,
					passed: !failed,
					summary: result,
				});
			}
		}
		this.updatePhase(calls);
	}

	markBlocked(reason: string): void {
		const summary = compact(reason);
		if (summary && !this.state.blockers.includes(summary))
			this.state.blockers.push(summary);
		this.state.phase = "blocked";
	}

	markHandoff(): void {
		this.state.phase = "handoff";
	}

	selectAdaptiveMode(): AdaptiveModeDecision {
		const objective = this.state.objective.toLowerCase();
		if (this.state.phase === "blocked" || this.state.toolFailures >= 2) {
			return {
				mode: "thinking-coding",
				reason: "recovery after repeated tool failures or a blocker",
			};
		}
		if (this.state.phase === "verify") {
			return {
				mode: "deterministic",
				reason: "verification favors reproducible output",
			};
		}
		if (this.state.phase === "implement") {
			return {
				mode: "instruct-coding",
				reason: "implementation phase favors precise code generation",
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

	toContext(): string {
		return formatTaskStateContext(this.state);
	}

	private pushEvidence(
		kind: TaskEvidence["kind"],
		summary: string,
		tool: string,
		iteration: number,
	): void {
		this.state.evidence.push({
			kind,
			summary: compact(summary),
			tool,
			iteration,
		});
		if (this.state.evidence.length > 30) this.state.evidence.shift();
	}

	private updatePhase(calls: ToolCall[]): void {
		if (calls.some(call => WRITE_TOOLS.has(call.name))) {
			this.state.phase = "implement";
			return;
		}
		if (
			calls.some(call => {
				if (call.name !== "bash") return false;
				try {
					const args = JSON.parse(call.arguments) as Record<string, unknown>;
					return VERIFY_PATTERN.test(stringArg(args, "command", "cmd") ?? "");
				} catch {
					return false;
				}
			})
		) {
			this.state.phase = "verify";
			return;
		}
		if (calls.some(call => READ_TOOLS.has(call.name) || call.name === "bash")) {
			this.state.phase = "investigate";
		}
	}
}
