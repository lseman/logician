// ── TodoTracker — phased todo enforcement ─────────────────────────────────────
// Emulates oh-my-pi's TodoTracker: eager prelude, mid-run nudge, completion reminder.

import type { Message, SoftToolRequirement } from "@logician/log-core";
import type { TodoCompletionTransition } from "@logician/log-core/events";

// ── Phased data types (local, matching TodosEvent shape) ─────────────────────

interface TodoPhase {
	name: string;
	tasks: Array<{
		content: string;
		status: "pending" | "in_progress" | "completed" | "abandoned";
		blocker?: string;
	}>;
}

// ── Host interface ────────────────────────────────────────────────────────────

export interface TodoTrackerHost {
	getActiveToolNames(): string[];
	getMutatingToolNames(): string[];
	getCurrentPromptText(): string | undefined;
	isFirstTurn(): boolean;
	isPlanMode(): boolean;
	modelSupportsToolChoice(): boolean;
	steer(message: string): void;
}

// ── Configuration ─────────────────────────────────────────────────────────────

const MID_RUN_NUDGE_MUTATION_THRESHOLD = 12;
const MID_RUN_NUDGE_MAX_PER_CYCLE = 2;
const REMINDERS_MAX = 3;

// ── Message builders ──────────────────────────────────────────────────────────

function buildEagerTodoPrelude(): Message {
	return {
		role: "system",
		content: `<system-reminder>
Before substantive work, create a phased todo.

You MUST call \`todo\` first in this turn.
You MUST initialize the todo list with a single \`init\` op.
You MUST cover the entire request from investigation through implementation and verification — not just the next immediate step.
Task descriptions MUST be concise, specific 5-10 word labels.
The \`init\` op only accepts phase names and task-label strings; do not invent task metadata fields.

After \`todo\` succeeds, continue the request in the same turn.
NEVER call \`todo\` again unless task state has materially changed.
</system-reminder>`,
	};
}

function buildMidRunNudge(incompleteCount: number): Message {
	const plural = incompleteCount !== 1;
	return {
		role: "system",
		content: `<system-reminder>
${incompleteCount} todo item${plural ? "s" : ""} still open. If you finished a task since last \`todo\` update, mark it done now so progress stays visible; otherwise keep working.
</system-reminder>`,
	};
}

function buildCompletionReminder(
	phases: TodoPhase[],
	reminderNum: number,
): Message {
	const incompleteByPhase = phases
		.map(phase => ({
			name: phase.name,
			tasks: phase.tasks.filter(
				t => t.status !== "completed" && t.status !== "abandoned",
			),
		}))
		.filter(phase => phase.tasks.length > 0);

	const todoList = incompleteByPhase
		.map(
			phase =>
				`- ${phase.name}\n${phase.tasks.map(task => `  - ${task.content}`).join("\n")}`,
		)
		.join("\n");

	const totalIncomplete = incompleteByPhase.reduce(
		(s, p) => s + p.tasks.length,
		0,
	);

	return {
		role: "system",
		content: `<system-reminder>
You stopped with ${totalIncomplete} incomplete todo item(s):

${todoList}

Please continue working on these tasks or mark them complete if finished.
(Reminder ${reminderNum}/${REMINDERS_MAX})
</system-reminder>`,
	};
}

// ── Main class ────────────────────────────────────────────────────────────────

export class TodoTracker {
	#phases: TodoPhase[] = [];
	#reminderCount = 0;
	#reminderAwaitingProgress = false;
	#mutationsSinceLastTouch = 0;
	#midRunNudgeCount = 0;

	constructor(private host: TodoTrackerHost) {}

	/** Returns the current phase list. */
	getPhases(): TodoPhase[] {
		return this.#clonePhases(this.#phases);
	}

	/** Replaces todo phases with a new list. Returns newly completed tasks. */
	setPhases(
		phases: TodoPhase[],
		completedTasks?: TodoCompletionTransition[],
	): TodoCompletionTransition[] | undefined {
		const prev = new Set<string>();
		for (const p of this.#phases) {
			for (const t of p.tasks) {
				if (t.status === "completed" || t.status === "abandoned") {
					prev.add(`${p.name}\x00${t.content}`);
				}
			}
		}

		this.#phases = this.#clonePhases(phases);

		// Detect newly completed tasks from completedTasks param.
		const newCompleted: TodoCompletionTransition[] = [];
		if (completedTasks) {
			for (const ct of completedTasks) {
				const key = `${ct.phase}\x00${ct.content}`;
				if (!prev.has(key)) {
					newCompleted.push(ct);
				}
			}
		}
		// Also check for newly completed tasks not in completedTasks.
		for (const p of this.#phases) {
			for (const t of p.tasks) {
				if (t.status === "completed" || t.status === "abandoned") {
					const key = `${p.name}\x00${t.content}`;
					if (!prev.has(key) && !newCompleted.some(
						ct => `${ct.phase}\x00${ct.content}` === key,
					)) {
						newCompleted.push({ phase: p.name, content: t.content });
					}
				}
			}
		}

		return newCompleted.length > 0 ? newCompleted : undefined;
	}

	/** Resets per-prompt reminder and mutation budgets. */
	resetCycle(): void {
		this.#reminderCount = 0;
		this.#reminderAwaitingProgress = false;
		this.#mutationsSinceLastTouch = 0;
		this.#midRunNudgeCount = 0;
	}

	// -- Eager prelude --

	createEagerTodoPrelude(): Message | undefined {
		if (this.host.isPlanMode()) return undefined;
		if (this.#phases.length > 0) return undefined;
		if (!this.host.modelSupportsToolChoice()) return undefined;
		const promptText = this.host.getCurrentPromptText();
		if (promptText !== undefined) {
			const trimmed = promptText.trimEnd();
			if (trimmed.endsWith("?") || trimmed.endsWith("!")) return undefined;
		}
		const active = this.host.getActiveToolNames();
		if (!active.includes("todo")) return undefined;
		return buildEagerTodoPrelude();
	}

	createEagerTodoRequirement(): SoftToolRequirement | undefined {
		const prelude = this.createEagerTodoPrelude();
		if (!prelude) return undefined;
		return {
			soft: true,
			id: "eager-todo",
			toolName: "todo",
			reminder: [prelude],
		};
	}

	// -- Mid-run nudge --

	onMutatingToolResult(): void {
		this.#mutationsSinceLastTouch++;
	}

	onTodoToolResult(): void {
		this.#mutationsSinceLastTouch = 0;
	}

	createMidRunNudge(): Message | undefined {
		if (
			this.#mutationsSinceLastTouch < MID_RUN_NUDGE_MUTATION_THRESHOLD ||
			this.#midRunNudgeCount >= MID_RUN_NUDGE_MAX_PER_CYCLE
		) {
			return undefined;
		}
		const active = this.host.getActiveToolNames();
		if (!active.includes("todo")) return undefined;
		const incomplete = this.#countIncomplete();
		if (incomplete === 0) return undefined;
		this.#midRunNudgeCount++;
		return buildMidRunNudge(incomplete);
	}

	// -- Completion check --

	checkCompletion(): boolean {
		if (this.host.isPlanMode()) return false;
		if (this.#reminderAwaitingProgress) return false;
		const active = this.host.getActiveToolNames();
		if (!active.includes("todo")) return false;
		const phases = this.#phases;
		if (phases.length === 0) return false;

		const incompleteByPhase = phases
			.map(phase => ({
				name: phase.name,
				tasks: phase.tasks.filter(
					t => t.status !== "completed" && t.status !== "abandoned",
				),
			}))
			.filter(phase => phase.tasks.length > 0);

		if (incompleteByPhase.length === 0) {
			this.#reminderCount = 0;
			this.#reminderAwaitingProgress = false;
			return false;
		}

		if (this.#reminderCount >= REMINDERS_MAX) return false;

		const reminder = buildCompletionReminder(
			phases,
			this.#reminderCount + 1,
		);
		if (reminder.content) this.host.steer(reminder.content);
		this.#reminderCount++;
		this.#reminderAwaitingProgress = true;
		return true;
	}

	syncFromTranscript(): void {
		// Placeholder — will be wired to transcript when session state is available.
	}

	#clonePhases(phases: TodoPhase[]): TodoPhase[] {
		return phases.map(phase => ({
			...phase,
			tasks: [...phase.tasks],
		}));
	}

	#countIncomplete(): number {
		return this.#phases.reduce(
			(sum, phase) =>
				sum +
				phase.tasks.filter(
					t => t.status !== "completed" && t.status !== "abandoned",
				).length,
			0,
		);
	}
}
