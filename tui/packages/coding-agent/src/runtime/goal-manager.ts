// ── Goal Manager ────────────────────────────────────────────────────────────────
// Codex-style /goal: set a completion condition, evaluator checks after each turn.
// Loop runs until condition met or turn/time limit reached.

export type GoalStatus = "idle" | "active" | "achieved" | "cancelled";

export interface GoalState {
	readonly condition: string;
	readonly status: GoalStatus;
	readonly turnCount: number;
	readonly maxTurns?: number;
	readonly startedAt: number;
	readonly achievedAt?: number;
	readonly lastReason?: string;
}

export type GoalAction =
	| { type: "set"; condition: string; maxTurns?: number }
	| { type: "clear" };

export type GoalStateHandler = (state: Readonly<GoalState> | null) => void;

const DEFAULT_MAX_TURNS = 50;
const EVALUATOR_SYSTEM_PROMPT = `You are a goal evaluator. The user has set a completion condition for an autonomous coding agent.

Given the conversation so far, decide whether the condition is met.

Respond with EXACTLY one line:
- "YES: <brief reason>" if the condition is met
- "NO: <brief reason>" if the condition is NOT met

Be strict. Only say YES when the condition is clearly satisfied. If uncertain, say NO.`;

export class GoalManager {
	private state: GoalState | null = null;
	private onStateChange?: GoalStateHandler;

	setOnStateChange(cb: GoalStateHandler): void {
		this.onStateChange = cb;
	}

	/** Start a goal. Runs the first evaluation immediately. */
	set(condition: string, maxTurns?: number): void {
		const normalized = condition.trim();
		if (!normalized) throw new Error("Goal condition cannot be empty");
		if (normalized.length > 4000)
			throw new Error("Goal condition must be ≤4000 characters");

		this.cancel();
		const now = Date.now();
		this.state = {
			condition: normalized,
			status: "active",
			turnCount: 0,
			maxTurns: maxTurns ?? DEFAULT_MAX_TURNS,
			startedAt: now,
		};
		this.notify();
	}

	/** Stop and cancel the active goal. */
	cancel(): void {
		if (this.state) {
			this.state = { ...this.state, status: "cancelled" };
			this.notify();
		}
	}

	/** Mark goal as achieved. */
	achieve(reason: string): void {
		if (!this.state) return;
		const now = Date.now();
		this.state = {
			...this.state,
			status: "achieved",
			achievedAt: now,
			lastReason: reason,
		};
		this.notify();
	}

	/** Record one completed evaluator pass without cancelling an unmet goal. */
	recordEvaluation(met: boolean, reason: string): void {
		if (!this.state || this.state.status !== "active") return;
		const turnCount = this.state.turnCount + 1;
		if (met) {
			this.state = { ...this.state, turnCount };
			this.achieve(reason);
			return;
		}
		if (this.state.maxTurns && turnCount >= this.state.maxTurns) {
			this.state = {
				...this.state,
				turnCount,
				status: "cancelled",
				lastReason: `Reached ${this.state.maxTurns} turn limit: ${reason}`,
			};
			this.notify();
			return;
		}
		this.state = { ...this.state, turnCount, lastReason: reason };
		this.notify();
	}

	isActive(): boolean {
		return this.state?.status === "active";
	}

	isSet(): boolean {
		return this.state !== null;
	}

	getState(): Readonly<GoalState> | null {
		return this.state ? Object.freeze({ ...this.state }) : null;
	}

	handleAction(action: GoalAction): void {
		if (action.type === "set") this.set(action.condition, action.maxTurns);
		else if (action.type === "clear") this.cancel();
	}

	/** Parse a condition string, extracting optional "or stop after N turns" clause. */
	static parseCondition(text: string): { condition: string; maxTurns?: number } {
		const turnMatch = text.match(/or\s+stop\s+after\s+(\d+)\s+turns?/i);
		const maxTurns = turnMatch ? Number(turnMatch[1]) : undefined;
		const condition = turnMatch ? text.replace(turnMatch[0], "").trim() : text.trim();
		return { condition: condition || text, maxTurns };
	}

	private notify(): void {
		this.onStateChange?.(this.getState());
	}

	/** Build the evaluator prompt from condition + conversation snapshot. */
	static buildEvaluatorPrompt(condition: string, conversationSnapshot: string): string {
		return `${EVALUATOR_SYSTEM_PROMPT}\n\n---\n\nGoal condition:\n${condition}\n\n---\n\nConversation so far:\n${conversationSnapshot}\n\n---\n\nEvaluate the condition against the conversation above.`;
	}

	/** Parse evaluator response: "YES: reason" or "NO: reason". */
	static parseEvaluatorResponse(response: string): { met: boolean; reason: string } {
		const trimmed = response.trim();
		if (trimmed.toUpperCase().startsWith("YES")) {
			const reason = trimmed.replace(/^[Yy][Ee][Ss]\s*:\s*/, "").trim();
			return { met: true, reason: reason || "Condition appears to be met." };
		}
		const reason = trimmed.replace(/^[Nn][Oo]\s*:\s*/, "").trim();
		return { met: false, reason: reason || "Condition not yet met." };
	}
}
