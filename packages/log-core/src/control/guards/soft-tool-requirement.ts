/**
 * Soft tool requirement manager — remind-then-escalate pattern.
 *
 * When the host wants a specific tool called before yielding or calling other
 * tools, but doesn't want to pay the message-cache invalidation cost of
 * forcing tool_choice up front, this manager handles:
 *   1. Injecting reminder messages once per id activation
 *   2. Tracking whether the model complied
 *   3. Escalating to forced tool choice after MAX_ESCALATIONS failures
 *
 * This mirrors the OMP approach from packages/agent/src/agent-loop.ts.
 */

import type {
	Message,
	SoftToolRequirement,
	SoftToolRequirementState,
} from "../../system/types/types-messages.ts";

/** Max consecutive escalations before aborting to avoid an unbounded force loop. */
export const MAX_ESCALATIONS = 3;

/** Sentinel error thrown when escalation limit is exceeded. */
export class SoftToolRequirementExceededError extends Error {
	constructor(toolName: string) {
		super(
			`Soft tool requirement '${toolName}' was not satisfied after ${MAX_ESCALATIONS} forced turns; aborting to avoid an unbounded force loop.`,
		);
		this.name = "SoftToolRequirementExceededError";
	}
}

export class SoftToolRequirementManager {
	private state: SoftToolRequirementState = {
		requirement: undefined,
		escalations: 0,
	};


	/**
	 * Set (or clear) the current soft requirement. Clears state when `req` is
	 * undefined. Returns true if this is a new/changed requirement (triggering
	 * reminder injection on the next check).
	 */
	setRequirement(req: SoftToolRequirement | undefined): boolean {
		const isNew =
			!this.state.requirement ||
			this.state.requirement.id !== req?.id ||
			this.state.requirement.toolName !== req?.toolName;
		if (isNew) {
			this.state.requirement = req;
			this.state.escalations = 0;
		}
		return isNew;
	}

	/** Get the current soft requirement state. */
	get stateValue(): SoftToolRequirementState {
		return this.state;
	}

	/**
	 * Check whether the model complied with the current soft requirement.
	 * Returns true if the requirement is satisfied (or not active).
	 * Updates escalation count on non-compliance.
	 */
	checkCompliance(toolCalls: Array<{ name: string; arguments?: Record<string, unknown> }>): boolean {
		const { requirement } = this.state;
		if (!requirement) return true;

		const satisfies = requirement.satisfies
			? (tc: { name: string; arguments?: Record<string, unknown> }) =>
					requirement.satisfies?.(tc) ?? tc.name === requirement.toolName
			: (tc: { name: string; arguments?: Record<string, unknown> }) =>
					tc.name === requirement.toolName;

		const allSatisfied =
			toolCalls.length === 0 || toolCalls.every(satisfies);

		if (allSatisfied) {
			// Reset escalations on compliance
			this.state.escalations = 0;
			return true;
		}

		// Non-compliance: increment escalations
		this.state.escalations++;

		if (this.state.escalations >= MAX_ESCALATIONS) {
			throw new SoftToolRequirementExceededError(requirement.toolName);
		}

		return false;
	}

	/**
	 * Get reminder messages to inject when a new requirement activates.
	 * Returns undefined if no requirement or already injected.
	 */
	getReminder(): Message[] | undefined {
		if (!this.state.requirement) return undefined;
		return this.state.requirement.reminder;
	}

	/**
	 * Get a forced tool choice for escalation (one-turn hard requirement).
	 * Returns undefined if no escalation needed.
	 */
	getEscalationToolChoice(): string | undefined {
		if (
			this.state.requirement &&
			this.state.escalations > 0 &&
			this.state.escalations < MAX_ESCALATIONS
		) {
			return this.state.requirement.toolName;
		}
		return undefined;
	}

	/** Reset all state (call at turn boundary). */
	reset(): void {
		this.state = {
			requirement: undefined,
			escalations: 0,
		};
	}
}
