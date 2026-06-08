// ── Async utilities ─────────────────────────────────────────────────────────
// Small shared async helpers used across the loop and harness.

import { AgentError, AgentErrorType } from "./types.ts";

/**
 * Run `promise` with a timeout. Rejects with `AgentError(TURN_TIMEOUT)` if the
 * timeout fires before `promise` settles. The timer is always cleared.
 */
export function withTimeout<T>(
	promise: Promise<T>,
	timeoutMs: number,
): Promise<T> {
	return new Promise<T>((resolve, reject) => {
		const timer = setTimeout(() => {
			reject(
				new AgentError({
					type: AgentErrorType.TURN_TIMEOUT,
					message: `Operation timed out after ${timeoutMs}ms`,
				}),
			);
		}, timeoutMs);
		promise.then(
			(value) => {
				clearTimeout(timer);
				resolve(value);
			},
			(reason) => {
				clearTimeout(timer);
				reject(reason);
			},
		);
	});
}
