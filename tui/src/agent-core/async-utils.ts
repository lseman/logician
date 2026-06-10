// ── Async utilities ─────────────────────────────────────────────────────────
// Small shared async helpers used across the loop and harness.

import { AgentError, AgentErrorType } from "./types.ts";

/**
 * Run `promise` with a timeout. Rejects with `AgentError(TURN_TIMEOUT)` if the
 * timeout fires before `promise` settles. The timer is always cleared.
 *
 * `onTimeout` runs in the timeout branch *before* the rejection, so the caller
 * can cancel the losing promise (e.g. abort the in-flight provider request).
 * Without it the underlying operation keeps running after the turn is
 * abandoned, streaming into a dead turn and holding the connection open.
 */
export function withTimeout<T>(
	promise: Promise<T>,
	timeoutMs: number,
	onTimeout?: () => void,
): Promise<T> {
	return new Promise<T>((resolve, reject) => {
		const timer = setTimeout(() => {
			onTimeout?.();
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
