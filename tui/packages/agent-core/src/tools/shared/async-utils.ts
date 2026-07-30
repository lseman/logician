// ── Async utilities ─────────────────────────────────────────────────────────
// Small shared async helpers used across the loop and harness.

/**
 * Sleep for `ms` milliseconds. Rejects early if `signal` is aborted.
 * Ported from pi packages/coding-agent/src/utils/sleep.ts.
 */
export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
	return new Promise<void>((resolve, reject) => {
		if (signal?.aborted) {
			reject(new DOMException("Aborted", "AbortError"));
			return;
		}
		const timer = setTimeout(resolve, ms);
		signal?.addEventListener(
			"abort",
			() => {
				clearTimeout(timer);
				reject(new DOMException("Aborted", "AbortError"));
			},
			{ once: true },
		);
	});
}

import { AgentError, AgentErrorType } from "../../agent/types.ts";

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
