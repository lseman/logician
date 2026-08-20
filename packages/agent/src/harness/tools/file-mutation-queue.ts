// ── File mutation queue ───────────────────────────────────────────────────
// Serialize mutations targeting the same file (keyed by canonical path),
// while mutations to different files still run in parallel. Ported from
// coding-agent's tools/shared/file-mutation-queue.ts, rebuilt on
// ExecutionEnv.canonicalPath/absolutePath instead of raw node:fs.

import type { ExecutionEnv } from "../../env/execution-env.ts";

const fileMutationQueues = new Map<string, Promise<void>>();
let registrationQueue = Promise.resolve();

async function getMutationQueueKey(
	env: ExecutionEnv,
	filePath: string,
): Promise<string> {
	const resolvedResult = await env.absolutePath(filePath);
	const resolvedPath = resolvedResult.ok ? resolvedResult.value : filePath;
	const canonical = await env.canonicalPath(resolvedPath);
	return canonical.ok ? canonical.value : resolvedPath;
}

export async function withFileMutationQueue<T>(
	env: ExecutionEnv,
	filePath: string,
	fn: () => Promise<T>,
): Promise<T> {
	const registration = registrationQueue.then(async () => {
		const key = await getMutationQueueKey(env, filePath);
		const currentQueue = fileMutationQueues.get(key) ?? Promise.resolve();

		let releaseNext!: () => void;
		const nextQueue = new Promise<void>(resolveQueue => {
			releaseNext = resolveQueue;
		});
		const chainedQueue = currentQueue.then(() => nextQueue);
		fileMutationQueues.set(key, chainedQueue);

		return { key, currentQueue, chainedQueue, releaseNext };
	});
	registrationQueue = registration.then(
		() => undefined,
		() => undefined,
	);

	const { key, currentQueue, chainedQueue, releaseNext } = await registration;
	await currentQueue;
	try {
		return await fn();
	} finally {
		releaseNext();
		if (fileMutationQueues.get(key) === chainedQueue) {
			fileMutationQueues.delete(key);
		}
	}
}
