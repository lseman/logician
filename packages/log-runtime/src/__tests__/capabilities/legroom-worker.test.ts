import { afterEach, test } from "bun:test";
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { LegroomWorker } from "../../capabilities/legroom/worker.ts";

const workers: LegroomWorker[] = [];
const repositoryRoot = resolve(import.meta.dirname, "../../../../../");
const legroomPython = resolve(repositoryRoot, "ecosystem/.venv/bin/python");
afterEach(() => {
	for (const worker of workers.splice(0)) worker.close();
});

void test("LegroomWorker transforms messages through the persistent SDK", async () => {
	const worker = new LegroomWorker({
		python: legroomPython,
		args: ["-m", "legroom.sdk_worker"],
		config: { optimize: false },
		failOpen: false,
	});
	workers.push(worker);
	const messages = [{ role: "user", content: "hello" }];

	assert.deepEqual(await worker.compress(messages, "gpt-4o"), messages);
	assert.deepEqual(await worker.compress([], "gpt-4o"), []);
});

void test("LegroomWorker fails open when the worker cannot start", async () => {
	const worker = new LegroomWorker({
		python: "/definitely/missing/python",
		timeoutMs: 100,
		failOpen: true,
	});
	workers.push(worker);
	const messages = [{ role: "user", content: "unchanged" }];

	assert.strictEqual(await worker.compress(messages, "gpt-4o"), messages);
});

void test("LegroomWorker surfaces SDK errors when fail-open is disabled", async () => {
	const worker = new LegroomWorker({
		python: legroomPython,
		config: { unknown_option: true },
		failOpen: false,
	});
	workers.push(worker);

	await assert.rejects(
		worker.compress([], "gpt-4o"),
		/unknown compression config field/,
	);
});
