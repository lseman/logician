import { test } from "bun:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import {
	applyEohCandidate,
	evaluateEohCandidate,
	loadEohFile,
	renderEohCandidate,
} from "../application/eoh/file.ts";

const SOURCE = `"""Maximize the score over a fixed integer dataset."""

# EOH-BEGIN
def heuristic(value: int) -> int:
    return value
# EOH-END

def evaluate(heuristic) -> float:
    return sum(heuristic(value) for value in [1, 2, 3])
`;

void test("self-evaluating EoH files preserve their evaluator", async () => {
	const directory = await mkdtemp(path.join(os.tmpdir(), "logician-eoh-test-"));
	try {
		const file = path.join(directory, "heuristic.py");
		await writeFile(file, SOURCE, "utf8");
		const target = await loadEohFile("heuristic.py", directory);

		assert.equal(target.functionSignature, "def heuristic(value: int):");
		assert.equal(
			await evaluateEohCandidate(target, target.heuristicCode, 5_000),
			6,
		);

		const improved = `def heuristic(value: int) -> int:
    return value * value`;
		assert.equal(await evaluateEohCandidate(target, improved, 5_000), 14);

		const rendered = renderEohCandidate(target, improved);
		assert.match(rendered, /def evaluate\(heuristic\)/);
		assert.match(rendered, /return value \* value/);

		applyEohCandidate(target, improved);
		const applied = await readFile(file, "utf8");
		assert.match(applied, /# EOH-BEGIN/);
		assert.match(applied, /# EOH-END/);
		assert.match(applied, /def evaluate\(heuristic\)/);
		assert.match(applied, /return value \* value/);
	} finally {
		await rm(directory, { recursive: true, force: true });
	}
});

void test("EoH rejects files without an isolated heuristic region", async () => {
	const directory = await mkdtemp(path.join(os.tmpdir(), "logician-eoh-test-"));
	try {
		await writeFile(
			path.join(directory, "heuristic.py"),
			"def heuristic(x): return x\ndef evaluate(fn): return 1\n",
			"utf8",
		);
		await assert.rejects(
			loadEohFile("heuristic.py", directory),
			/EOH-BEGIN.*EOH-END/,
		);
	} finally {
		await rm(directory, { recursive: true, force: true });
	}
});
