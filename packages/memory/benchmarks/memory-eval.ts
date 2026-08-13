import { unlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createMemoryStore } from "../src/store/index.js";
import type { ClaimStatus, ObservationTrust } from "../src/types.js";
import corpus from "./memory-eval-corpus.json";

type Case = (typeof corpus)[number];

function percentile(values: number[], fraction: number): number {
	if (!values.length) return 0;
	const sorted = [...values].sort((a, b) => a - b);
	return sorted[
		Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))
	];
}

function runCase(testCase: Case) {
	const path = join(
		tmpdir(),
		`logician-memory-eval-${testCase.id}-${crypto.randomUUID()}.db`,
	);
	const store = createMemoryStore(path);
	store.setCurrentWorkspace("/eval");
	store.createSession("query", { cwd: "/eval" });
	for (const claim of testCase.claims) {
		const sessionId = `session-${claim.id}`;
		const observationId = `eval-${testCase.id}-${claim.id}`;
		store.createSession(sessionId, { cwd: "/eval" });
		store.observe(
			{
				id: observationId,
				sessionId,
				timestamp: claim.timestamp,
				hookType: "stop",
				workspace: "/eval",
				raw: { benchmark: testCase.id },
			},
			{
				id: observationId,
				sessionId,
				timestamp: claim.timestamp,
				type: "decision",
				title: testCase.id,
				facts: [],
				narrative: claim.text,
				concepts: testCase.query.split(/\s+/).slice(0, 4),
				files: [],
				importance: 9,
				consolidated: false,
				claims: [
					{
						text: claim.text,
						status: claim.status as ClaimStatus,
						confidence: claim.confidence,
						evidenceEventIds: [`eval:${testCase.id}:${claim.id}`],
					},
				],
				provenance: {
					source: "deterministic",
					trust: claim.trust as ObservationTrust,
					extractorVersion: "memory-eval/1",
					schemaVersion: 1,
				},
			},
		);
	}
	const context = store.getContext("query", 1200, testCase.query);
	const trace = store.listRetrievalTraces(1)[0];
	const selected = trace?.selected.map(item => item.id) || [];
	const hits = testCase.expected.filter(id => selected.includes(id)).length;
	const recall = testCase.expected.length ? hits / testCase.expected.length : 1;
	const firstRelevant = selected.findIndex(id =>
		testCase.expected.includes(id),
	);
	const ndcg = testCase.expected.length
		? firstRelevant >= 0
			? 1 / Math.log2(firstRelevant + 2)
			: 0
		: 1;
	const obsoleteRejected = testCase.forbidden.every(
		text => !context.includes(text),
	);
	const abstentionCorrect = Boolean(trace?.abstained) === testCase.abstain;
	const environmentPassed =
		recall === 1 && obsoleteRejected && abstentionCorrect;
	if (trace) {
		store.recordOutcomeReceipt({
			retrievalTraceId: trace.id,
			taskId: `memory-eval/${testCase.id}`,
			trialId: "deterministic-seed-1",
			outcome: { environmentPassed },
		});
	}
	// A learned recommendation may evolve, but shadow mode must not alter the
	// deterministic production selection until a repeated external gate wins.
	store.getContext("query", 1200, testCase.query);
	const selectedAfterLearning =
		store.listRetrievalTraces(1)[0]?.selected.map(item => item.id) || [];
	const shadowNonInterference =
		JSON.stringify(selectedAfterLearning) === JSON.stringify(selected);
	store.close();
	for (const suffix of ["", "-wal", "-shm"])
		try {
			unlinkSync(path + suffix);
		} catch {}
	return {
		id: testCase.id,
		recallAt5: recall,
		ndcgAt5: ndcg,
		obsoleteRejected,
		abstentionCorrect,
		shadowNonInterference,
		latencyMs: trace?.latencyMs || 0,
		passed: environmentPassed && shadowNonInterference,
	};
}

const results = corpus.map(runCase);
const latencies = results.map(result => result.latencyMs);
const summary = {
	cases: results.length,
	passed: results.filter(result => result.passed).length,
	recallAt5:
		results.reduce((sum, result) => sum + result.recallAt5, 0) / results.length,
	ndcgAt5:
		results.reduce((sum, result) => sum + result.ndcgAt5, 0) / results.length,
	obsoleteFactRejection:
		results.filter(result => result.obsoleteRejected).length / results.length,
	abstentionAccuracy:
		results.filter(result => result.abstentionCorrect).length / results.length,
	shadowNonInterference:
		results.filter(result => result.shadowNonInterference).length /
		results.length,
	latencyP50Ms: percentile(latencies, 0.5),
	latencyP95Ms: percentile(latencies, 0.95),
};

console.log(JSON.stringify({ summary, results }, null, 2));
if (summary.passed !== summary.cases) process.exitCode = 1;
