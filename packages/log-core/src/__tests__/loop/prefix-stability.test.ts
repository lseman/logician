import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	type PrefixDivergence,
	PrefixStabilityTracker,
} from "../../loop/prefix-stability.ts";
import {
	convertToChatFormat,
	convertToLlm,
	createAssistantMessage,
	createSystemMessage,
	createToolResultMessage,
	createUserMessage,
} from "../../provider/messages.ts";
import { ToolRegistry } from "../../tools/registry.ts";
import type { Message } from "../../types/messages.ts";

// ── Tracker semantics ─────────────────────────────────────────────────────

const msg = (content: string): Record<string, unknown> => ({
	role: "user",
	content,
});
const toolDefs = (names: string[]) =>
	names.map(name => ({ type: "function", function: { name } }));

void test("first record is trivially stable and starts the baseline", () => {
	const tracker = new PrefixStabilityTracker();
	const report = tracker.record([msg("a")], toolDefs(["read"]));
	assert.deepEqual(report, {
		stable: true,
		divergedAt: 0,
		previousLength: 0,
		rewritten: false,
	});
});

void test("append-only growth keeps the prefix stable", () => {
	const tracker = new PrefixStabilityTracker();
	tracker.record([msg("a"), msg("b")], toolDefs(["read"]));
	const report = tracker.record(
		[msg("a"), msg("b"), msg("c")],
		toolDefs(["read"]),
	);
	assert.equal(report.stable, true);
	assert.equal(report.divergedAt, 2, "pure append diverges at the old length");
	assert.equal(report.previousLength, 2);
	assert.equal(report.rewritten, false);
});

void test("in-place rewrite diverges at the rewritten index", () => {
	const tracker = new PrefixStabilityTracker();
	tracker.record([msg("a"), msg("b"), msg("c")], toolDefs(["read"]));
	const report = tracker.record(
		[msg("a"), msg("B"), msg("c"), msg("d")],
		toolDefs(["read"]),
	);
	assert.equal(report.stable, false);
	assert.equal(report.divergedAt, 1);
	assert.equal(report.previousLength, 3);
	assert.equal(report.rewritten, false);
});

void test("payload shrink (compaction) reports a rewrite, not a silent divergence", () => {
	const tracker = new PrefixStabilityTracker();
	tracker.record([msg("a"), msg("b"), msg("c")], toolDefs(["read"]));
	const report = tracker.record([msg("summary")], toolDefs(["read"]));
	assert.equal(report.stable, false);
	assert.equal(report.rewritten, true);
	assert.equal(report.previousLength, 3);
});

void test("tool-spec change invalidates the whole prefix", () => {
	const tracker = new PrefixStabilityTracker();
	tracker.record([msg("a")], toolDefs(["read"]));
	const report = tracker.record(
		[msg("a"), msg("b")],
		toolDefs(["read", "bash"]),
	);
	assert.equal(report.stable, false);
	assert.equal(report.divergedAt, 0);
	assert.equal(report.rewritten, true);
});

void test("reset drops the baseline", () => {
	const tracker = new PrefixStabilityTracker();
	tracker.record([msg("a")], toolDefs(["read"]));
	tracker.reset();
	const report: PrefixDivergence = tracker.record(
		[msg("z")],
		toolDefs(["read"]),
	);
	assert.equal(report.previousLength, 0);
	assert.equal(report.stable, true);
});

// ── Pipeline contract: the bytes the provider actually receives ───────────
// Pins the append-only prompt-prefix property every prompt-cache/KV-prefix
// hit depends on. These tests must fail the moment any part of the
// messages → chat payload path stops being a pure function of message
// content, or the transcript stops being append-only in its stable prefix.

function makeRegistry(): ToolRegistry {
	const registry = new ToolRegistry();
	const tool = (name: string) => ({
		name,
		description: `The ${name} tool.`,
		parameters: {
			type: "object",
			properties: { path: { type: "string" } },
			required: ["path"],
		},
		execute: async (): Promise<string> => "ok",
	});
	registry.register(tool("read"));
	registry.register(tool("bash"));
	registry.register(tool("write"));
	return registry;
}

const serialize = (value: unknown): string => JSON.stringify(value);

function prefixMatches(
	previous: Record<string, unknown>[],
	current: Record<string, unknown>[],
): number {
	// Returns the first index where bytes differ, or previous.length if the
	// whole previous payload is a byte-identical prefix of the current one.
	const bound = Math.min(previous.length, current.length);
	for (let i = 0; i < bound; i++) {
		if (serialize(previous[i]) !== serialize(current[i])) return i;
	}
	return previous.length;
}

void test("append-only transcript produces a byte-identical payload prefix", () => {
	const registry = makeRegistry();
	const base: Message[] = [
		createSystemMessage("You are logician."),
		createUserMessage("read the file"),
		createAssistantMessage("", [
			{ id: "c1", name: "read", arguments: '{"path":"a.txt"}' },
		]),
		createToolResultMessage("c1", "read", "line A\nline B"),
	];

	const payload1 = convertToChatFormat(convertToLlm(base));
	const defs1 = registry.toToolDefinitions();

	// Turn 2: append-only growth, fresh toToolDefinitions() call.
	const payload2 = convertToChatFormat(
		convertToLlm([...base, createUserMessage("and the next file")]),
	);
	const defs2 = registry.toToolDefinitions();

	assert.equal(
		prefixMatches(payload1, payload2),
		payload1.length,
		"every previous message must serialize byte-identically",
	);
	assert.equal(
		serialize(defs1),
		serialize(defs2),
		"tool definitions must be byte-stable while the tool set is unchanged",
	);
});

void test("memoriam-style trailing memory append keeps the prefix stable", () => {
	const base: Message[] = [
		createSystemMessage("You are logician."),
		createUserMessage("work"),
	];
	const payload1 = convertToChatFormat(convertToLlm(base));
	// The memory block rides on the payload as a trailing system message
	// (see MemoriamGateway) — it must never reach into the prefix.
	const payload2 = [
		...payload1,
		{ role: "system", content: "MEMORY: user prefers tabs" },
	];
	assert.equal(prefixMatches(payload1, payload2), payload1.length);
});

void test("compaction rewrites only the tail; the system message survives", () => {
	const base: Message[] = [
		createSystemMessage("You are logician."),
		createUserMessage("work"),
		createAssistantMessage("understood"),
	];
	const before = convertToChatFormat(convertToLlm(base));
	const system = base[0] as Message;
	const after = convertToChatFormat(
		convertToLlm([
			system,
			{
				role: "compactionSummary",
				content: "Summary of prior work.",
				tokensBefore: 5000,
				timestamp: Date.now(),
			},
		]),
	);
	assert.equal(
		serialize(before[0]),
		serialize(after[0]),
		"the system message must survive compaction byte-identically",
	);
	assert.notEqual(serialize(before[1]), serialize(after[1]));
});

void test("the tracker sees the real pipeline as stable across turns", () => {
	const registry = makeRegistry();
	const base: Message[] = [
		createSystemMessage("You are logician."),
		createUserMessage("read the file"),
		createToolResultMessage("c1", "read", "line A"),
	];
	const tracker = new PrefixStabilityTracker();
	const defs = registry.toToolDefinitions();
	tracker.record(convertToChatFormat(convertToLlm(base)), defs);

	const grown = convertToChatFormat(
		convertToLlm([...base, createUserMessage("next")]),
	);
	const report = tracker.record(grown, registry.toToolDefinitions());
	assert.equal(report.stable, true);
	assert.equal(report.divergedAt, base.length);

	// A mid-history rewrite is the failure mode the tracker exists to catch.
	const rewritten = grown.slice();
	rewritten[1] = { role: "user", content: "read the OTHER file" };
	const diverged = tracker.record(rewritten, registry.toToolDefinitions());
	assert.equal(diverged.stable, false);
	assert.equal(diverged.divergedAt, 1);
});
