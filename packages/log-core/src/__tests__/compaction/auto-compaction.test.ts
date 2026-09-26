// Regression tests for the auto-compaction trigger paths and snapcompact
// delivery. Each block names the failure it pins.

import { describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { compactToFit } from "../../compaction/engine.ts";
import { runCompaction } from "../../compaction/orchestration.ts";
import { AgentSession } from "../../harness/agent-session.ts";
import { buildBuiltinHooks } from "../../hooks/builtin/builtin-hooks.ts";
import {
	convertToLlm,
	estimateChatPayloadTokens,
} from "../../provider/messages.ts";
import { SessionStore } from "../../session/session-store.ts";
import type { AgentConfig } from "../../types/config.ts";
import type { AgentEvent, Message } from "../../types/messages.ts";
import { FakeBackend, textResponse } from "../fake-backend.ts";

const words = (n: number, seed: string) =>
	Array.from({ length: n }, (_, i) => `${seed}${i}`).join(" ");

/** Plain chat: nothing heavy for shake to drop. */
function chat(turns: number, size: number): Message[] {
	const out: Message[] = [];
	for (let i = 0; i < turns; i++) {
		out.push({ role: "user", content: `Question ${i}: ${words(size, "q")}` });
		out.push({
			role: "assistant",
			content: `Answer ${i}: ${words(size, "a")}`,
		});
	}
	return out;
}

/** One user message, then many OpenAI-shaped tool rounds (the loop's shape). */
function agenticRun(rounds: number): Message[] {
	const out: Message[] = [
		{ role: "user", content: "Refactor the parser in src/parse.ts" },
	];
	for (let i = 0; i < rounds; i++) {
		out.push({
			role: "assistant",
			content: `Step ${i}`,
			tool_calls: [
				{
					id: `c${i}`,
					name: "read",
					arguments: JSON.stringify({ path: `src/f${i}.ts` }),
				},
			],
		} as Message);
		out.push({
			role: "tool",
			tool_call_id: `c${i}`,
			content: words(300, `t${i}_`),
		} as Message);
	}
	return out;
}

function session(
	config: Partial<AgentConfig>,
	backend = new FakeBackend([() => textResponse("ok")]),
): { session: AgentSession; events: AgentEvent[] } {
	const events: AgentEvent[] = [];
	const s = new AgentSession({
		config: {
			baseUrl: "http://fake",
			model: "fake",
			continuationEnabled: false,
			...config,
		},
		backend,
	});
	s.observe({ event: e => events.push(e) });
	return { session: s, events };
}

const compactions = (events: AgentEvent[]) =>
	events.filter(e => e.type === "compaction");
const summaryOf = (messages: Message[]) =>
	messages.find(m => String(m.role) === "compactionSummary") as
		| (Message & { snapcompact?: Record<string, unknown> })
		| undefined;

describe("runCompaction", () => {
	test("compacts plain chat in llm/snapcompact modes (shake is a pre-pass, not a gate)", async () => {
		const history = chat(300, 20);
		const before = await estimateChatPayloadTokens(history as never);
		for (const mode of ["snapcompact", "llm"] as const) {
			const backend = new FakeBackend([() => textResponse("SUMMARY")]);
			const result = await runCompaction(backend, history, before, {
				reason: "auto",
				mode,
				keepRecentTokens: 2_000,
			});
			expect(result.changed).toBe(true);
			expect(result.messages.length).toBeLessThan(history.length);
			if (mode === "llm") expect(backend.calls).toBe(1);
		}
	});

	test("keeps shake's savings when the summarizing pass finds nothing to cut", async () => {
		const history = chat(40, 400);
		const before = await estimateChatPayloadTokens(history as never);
		const result = await runCompaction(new FakeBackend([]), history, before, {
			reason: "auto",
			mode: "snapcompact",
		});
		expect(result.changed).toBe(true);
		expect(result.tokensAfter).toBeLessThan(before);
	});
});

describe("session auto-compaction", () => {
	const settings = {
		enabled: true,
		reserveTokens: 4_000,
		keepRecentTokens: 2_000,
	};

	test("triggers on the config's contextWindowTokens and commits the result", async () => {
		const { session: s, events } = session({ contextWindowTokens: 16_000 });
		s.setAutoCompactionSettings({ ...settings, mode: "snapcompact" });
		s.setHistory(chat(30, 150));
		await s.prompt("next");
		const done = compactions(events).at(-1) as
			| { tokensBefore?: number; tokensAfter?: number }
			| undefined;
		expect(done?.tokensAfter).toBeLessThan(done?.tokensBefore ?? 0);
		expect(summaryOf(s.messages)).toBeDefined();
	});

	test("uses the active model's per-model window over the global one", async () => {
		const { session: s } = session({
			contextWindowTokens: 1_000_000,
			models: [
				{ name: "small", model: "fake", url: "x", contextWindow: 16_000 },
			],
		});
		s.setAutoCompactionSettings(settings);
		s.setHistory(chat(30, 150));
		await s.prompt("next");
		expect(summaryOf(s.messages)).toBeDefined();
	});

	test("does nothing — no events, no phase change — below the threshold", async () => {
		const { session: s, events } = session({ contextWindowTokens: 128_000 });
		s.setAutoCompactionSettings(settings);
		s.setHistory(chat(3, 20));
		await s.prompt("next");
		expect(compactions(events)).toEqual([]);
	});

	test("a configured tail larger than the window is capped so there is something to cut", async () => {
		const { session: s } = session({ contextWindowTokens: 16_000 });
		s.setAutoCompactionSettings({ ...settings, keepRecentTokens: 20_000 });
		s.setHistory(chat(30, 150));
		await s.prompt("next");
		expect(summaryOf(s.messages)).toBeDefined();
	});

	test("honors the configured mode (llm calls the model; snapcompact archives)", async () => {
		const llmBackend = new FakeBackend([
			() => textResponse("LLM SUMMARY"),
			() => textResponse("ok"),
		]);
		const llm = session({ contextWindowTokens: 16_000 }, llmBackend);
		llm.session.setAutoCompactionSettings({ ...settings, mode: "llm" });
		llm.session.setHistory(chat(30, 150));
		await llm.session.prompt("next");
		expect(summaryOf(llm.session.messages)?.content).toContain("LLM SUMMARY");

		const snap = session({ contextWindowTokens: 16_000 });
		snap.session.setAutoCompactionSettings({
			...settings,
			mode: "snapcompact",
		});
		snap.session.setHistory(chat(30, 150));
		await snap.session.prompt("next");
		expect(summaryOf(snap.session.messages)?.snapcompact).toBeDefined();
	});
});

describe("in-loop threshold compaction", () => {
	const run = agenticRun(20);
	const hooks = (mode?: "snapcompact") =>
		buildBuiltinHooks({
			config: { baseUrl: "x", model: "fake" } as AgentConfig,
			contextWindowTokens: () => 16_000,
			toolDefs: () => [],
			compactionSettings: () => (mode ? { mode } : {}),
		});

	test("compacts a single long agentic turn at a valid split-turn cut", async () => {
		const result = await hooks().prepareNextTurn?.({
			messages: run,
			iteration: 10,
			hadToolCalls: true,
		} as never);
		const out = (result?.messages ?? []) as Array<
			Message & { tool_calls?: Array<{ id: string }>; tool_call_id?: string }
		>;
		expect(out.length).toBeLessThan(run.length);
		expect(String(out[0]?.role)).toBe("compactionSummary");
		const callIds = new Set(
			out.flatMap(m => m.tool_calls?.map(c => c.id) ?? []),
		);
		expect(
			out.filter(m => m.tool_call_id && !callIds.has(m.tool_call_id)),
		).toEqual([]);
	});

	test("triggers on the exact token count, not the engine's heuristic", async () => {
		// Well under 80% of the window by BPE count → no compaction.
		const small = agenticRun(3);
		expect(
			await hooks().prepareNextTurn?.({
				messages: small,
				iteration: 10,
				hadToolCalls: true,
			} as never),
		).toBeUndefined();
	});

	test("uses snapcompact when configured, archiving tool calls and results", async () => {
		const result = await hooks("snapcompact").prepareNextTurn?.({
			messages: run,
			iteration: 10,
			hadToolCalls: true,
		} as never);
		const summary = summaryOf((result?.messages ?? []) as Message[]);
		const archive = summary?.snapcompact?.snapcompact as
			| { totalChars: number }
			| undefined;
		expect(archive?.totalChars).toBeGreaterThan(20_000);
		expect(String(summary?.content)).toContain("Files: src/f0.ts");
	});
});

describe("snapcompact delivery", () => {
	test("the model receives the archived head/tail text as well as the frames", async () => {
		const { messages } = await compactToFit(chat(30, 150) as never, {
			triggerTokens: 0,
			settings: { mode: "snapcompact", keepRecentTokens: 2_000 },
		});
		const [llm] = convertToLlm(messages as never) as Array<
			Message & { images?: unknown[] }
		>;
		expect(String(llm?.content)).toContain("<archived-history>");
		expect(String(llm?.content)).toContain("Question 0:");
		expect(llm?.images?.length ?? 0).toBeGreaterThan(0);
	});

	test("the archive survives a session reload", async () => {
		const dir = mkdtempSync(join(tmpdir(), "snapcompact-resume-"));
		const store = new SessionStore("resume", { baseDir: dir });
		const { session: s } = session({ contextWindowTokens: 16_000 });
		s.attachSession(store);
		s.setAutoCompactionSettings({
			enabled: true,
			reserveTokens: 4_000,
			keepRecentTokens: 2_000,
			mode: "snapcompact",
		});
		s.setHistory(chat(30, 150));
		await s.prompt("next");

		const context = new SessionStore("resume", { baseDir: dir }).buildContext();
		const [llm] = convertToLlm(context.messages as never) as Array<
			Message & { images?: unknown[] }
		>;
		expect(String(llm?.content)).toContain("<archived-history>");
		expect(llm?.images?.length ?? 0).toBeGreaterThan(0);
	});
});
