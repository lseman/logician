import { describe, expect, test } from "bun:test";
import { LegroomGateway } from "../../../capabilities/legroom/legroom-gateway.ts";
import { MemoriamGateway } from "../../../capabilities/memoriam/memoriam-gateway.ts";

function stubWorker(gateway: unknown, stub: Record<string, unknown>): void {
	Object.defineProperty(gateway, "worker", { value: stub, configurable: true });
}

const msg = (role: string, content: string) => ({ role, content });

describe("MemoriamGateway prompt-cache behavior", () => {
	function make() {
		const gateway = new MemoriamGateway({ mode: "sdk" });
		const calls: Array<{ sessionId: string; query: string; budget: number }> =
			[];
		let text = "mem-v1";
		let fail = false;
		stubWorker(gateway, {
			getContext: async (sessionId: string, query: string, budget: number) => {
				if (fail) throw new Error("worker down");
				calls.push({ sessionId, query, budget });
				return text;
			},
			createMemory: async () => ({ id: "m1", content: "" }),
			close: () => {},
		});
		return {
			gateway,
			calls,
			setText: (t: string) => (text = t),
			setFail: (f: boolean) => (fail = f),
		};
	}

	const hookCtx = (messages: unknown[]) => ({
		payload: { messages, maxTokens: 1000 },
		model: "m",
		hookSessionId: "s1",
	});

	test("appends memory context as a trailing system message", async () => {
		const { gateway } = make();
		const hooks = gateway.createHooks(undefined);
		const result = await hooks.beforeProviderPayload?.(
			hookCtx([msg("system", "sys"), msg("user", "hello")]),
		);
		const messages = (result?.payload.messages ?? []) as {
			role: string;
			content: string;
		}[];
		// Leading system prompt and history are untouched — the cacheable prefix.
		expect(messages[0]).toEqual(msg("system", "sys"));
		expect(messages[1]).toEqual(msg("user", "hello"));
		expect(messages[messages.length - 1]).toEqual({
			role: "system",
			content: "# Memoriam Memory Context\nmem-v1",
		});
	});

	test("serves cached context while the memory revision is unchanged", async () => {
		const { gateway, calls } = make();
		const hooks = gateway.createHooks(undefined);
		for (let i = 0; i < 3; i++) {
			await hooks.beforeProviderPayload?.(hookCtx([msg("user", `turn ${i}`)]));
		}
		expect(calls.length).toBe(1);
	});

	test("re-retrieves after a memory mutation bumps the revision", async () => {
		const { gateway, calls, setText } = make();
		const hooks = gateway.createHooks(undefined);
		await hooks.beforeProviderPayload?.(hookCtx([msg("user", "a")]));
		await gateway.createMemory("new fact");
		setText("mem-v2");
		await hooks.beforeProviderPayload?.(hookCtx([msg("user", "b")]));
		expect(calls.length).toBe(2);
		const result = await hooks.beforeProviderPayload?.(
			hookCtx([msg("user", "c")]),
		);
		expect(calls.length).toBe(2);
		const messages = (result?.payload.messages ?? []) as { content: string }[];
		expect(messages[messages.length - 1].content).toContain("mem-v2");
	});

	test("leaves the payload unchanged and retries after retrieval failure", async () => {
		const { gateway, calls, setFail } = make();
		const hooks = gateway.createHooks(undefined);
		setFail(true);
		const ctx = hookCtx([msg("user", "hello")]);
		const result = await hooks.beforeProviderPayload?.(ctx);
		expect(result?.payload).toBe(ctx.payload);
		expect(calls.length).toBe(0);
		setFail(false);
		await hooks.beforeProviderPayload?.(hookCtx([msg("user", "hello")]));
		expect(calls.length).toBe(1);
	});
});

describe("LegroomGateway prompt-cache behavior", () => {
	function make(config: Record<string, unknown> = {}) {
		const gateway = new LegroomGateway({ mode: "sdk", config });
		const callSizes: number[] = [];
		stubWorker(gateway, {
			compress: async (messages: Record<string, unknown>[]) => {
				callSizes.push(messages.length);
				return messages.map(m => ({ ...m, legroom: "done" }));
			},
			close: () => {},
		});
		return { gateway, callSizes };
	}

	const hookCtx = (messages: unknown[]) => ({
		payload: { messages },
		model: "m",
	});

	test("identical payload is served from the memo without a worker call", async () => {
		const { gateway, callSizes } = make();
		const hooks = gateway.createHooks(undefined);
		const build = () => [
			msg("system", "sys"),
			msg("user", "q1"),
			msg("assistant", "a1"),
			msg("user", "q2"),
		];
		await hooks.beforeProviderPayload?.(hookCtx(build()));
		await hooks.beforeProviderPayload?.(hookCtx(build()));
		// Cold pass only; the retry-shaped second call never reaches the worker.
		expect(callSizes).toEqual([4]);
	});

	test("append-only growth sends only the unstable tail and splices", async () => {
		const { gateway, callSizes } = make();
		const hooks = gateway.createHooks(undefined);
		const first = [
			msg("system", "sys"),
			msg("user", "q1"),
			msg("assistant", "a1"),
			msg("user", "q2"),
			msg("assistant", "a2"),
		];
		const r1 = await hooks.beforeProviderPayload?.(hookCtx(first));
		const firstResult = r1?.payload.messages as Record<string, unknown>[];
		// default protect_recent = 3 → stable prefix = first 2 messages;
		// tail = last 3 (previously protected) + 2 new = 5 messages.
		const second = [...first, msg("user", "q3"), msg("assistant", "a3")];
		const r2 = await hooks.beforeProviderPayload?.(hookCtx(second));
		const secondResult = r2?.payload.messages as Record<string, unknown>[];
		expect(callSizes).toEqual([5, 5]);
		expect(secondResult.length).toBe(7);
		// Splice reuses the cached compressed prefix by reference.
		expect(secondResult[0]).toBe(firstResult[0]);
		expect(secondResult[1]).toBe(firstResult[1]);
		expect(secondResult[2]).not.toBe(firstResult[2]);
	});

	test("rewritten history falls back to a full pass", async () => {
		const { gateway, callSizes } = make();
		const hooks = gateway.createHooks(undefined);
		const base = [
			msg("system", "sys"),
			msg("user", "q1"),
			msg("assistant", "a1"),
			msg("user", "q2"),
		];
		await hooks.beforeProviderPayload?.(hookCtx(base));
		const rewritten = [msg("system", "sys-CHANGED"), ...base.slice(1)];
		await hooks.beforeProviderPayload?.(hookCtx(rewritten));
		expect(callSizes).toEqual([4, 4]);
	});

	test("shrunken history falls back to a full pass", async () => {
		const { gateway, callSizes } = make();
		const hooks = gateway.createHooks(undefined);
		const base = [
			msg("system", "sys"),
			msg("user", "q1"),
			msg("assistant", "a1"),
			msg("user", "q2"),
		];
		await hooks.beforeProviderPayload?.(hookCtx(base));
		await hooks.beforeProviderPayload?.(hookCtx(base.slice(0, 2)));
		expect(callSizes).toEqual([4, 2]);
	});

	test("cross-message compression config disables splicing", async () => {
		const { gateway, callSizes } = make({ semantic_dedup_enabled: true });
		const hooks = gateway.createHooks(undefined);
		const base = [
			msg("system", "sys"),
			msg("user", "q1"),
			msg("assistant", "a1"),
			msg("user", "q2"),
		];
		await hooks.beforeProviderPayload?.(hookCtx(base));
		const grown = [...base, msg("user", "q3")];
		await hooks.beforeProviderPayload?.(hookCtx(grown));
		expect(callSizes).toEqual([4, 5]);
	});
});
