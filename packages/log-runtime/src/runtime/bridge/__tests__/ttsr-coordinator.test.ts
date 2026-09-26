// ── TTSR Coordinator Integration Tests ────────────────────────────────────────

import { describe, expect, test } from "bun:test";
import {
	type TtsrJudge,
	TtsrManager,
	type TtsrRule,
	type TtsrSettings,
} from "@logician/log-core";
import type { RuntimeEvent } from "@logician/log-core/events";
import { TtsrCoordinator } from "../ttsr-coordinator.ts";

function harness(
	settings: Partial<TtsrSettings> = {},
	judge?: TtsrJudge,
): {
	coordinator: TtsrCoordinator;
	interrupts: string[];
	followUps: string[];
	events: RuntimeEvent[];
	judgeErrors: unknown[];
} {
	const interrupts: string[] = [];
	const followUps: string[] = [];
	const events: RuntimeEvent[] = [];
	const judgeErrors: unknown[] = [];
	const coordinator = new TtsrCoordinator({
		manager: new TtsrManager({ builtinRules: false, ...settings }),
		interrupt: text => interrupts.push(text),
		followUp: text => followUps.push(text),
		emit: event => events.push(event),
		judge,
		onJudgeError: error => judgeErrors.push(error),
	});
	return { coordinator, interrupts, followUps, events, judgeErrors };
}

function rule(overrides: Partial<TtsrRule> & { name: string }): TtsrRule {
	return {
		path: "test.md",
		description: "test rule",
		content: `${overrides.name} content`,
		conditions: [],
		scope: ["text", "tool"],
		interruptMode: "always",
		...overrides,
	};
}

describe("TtsrCoordinator stream rules", () => {
	test("loads built-in rules when builtinRules is enabled", () => {
		const { coordinator } = harness({ builtinRules: true });
		expect(coordinator.getRules().length).toBe(7);
	});

	test("interrupting text match replaces the stream with the rule", () => {
		const { coordinator, interrupts, events } = harness({ builtinRules: true });
		coordinator.processEvent({ type: "token", token: 'password: "' });
		coordinator.processEvent({ type: "token", token: 'mySecretKey12345"' });

		expect(interrupts).toHaveLength(1);
		expect(interrupts[0]).toContain('rule="secret-exposure"');
		expect(events.map(event => event.type)).toEqual([
			"ttsr_triggered",
			"ttsr_injected",
		]);
	});

	test("tool-scoped rules see the tool name from tool_call_start", () => {
		const { coordinator, interrupts } = harness({ builtinRules: true });
		coordinator.processEvent({
			type: "tool_call_start",
			toolName: "bash",
			toolCallId: "tc1",
			args: {},
		});
		coordinator.processEvent({
			type: "tool_call_update",
			toolCallId: "tc1",
			delta: '{"command":"sudo rm -rf /var',
		});
		expect(interrupts).toHaveLength(1);
		expect(interrupts[0]).toContain('rule="unsafe-shell"');
	});

	test("tool-scoped rules ignore other tools", () => {
		const { coordinator, interrupts } = harness({ builtinRules: true });
		coordinator.processEvent({
			type: "tool_call_start",
			toolName: "write",
			toolCallId: "tc1",
			args: {},
		});
		coordinator.processEvent({
			type: "tool_call_update",
			toolCallId: "tc1",
			delta: '{"content":"sudo rm -rf /var"}',
		});
		expect(interrupts).toHaveLength(0);
	});

	test("non-interrupting tool match folds into that call's result once", () => {
		const { coordinator, interrupts, followUps } = harness();
		coordinator.addRule(
			rule({
				name: "no-console",
				conditions: ["console\\.log"],
				interruptMode: "never",
			}),
		);
		coordinator.processEvent({
			type: "tool_call_start",
			toolName: "write",
			toolCallId: "tc1",
			args: {},
		});
		coordinator.processEvent({
			type: "tool_call_update",
			toolCallId: "tc1",
			delta: '{"path":"a.ts","content":"console.log(1)"}',
		});

		expect(interrupts).toHaveLength(0);
		expect(followUps).toHaveLength(0);
		expect(coordinator.buildToolReminder("tc2")).toBeNull();
		expect(coordinator.buildToolReminder("tc1")).toContain('rule="no-console"');
		expect(coordinator.buildToolReminder("tc1")).toBeNull();
	});

	test("path globs use the path streamed in the arguments", () => {
		const { coordinator, interrupts } = harness();
		coordinator.addRule(
			rule({ name: "py-only", conditions: ["import os"], globs: ["**/*.py"] }),
		);
		coordinator.processEvent({
			type: "tool_call_start",
			toolName: "write",
			toolCallId: "ts",
			args: {},
		});
		coordinator.processEvent({
			type: "tool_call_update",
			toolCallId: "ts",
			delta: '{"path":"src/a.ts","content":"import os"}',
		});
		expect(interrupts).toHaveLength(0);

		coordinator.processEvent({
			type: "tool_call_start",
			toolName: "write",
			toolCallId: "py",
			args: {},
		});
		coordinator.processEvent({
			type: "tool_call_update",
			toolCallId: "py",
			delta: '{"path":"src/a.py","content":"import os"}',
		});
		expect(interrupts).toHaveLength(1);
	});

	test("non-interrupting prose match is queued as a follow-up", () => {
		const { coordinator, interrupts, followUps } = harness({
			builtinRules: true,
		});
		coordinator.processEvent({
			type: "token",
			token: "I will add documentation for this later.",
		});
		expect(interrupts).toHaveLength(0);
		expect(followUps).toHaveLength(1);
		expect(followUps[0]).toContain('rule="doc-gap"');
	});

	test("persistInjected/restoreInjected round-trip keeps a rule from re-firing", () => {
		const first = harness({ builtinRules: true });
		first.coordinator.processEvent({
			type: "token",
			token: 'password: "secret12345678"',
		});
		const injected = first.coordinator.persistInjected();
		expect(injected).toContain("secret-exposure");

		const second = harness({ builtinRules: true });
		second.coordinator.restoreInjected(injected);
		second.coordinator.processEvent({
			type: "token",
			token: 'password: "secret12345678"',
		});
		expect(second.interrupts).toHaveLength(0);
	});
});

describe("TtsrCoordinator AST rules", () => {
	test("interrupting AST match blocks the finalized call", async () => {
		const { coordinator, events } = harness();
		coordinator.addRule(
			rule({ name: "no-eval", astConditions: ["eval($A)"], scope: ["tool"] }),
		);
		const blocked = await coordinator.beforeToolCall(
			{ id: "tc1", name: "write" },
			{ path: "src/a.ts", content: "const x = eval(input);" },
		);
		expect(blocked).toContain('rule="no-eval"');
		expect(blocked).toContain("was blocked and did not run");
		expect(events.some(event => event.type === "ttsr_injected")).toBe(true);
	});

	test("AST rules only match real structure, not text", async () => {
		const { coordinator } = harness();
		coordinator.addRule(
			rule({ name: "no-eval", astConditions: ["eval($A)"], scope: ["tool"] }),
		);
		const blocked = await coordinator.beforeToolCall(
			{ id: "tc1", name: "write" },
			{
				path: "src/a.ts",
				content: '// never call eval(x)\nconst s = "eval(y)";',
			},
		);
		expect(blocked).toBeUndefined();
	});

	test("non-interrupting AST match becomes a tool-result reminder", async () => {
		const { coordinator } = harness();
		coordinator.addRule(
			rule({
				name: "no-eval",
				astConditions: ["eval($A)"],
				scope: ["tool"],
				interruptMode: "never",
			}),
		);
		const blocked = await coordinator.beforeToolCall(
			{ id: "tc1", name: "edit" },
			{ path: "src/a.ts", edits: [{ oldText: "a", newText: "eval(b)" }] },
		);
		expect(blocked).toBeUndefined();
		expect(coordinator.buildToolReminder("tc1")).toContain('rule="no-eval"');
	});

	test("calls without a path never AST-match (no language)", async () => {
		const { coordinator } = harness();
		coordinator.addRule(
			rule({ name: "no-eval", astConditions: ["eval($A)"], scope: ["tool"] }),
		);
		const blocked = await coordinator.beforeToolCall(
			{ id: "tc1", name: "bash" },
			{ content: "eval(x)" },
		);
		expect(blocked).toBeUndefined();
	});
});

describe("TtsrCoordinator judged rules", () => {
	const judgedRule = rule({
		name: "no-mocks",
		question: "Does the output mock the database in an integration test?",
		scope: ["text"],
	});

	test("a yes verdict is delivered once as a follow-up warning", async () => {
		const prompts: string[] = [];
		const judge: TtsrJudge = async ({ user }) => {
			prompts.push(user);
			return '```json\n{"q0": "yes"}\n```';
		};
		const { coordinator, followUps, interrupts, events } = harness({}, judge);
		expect(coordinator.addRule(judgedRule)).toBe(true);

		// Judged rules never match mid-stream.
		coordinator.processEvent({ type: "token", token: "mock the database" });
		expect(followUps).toHaveLength(0);

		coordinator.onAssistantResponse(
			"I mocked the database in the integration test.",
		);
		await coordinator.settleJudgments();
		expect(interrupts).toHaveLength(0);
		expect(followUps).toHaveLength(1);
		expect(followUps[0]).toContain("Rule judge flagged your reply");
		expect(prompts[0]).toContain("q0: Does the output mock the database");
		expect(
			events.some(
				event =>
					event.type === "ttsr_triggered" && event.streamKey === "judge:text",
			),
		).toBe(true);

		// Claimed: the next output is not judged again for this rule.
		coordinator.onAssistantResponse("Mocked it again.");
		await coordinator.settleJudgments();
		expect(prompts).toHaveLength(1);
	});

	test("no verdict, unparseable replies, and judge errors deliver nothing", async () => {
		for (const reply of ['{"q0":"no"}', "I think so?", ""]) {
			const { coordinator, followUps } = harness({}, async () => reply);
			coordinator.addRule(judgedRule);
			coordinator.onAssistantResponse("output");
			await coordinator.settleJudgments();
			expect(followUps).toHaveLength(0);
		}
		const failing = harness({}, async () => {
			throw new Error("judge down");
		});
		failing.coordinator.addRule(judgedRule);
		failing.coordinator.onAssistantResponse("output");
		await failing.coordinator.settleJudgments();
		expect(failing.followUps).toHaveLength(0);
		expect(failing.judgeErrors).toHaveLength(1);
	});

	test("conditions prefilter which outputs reach the judge", async () => {
		let calls = 0;
		const { coordinator } = harness({}, async () => {
			calls++;
			return '{"q0":"no"}';
		});
		coordinator.addRule({ ...judgedRule, conditions: ["\\bmock"] });
		coordinator.onAssistantResponse("Refactored the parser.");
		await coordinator.settleJudgments();
		expect(calls).toBe(0);
		coordinator.onAssistantResponse("Added a mock for the client.");
		await coordinator.settleJudgments();
		expect(calls).toBe(1);
	});

	test("verdicts from a replaced session are dropped", async () => {
		let release: (() => void) | undefined;
		const gate = new Promise<void>(resolve => {
			release = resolve;
		});
		const { coordinator, followUps } = harness({}, async () => {
			await gate;
			return '{"q0":"yes"}';
		});
		coordinator.addRule(judgedRule);
		coordinator.onAssistantResponse("output");
		coordinator.newSession();
		release?.();
		await coordinator.settleJudgments();
		expect(followUps).toHaveLength(0);
	});

	test("judge: false disables judged rules", async () => {
		let calls = 0;
		const { coordinator } = harness({ judge: false }, async () => {
			calls++;
			return '{"q0":"yes"}';
		});
		coordinator.addRule(judgedRule);
		coordinator.onAssistantResponse("output");
		await coordinator.settleJudgments();
		expect(calls).toBe(0);
	});

	test("tool calls are judged with their path in the subject", async () => {
		const users: string[] = [];
		const { coordinator, followUps } = harness({}, async ({ user }) => {
			users.push(user);
			return '{"q0":"yes"}';
		});
		coordinator.addRule({ ...judgedRule, scope: ["tool"], globs: ["**/*.ts"] });
		await coordinator.beforeToolCall(
			{ id: "tc1", name: "write" },
			{ path: "src/db.test.ts", content: "jest.mock('db')" },
		);
		await coordinator.settleJudgments();
		expect(users[0]).toContain("`write` call on `src/db.test.ts`");
		expect(users[0]).toContain("jest.mock('db')");
		expect(followUps).toHaveLength(1);
	});
});
