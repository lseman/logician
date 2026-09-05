import { expect, test } from "bun:test";
import { createVerifiedStopPolicy } from "../../control/policy/verified-stop-policy.ts";

const policy = createVerifiedStopPolicy();

test("verified-stop ignores a no-op mutation receipt", async () => {
	const decision = await policy.evaluate({
		messages: [],
		iteration: 1,
		newMessages: [
			{
				role: "assistant",
				content: null,
				tool_calls: [{ id: "edit", name: "edit_file", arguments: "{}" }],
			},
			{
				role: "tool",
				tool_call_id: "edit",
				content: "No changes made",
				details: {
					mutation: {
						kind: "mutation",
						applied: false,
						changed: false,
						paths: [],
						filesAffected: 0,
						revisions: [],
					},
				},
			},
		],
	});
	expect(decision).toBeUndefined();
});

test("verified-stop requests verification after an edit", async () => {
	const decision = await policy.evaluate({
		messages: [],
		iteration: 1,
		newMessages: [
			{
				role: "assistant",
				content: null,
				tool_calls: [
					{ id: "edit", name: "edit_file", arguments: '{"path":"a.ts"}' },
				],
			},
			{
				role: "tool",
				tool_call_id: "edit",
				content: "Applied",
				details: {
					mutation: {
						kind: "mutation",
						applied: true,
						changed: true,
						paths: ["a.ts"],
						filesAffected: 1,
						revisions: [{ path: "a.ts", beforeHash: "before", afterHash: "after" }],
					},
				},
			},
		],
	});
	expect(decision?.action).toBe("continue");
});

test("verified-stop accepts successful verification after the final edit", async () => {
	const decision = await policy.evaluate({
		messages: [],
		iteration: 2,
		newMessages: [
			{
				role: "assistant",
				content: null,
				tool_calls: [
					{ id: "edit", name: "edit_file", arguments: '{"path":"a.ts"}' },
				],
			},
			{
				role: "tool",
				tool_call_id: "edit",
				content: "Applied",
				details: {
					mutation: {
						kind: "mutation",
						applied: true,
						changed: true,
						paths: ["a.ts"],
						filesAffected: 1,
						revisions: [{ path: "a.ts", beforeHash: "before", afterHash: "after" }],
					},
				},
			},
			{
				role: "assistant",
				content: null,
				tool_calls: [
					{ id: "verify", name: "bash", arguments: '{"command":"bun test"}' },
				],
			},
			{ role: "tool", tool_call_id: "verify", content: "12 pass, 0 fail" },
		],
	});
	expect(decision).toBeUndefined();
});

test("verified-stop rejects failed or stale verification", async () => {
	const decision = await policy.evaluate({
		messages: [],
		iteration: 3,
		newMessages: [
			{
				role: "assistant",
				content: null,
				tool_calls: [
					{ id: "verify", name: "bash", arguments: '{"command":"bun test"}' },
				],
			},
			{ role: "tool", tool_call_id: "verify", content: "0 fail" },
			{
				role: "assistant",
				content: null,
				tool_calls: [
					{ id: "edit", name: "write_file", arguments: '{"path":"a.ts"}' },
				],
			},
			{
				role: "tool",
				tool_call_id: "edit",
				content: "Applied",
				details: {
					mutation: {
						kind: "mutation",
						applied: true,
						changed: true,
						paths: ["a.ts"],
						filesAffected: 1,
						revisions: [{ path: "a.ts", beforeHash: "before", afterHash: "after" }],
					},
				},
			},
		],
	});
	expect(decision?.action).toBe("continue");
});
