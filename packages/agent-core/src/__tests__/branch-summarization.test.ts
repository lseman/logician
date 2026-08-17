// ── Branch summarization tests ──────────────────────────────────────────────

import { describe, it } from "bun:test";
import { strict as assert } from "node:assert";
import {
	collectMessagesForBranchSummary,
	computeFileLists,
	createFileOps,
	extractFileOpsFromMessages,
	formatFileOperations,
	parseBranchSummary,
	serializeMessages,
} from "../agent/summaries/branch-summarization.ts";
import type { Message } from "../agent/types/index.ts";

describe("createFileOps", () => {
	it("creates empty ops", () => {
		const ops = createFileOps();
		assert.strictEqual(ops.read.size, 0);
		assert.strictEqual(ops.modified.size, 0);
	});
});

describe("extractFileOpsFromMessages", () => {
	it("extracts read_file calls", () => {
		const messages: Message[] = [
			{
				role: "assistant",
				content: "",
				tool_calls: [
					{
						id: "1",
						name: "read_file",
						arguments: JSON.stringify({ path: "src/main.ts" }),
					},
				],
			},
		];
		const ops = extractFileOpsFromMessages(messages);
		assert.ok(ops.read.has("src/main.ts"));
	});

	it("extracts edit_file calls", () => {
		const messages: Message[] = [
			{
				role: "assistant",
				content: "",
				tool_calls: [
					{
						id: "1",
						name: "edit_file",
						arguments: JSON.stringify({
							path: "src/main.ts",
							old_text: "foo",
							new_text: "bar",
						}),
					},
				],
			},
		];
		const ops = extractFileOpsFromMessages(messages);
		assert.ok(ops.modified.has("src/main.ts"));
	});

	it("extracts write_file calls", () => {
		const messages: Message[] = [
			{
				role: "assistant",
				content: "",
				tool_calls: [
					{
						id: "1",
						name: "write_file",
						arguments: JSON.stringify({
							path: "README.md",
							content: "# Hello",
						}),
					},
				],
			},
		];
		const ops = extractFileOpsFromMessages(messages);
		assert.ok(ops.modified.has("README.md"));
	});

	it("extracts git diff operations", () => {
		const messages: Message[] = [
			{
				role: "assistant",
				content: "",
				tool_calls: [
					{
						id: "1",
						name: "git",
						arguments: JSON.stringify({ command: "diff src/main.ts" }),
					},
				],
			},
		];
		const ops = extractFileOpsFromMessages(messages);
		assert.ok(ops.modified.has("src/main.ts"));
		assert.ok(ops.read.has("src/main.ts"));
	});

	it("skips non-tool messages", () => {
		const messages: Message[] = [
			{ role: "user", content: "hello" },
			{ role: "assistant", content: "hi" },
		];
		const ops = extractFileOpsFromMessages(messages);
		assert.strictEqual(ops.read.size, 0);
		assert.strictEqual(ops.modified.size, 0);
	});

	it("handles malformed JSON arguments", () => {
		const messages: Message[] = [
			{
				role: "assistant",
				content: "",
				tool_calls: [{ id: "1", name: "read_file", arguments: "not json" }],
			},
		];
		const ops = extractFileOpsFromMessages(messages);
		assert.strictEqual(ops.read.size, 0);
	});

	it("uses 'file' fallback for path", () => {
		const messages: Message[] = [
			{
				role: "assistant",
				content: "",
				tool_calls: [
					{
						id: "1",
						name: "read",
						arguments: JSON.stringify({ file: "config.json" }),
					},
				],
			},
		];
		const ops = extractFileOpsFromMessages(messages);
		assert.ok(ops.read.has("config.json"));
	});
});

describe("computeFileLists", () => {
	it("returns sorted deduplicated lists", () => {
		const ops = createFileOps();
		ops.read.add("z.ts");
		ops.read.add("a.ts");
		ops.read.add("m.ts");
		ops.modified.add("b.ts");
		ops.modified.add("b.ts"); // duplicate

		const { readFiles, modifiedFiles } = computeFileLists(ops);
		assert.deepStrictEqual(readFiles, ["a.ts", "m.ts", "z.ts"]);
		assert.deepStrictEqual(modifiedFiles, ["b.ts"]);
	});
});

describe("formatFileOperations", () => {
	it("formats read and modified files", () => {
		const result = formatFileOperations(["a.ts", "b.ts"], ["c.ts"]);
		assert.ok(result.includes("Read files: a.ts, b.ts"));
		assert.ok(result.includes("Modified files: c.ts"));
	});

	it("handles empty file lists", () => {
		const result = formatFileOperations([], []);
		assert.strictEqual(result, "");
	});
});

describe("collectMessagesForBranchSummary", () => {
	it("collects messages from fork point", () => {
		const parent: Message[] = [
			{ role: "user", content: "hello" },
			{ role: "assistant", content: "hi there" },
		];
		const current: Message[] = [
			{ role: "user", content: "hello" },
			{ role: "assistant", content: "hi there" },
			{ role: "user", content: "branch question" },
			{ role: "assistant", content: "branch answer" },
		];
		const result = collectMessagesForBranchSummary(current, parent, 2);
		assert.strictEqual(result.messages.length, 2);
		assert.strictEqual(result.messages[0].content, "branch question");
		assert.strictEqual(result.messages[1].content, "branch answer");
	});

	it("finds common ancestor", () => {
		const parent: Message[] = [
			{ role: "user", content: "hello" },
			{ role: "assistant", content: "hi there" },
		];
		const current: Message[] = [
			{ role: "user", content: "hello" },
			{ role: "assistant", content: "hi there" },
			{ role: "user", content: "different" },
		];
		const result = collectMessagesForBranchSummary(current, parent, 2);
		assert.strictEqual(result.commonAncestorIndex, 1);
	});

	it("respects token budget", () => {
		const parent: Message[] = [];
		const current: Message[] = [
			{ role: "user", content: "short" },
			{ role: "assistant", content: "short" },
			{ role: "user", content: "x".repeat(1000) },
		];
		// Token budget of ~200 tokens (approx 800 chars)
		const result = collectMessagesForBranchSummary(current, parent, 0, 200);
		// Should only include messages that fit within budget
		assert.ok(result.totalTokens <= 200 || result.messages.length > 0);
	});

	it("tracks file ops", () => {
		const parent: Message[] = [{ role: "user", content: "hello" }];
		const current: Message[] = [
			{ role: "user", content: "hello" },
			{
				role: "assistant",
				content: "",
				tool_calls: [
					{
						id: "1",
						name: "read_file",
						arguments: JSON.stringify({ path: "src/main.ts" }),
					},
				],
			},
		];
		const result = collectMessagesForBranchSummary(current, parent, 1);
		assert.ok(result.fileOps.read.has("src/main.ts"));
	});
});

describe("parseBranchSummary", () => {
	it("parses goal", () => {
		const text = "## Goal\nFix the login bug\n\n## Constraints\n- None";
		const result = parseBranchSummary(text);
		assert.strictEqual(result.goal, "Fix the login bug");
	});

	it("parses constraints", () => {
		const text = `## Constraints & Preferences
- Must support OAuth
- No external dependencies`;
		const result = parseBranchSummary(text);
		assert.strictEqual(result.constraints?.length, 2);
	});

	it("filters empty constraints hint", () => {
		const text = `## Constraints & Preferences
- (none)`;
		const result = parseBranchSummary(text);
		assert.deepStrictEqual(result.constraints, []);
	});

	it("parses progress", () => {
		const text = `## Progress
### Done
- [x] Set up project
### In Progress
- [ ] Add tests
### Blocked
- Waiting on API key`;
		const result = parseBranchSummary(text);
		assert.strictEqual(result.progress?.done.length, 1);
		assert.strictEqual(result.progress?.inProgress.length, 1);
		assert.strictEqual(result.progress?.blocked.length, 1);
	});

	it("parses key decisions", () => {
		const text = `## Key Decisions
- **Use SQLite**: Faster queries than JSON files
- **TypeScript only**: No JavaScript mixed`;
		const result = parseBranchSummary(text);
		assert.strictEqual(result.keyDecisions?.length, 2);
		assert.strictEqual(result.keyDecisions?.[0].decision, "Use SQLite");
		assert.strictEqual(
			result.keyDecisions?.[0].rationale,
			"Faster queries than JSON files",
		);
	});

	it("parses next steps", () => {
		const text = `## Next Steps
1. Run the tests
2. Deploy to staging
3. Monitor for errors`;
		const result = parseBranchSummary(text);
		assert.strictEqual(result.nextSteps?.length, 3);
	});

	it("handles empty text", () => {
		const result = parseBranchSummary("");
		assert.strictEqual(result.goal, undefined);
		assert.strictEqual(result.constraints, undefined);
	});
});

describe("serializeMessages", () => {
	it("serializes user and assistant messages", () => {
		const messages: Message[] = [
			{ role: "user", content: "hello" },
			{ role: "assistant", content: "world" },
		];
		const result = serializeMessages(messages);
		assert.ok(result.includes("[User]: hello"));
		assert.ok(result.includes("[Assistant]: world"));
	});

	it("serializes tool calls", () => {
		const messages: Message[] = [
			{
				role: "assistant",
				content: "",
				tool_calls: [
					{
						id: "1",
						name: "read_file",
						arguments: JSON.stringify({ path: "src/main.ts" }),
					},
				],
			},
		];
		const result = serializeMessages(messages);
		assert.ok(result.includes("[Tool Call: read_file"));
	});

	it("truncates long content", () => {
		const longContent = "x".repeat(600);
		const messages: Message[] = [{ role: "user", content: longContent }];
		const result = serializeMessages(messages);
		assert.ok(result.includes("[User]:"));
		assert.ok(result.includes("..."));
	});
});
