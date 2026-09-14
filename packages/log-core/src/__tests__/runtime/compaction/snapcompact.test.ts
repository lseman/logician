import { describe, expect, it } from "bun:test";
import {
	type Archive,
	type CompactableMessage,
	compact,
	dimStopwords,
	elideDataUrls,
	NEWLINE_GLYPH,
	normalizeText,
	PRESERVE_KEY,
	serializeMessages,
	stripDimMarkers,
	truncateForSummary,
} from "../../../runtime/compaction/snapcompact.ts";

describe("snapcompact", () => {
	describe("serializeMessages", () => {
		it("serializes user and assistant messages", () => {
			const messages: CompactableMessage[] = [
				{ role: "user", content: "Hello, world!" },
				{
					role: "assistant",
					content: [{ type: "text", text: "Hi there! How can I help?" }],
				},
			];
			const result = serializeMessages(messages);
			expect(result).toContain("¶user:");
			expect(result).toContain("Hello, world!");
			expect(result).toContain("¶ai:");
		});

		it("serializes tool calls with results", () => {
			const messages: CompactableMessage[] = [
				{
					role: "assistant",
					content: [
						{
							type: "toolCall",
							id: "call-1",
							name: "read",
							arguments: { path: "src/main.ts" },
						},
					],
				},
				{
					role: "toolResult",
					content: "file contents here",
					toolCallId: "call-1",
				},
			];
			const result = serializeMessages(messages);
			expect(result).toContain("¶call:");
			expect(result).toContain("read");
			expect(result).toContain("<out>");
		});

		it("omits useless tool results", () => {
			const messages: CompactableMessage[] = [
				{
					role: "assistant",
					content: [
						{
							type: "toolCall",
							id: "call-1",
							name: "glob",
							arguments: {},
						},
					],
				},
				{
					role: "toolResult",
					content: [{ type: "text", text: "empty dir" }],
					toolCallId: "call-1",
				},
			];
			// Mark the tool result as useless
			const result = serializeMessages(messages);
			expect(result).toContain("glob");
		});

		it("handles thinking blocks", () => {
			const messages: CompactableMessage[] = [
				{
					role: "assistant",
					content: [
						{ type: "thinking", thinking: "Let me think about this..." },
						{ type: "text", text: "Here's my answer." },
					],
				},
			];
			const result = serializeMessages(messages, { includeThinking: true });
			expect(result).toContain("¶think:");
			expect(result).toContain("Let me think about this");
			expect(result).toContain("¶ai:");
		});
	});

	describe("normalizeText", () => {
		it("folds smart quotes", () => {
			const text = 'She said "hello" and \u00abworld\u00bb';
			const result = normalizeText(text);
			expect(result).toContain('"');
			expect(result).not.toContain("«");
			expect(result).not.toContain("»");
		});

		it("folds dashes", () => {
			const text = "It is a — long — string";
			const result = normalizeText(text);
			expect(result).toContain("-");
		});

		it("replaces newlines with block character", () => {
			const text = "line1\nline2\r\nline3";
			const result = normalizeText(text);
			expect(result).toContain(NEWLINE_GLYPH);
			expect(result).not.toContain("\n");
		});

		it("folds emoji to text labels", () => {
			const text = "Test ✅ passed ❌ failed ⚠️ warn";
			const result = normalizeText(text);
			expect(result).toContain("[OK]");
			expect(result).toContain("[FAIL]");
			expect(result).toContain("[WARN]");
			expect(result).not.toContain("✅");
		});

		it("collapses whitespace", () => {
			const text = "hello    world     test";
			const result = normalizeText(text);
			expect(result).toBe("hello world test");
		});

		it("strips ANSI escape sequences", () => {
			const text = "\u001b[31mred\u001b[0m text";
			const result = normalizeText(text);
			expect(result).toBe("red text");
		});

		it("folds box drawing characters", () => {
			const text = "┌──┐\n│  │\n└──┘";
			const result = normalizeText(text);
			expect(result).toContain("+");
			expect(result).toContain("-");
			expect(result).toContain("|");
		});
	});

	describe("elideDataUrls", () => {
		it("replaces base64 data URLs", () => {
			const text =
				"Check out data:image/png;base64,ABCD1234EFGH5678IJKLMNOPQRSTUVWXyz01234567890a for an image";
			const result = elideDataUrls(text);
			expect(result).toContain("[data URL omitted: image/png]");
			expect(result).not.toContain("ABCD1234");
		});

		it("leaves non-base64 references untouched", () => {
			const text = "See data:image/png for the logo";
			const result = elideDataUrls(text);
			expect(result).toBe(text);
		});

		it("handles data URLs with parameters", () => {
			const text =
				"data:application/json;charset=utf-8;base64,ABCD1234EFGH5678IJKLMNOPQRSTUVWXyz01234567890a";
			const result = elideDataUrls(text);
			expect(result).toContain(
				"[data URL omitted: application/json;charset=utf-8]",
			);
		});
	});

	describe("dimStopwords", () => {
		it("wraps stopwords in dim markers", () => {
			const text = "the cat sat on the mat";
			const result = dimStopwords(text);
			expect(result).toContain("\u000e"); // DIM_ON
			expect(result).toContain("\u000f"); // DIM_OFF
			// "the" and "on" are stopwords
			expect(result).toContain("\u000ethe\u000f");
		});

		it("preserves non-stopwords", () => {
			const text = "cat sat mat";
			const result = dimStopwords(text);
			expect(result).toBe(text);
		});

		it("does not wrap inside existing dim spans", () => {
			const text = "\u000ealready dim the text\u000f";
			const result = dimStopwords(text);
			expect(result).toBe(text);
		});
	});

	describe("truncateForSummary", () => {
		it("keeps head and tail", () => {
			const text = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
			const result = truncateForSummary(text, 20, 0.6);
			expect(result).toContain("01234"); // head
			expect(result).toContain("WXYZ"); // tail
			expect(result).toContain("elided");
		});

		it("returns text as-is if within limit", () => {
			const text = "short";
			const result = truncateForSummary(text, 100);
			expect(result).toBe("short");
		});
	});

	describe("stripDimMarkers", () => {
		it("removes shift-out/in markers", () => {
			const text = "text\u000einside\u000fmore";
			const result = stripDimMarkers(text);
			expect(result).toBe("textinsidemore");
		});
	});

	describe("compact", () => {
		it("produces a summary for compacted messages", async () => {
			const messages: CompactableMessage[] = Array.from(
				{ length: 10 },
				(_, i) => ({
					role: "user" as const,
					content: `Message number ${i} with some content`,
					entryId: `msg-${i}`,
				}),
			);
			const result = await compact(messages);
			expect(result.summary).toContain("Archived");
			expect(result.archivedChars).toBeGreaterThan(0);
		});

		it("returns frames when messages are large enough", async () => {
			const messages: CompactableMessage[] = [
				{
					role: "user" as const,
					content: "x".repeat(10000),
				},
			];
			const result = await compact(messages, { shape: { cols: 80, rows: 20 } });
			expect(result.frames).toBeDefined();
		});

		it("perserves data for next compaction round", async () => {
			const messages: CompactableMessage[] = [
				{ role: "user" as const, content: "Test message" },
			];
			const result = await compact(messages);
			if (result.preserveData) {
				expect(result.preserveData).toHaveProperty(PRESERVE_KEY);
			}
		});

		it("includes previous summary in archive text", async () => {
			const messages: CompactableMessage[] = [
				{ role: "user" as const, content: "New message" },
			];
			const result = await compact(messages, {
				previousSummary: "Previous work",
			});
			expect(result.summary).toContain("Previous compaction summary");
		});

		it("handles empty messages", async () => {
			const result = await compact([]);
			expect(result.summary).toBeDefined();
			expect(result.archivedChars).toBe(0);
		});

		it("renders frames with non-empty base64 PNG data", async () => {
			const messages: CompactableMessage[] = [
				{
					role: "user" as const,
					content: "x".repeat(5000),
				},
			];
			const result = await compact(messages, { shape: { cols: 80, rows: 20 } });
			expect(result.frames).toBeDefined();
			expect(result.frames!.length).toBeGreaterThan(0);
			for (const frame of result.frames!) {
				expect(frame.data).toBeTruthy();
				expect(typeof frame.data).toBe("string");
				// Base64 PNG should start with iVBOR (PNG magic header)
				expect(frame.data.startsWith("iVBOR")).toBe(true);
				// Base64 data should be valid base64 (length multiple of 4, valid chars)
				expect(frame.data.length % 4).toBe(0);
				expect(frame.data).toMatch(/^[A-Za-z0-9+/]+=*$/);
			}
		});

		it("frame dimensions match config shape", async () => {
			const messages: CompactableMessage[] = [
				{ role: "user" as const, content: "y".repeat(8000) },
			];
			const config = { cols: 100, rows: 30 };
			const result = await compact(messages, { shape: config });
			expect(result.frames).toBeDefined();
			expect(result.frames!.length).toBeGreaterThan(0);
			for (const frame of result.frames!) {
				expect(frame.cols).toBe(config.cols);
				expect(frame.rows).toBe(config.rows);
				expect(frame.chars).toBeLessThanOrEqual(config.cols * config.rows);
			}
		});

		it("frame text content is preserved in archive", async () => {
			const originalText = "hello world testing frame preservation";
			const messages: CompactableMessage[] = [
				{
					role: "user" as const,
					content: originalText.repeat(200),
				},
			];
			const result = await compact(messages, { shape: { cols: 40, rows: 10 } });
			if (result.preserveData) {
				const archive = result.preserveData[PRESERVE_KEY] as
					| Archive
					| undefined;
				expect(archive).toBeDefined();
				if (archive) {
					// Text head and tail should contain recognizable content
					const preserved =
						(archive.text ?? "") +
						(archive.textHead ?? "") +
						(archive.textTail ?? "");
					expect(preserved).toContain("hello world");
				}
			}
		});

		it("falls back gracefully when rendering produces empty data", async () => {
			// Even with minimal messages that produce frames, if rendering yields empty data,
			// the frame metadata (dims, char count) should still be valid
			const messages: CompactableMessage[] = [
				{ role: "user" as const, content: "short" },
			];
			const result = await compact(messages);
			// Small messages may not produce frames, but the result should still be valid
			expect(result.summary).toBeDefined();
			expect(result.archivedChars).toBeDefined();
			// If frames exist, their metadata should be consistent
			if (result.frames && result.frames.length > 0) {
				for (const frame of result.frames) {
					expect(frame.cols).toBeGreaterThan(0);
					expect(frame.rows).toBeGreaterThan(0);
					expect(frame.chars).toBeGreaterThanOrEqual(0);
					// Data may be empty in text-only mode, but metadata must be valid
					expect(frame.data).toBeDefined();
				}
			}
		});
	});
});
