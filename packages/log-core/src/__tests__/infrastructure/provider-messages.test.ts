import { expect, test } from "bun:test";
import {
	convertToChatFormat,
	convertToLlm,
} from "../../capabilities/provider/messages.ts";
import { PRESERVE_KEY } from "../../runtime/compaction/snapcompact.ts";
import type { AgentMessage } from "../../system/types/types-messages.ts";

test("convertToLlm attaches snapcompact frames as images on the compaction summary message", () => {
	const messages: AgentMessage[] = [
		{
			role: "compactionSummary",
			content: "Archived 1,234 chars. Rendered onto 2 frames.",
			tokensBefore: 100,
			timestamp: 0,
			snapcompact: {
				[PRESERVE_KEY]: {
					frames: [
						{ data: "aGVsbG8=", cols: 140, rows: 50, chars: 10 },
						{ data: "d29ybGQ=", cols: 140, rows: 50, chars: 10 },
						// Frame with no rendered data (render disabled/failed) must be dropped.
						{ data: "", cols: 140, rows: 50, chars: 0 },
					],
					totalChars: 1234,
					truncatedChars: 0,
				},
			},
		},
	];

	const [converted] = convertToLlm(messages);
	expect(converted?.role).toBe("user");
	expect(converted?.images).toEqual([
		{ data: "aGVsbG8=", mimeType: "image/png" },
		{ data: "d29ybGQ=", mimeType: "image/png" },
	]);
	expect(String(converted?.content)).toContain(
		"Archived 1,234 chars. Rendered onto 2 frames.",
	);
	expect(String(converted?.content)).toContain("archived conversation history");

	const [chatMessage] = convertToChatFormat(converted ? [converted] : []);
	expect(Array.isArray(chatMessage?.content)).toBe(true);
	const parts = chatMessage?.content as Array<Record<string, unknown>>;
	expect(parts[0]?.type).toBe("text");
	expect(parts[1]).toEqual({
		type: "image_url",
		image_url: { url: "data:image/png;base64,aGVsbG8=" },
	});
	expect(parts[2]).toEqual({
		type: "image_url",
		image_url: { url: "data:image/png;base64,d29ybGQ=" },
	});
	expect(parts.length).toBe(3);
});

test("convertToLlm omits images and the frames note when no frames are present", () => {
	const messages: AgentMessage[] = [
		{
			role: "compactionSummary",
			content: "No prior history.",
			tokensBefore: 0,
			timestamp: 0,
		},
	];

	const [converted] = convertToLlm(messages);
	expect(converted?.images).toBeUndefined();
	expect(String(converted?.content)).not.toContain("archived conversation history");

	const [chatMessage] = convertToChatFormat(converted ? [converted] : []);
	expect(typeof chatMessage?.content).toBe("string");
});
