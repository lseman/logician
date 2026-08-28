import { describe, expect, test } from "bun:test";
import {
	SessionCompactor,
	type SessionCompactorDependencies,
} from "../../runtime/harness/internal/session-compactor.ts";
import type { AgentConfig } from "../../system/types/types-config.ts";
import type { Message } from "../../system/types/types-messages.ts";
import { FakeBackend } from "../fake-backend.ts";

function largeHistory(): Message[] {
	return Array.from({ length: 24 }, (_, index) => ({
		role: index % 2 === 0 ? ("user" as const) : ("assistant" as const),
		content: `${index}:${"context ".repeat(1_000)}`,
	}));
}

function dependencies(
	history: () => Message[],
	overrides: Partial<SessionCompactorDependencies> = {},
): SessionCompactorDependencies {
	return {
		backend: () => new FakeBackend([]),
		history,
		commitHistory: () => true,
		config: () => ({}) as AgentConfig,
		identity: () => ({ sessionId: "session", cwd: "/workspace" }),
		extensionRunner: () => undefined,
		beforeCompact: async () => undefined,
		afterCompact: async () => {},
		persistCompaction: () => {},
		estimateTokens: () => 50_000,
		emit: () => {},
		...overrides,
	};
}

describe("SessionCompactor", () => {
	test("records durable summary metadata through its interface", () => {
		const persisted: Array<{
			summary: string;
			tokensBefore: number;
			firstKeptEntryId?: string;
		}> = [];
		const compactor = new SessionCompactor(
			dependencies(() => [], {
				persistCompaction: (summary, tokensBefore, firstKeptEntryId) =>
					persisted.push({ summary, tokensBefore, firstKeptEntryId }),
			}),
		);

		compactor.recordCompaction(
			[
				{
					role: "compactionSummary",
					content: " durable summary ",
				} as unknown as Message,
				{
					role: "user",
					content: "kept",
					entryId: "entry-kept",
				} as Message & { entryId: string },
			],
			12_345,
		);

		expect(persisted).toEqual([
			{
				summary: " durable summary ",
				tokensBefore: 12_345,
				firstKeptEntryId: "entry-kept",
			},
		]);
	});

	test("always completes post-compaction lifecycle when preparation fails", async () => {
		let postCompactCalls = 0;
		const history = largeHistory();
		const compactor = new SessionCompactor(
			dependencies(() => history, {
				beforeCompact: async () => {
					throw new Error("compaction preparation failed");
				},
				afterCompact: async () => {
					postCompactCalls++;
				},
			}),
		);

		await expect(compactor.compact("manual", true)).rejects.toThrow(
			"compaction preparation failed",
		);
		expect(postCompactCalls).toBe(1);
	});

	test("does not overwrite history changed during compaction", async () => {
		let current = largeHistory();
		let persistenceCalls = 0;
		const replacement: Message[] = [{ role: "user", content: "new history" }];
		const compactor = new SessionCompactor(
			dependencies(() => current, {
				beforeCompact: async () => {
					current = replacement;
					return { summary: "summary" };
				},
				commitHistory: (expected, compacted) => {
					if (current !== expected) return false;
					current = compacted;
					return true;
				},
				persistCompaction: () => {
					persistenceCalls++;
				},
			}),
		);

		expect(await compactor.compact("manual", true)).toBe(0);
		expect(current).toBe(replacement);
		expect(persistenceCalls).toBe(0);
	});
});
