import { describe, expect, test } from "bun:test";
import {
	SessionCompactor,
	type SessionCompactorDependencies,
} from "../../harness/internal/session-compactor.ts";
import type { AgentConfig } from "../../types/config.ts";
import type { Message } from "../../types/messages.ts";
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
		historyRevision: () => 0,
		commitHistory: () => true,
		config: () => ({}) as AgentConfig,
		identity: () => ({ sessionId: "session", cwd: "/workspace" }),
		extensionRunner: () => undefined,
		beforeCompact: async () => undefined,
		afterCompact: async () => {},
		persistCompaction: () => {},
		estimateTokens: async () => 50_000,
		contextWindowTokens: () => undefined,
		emit: () => {},
		...overrides,
	};
}

describe("SessionCompactor", () => {
	test("records durable summary metadata through its interface", async () => {
		const persisted: Array<{
			summary: string;
			tokensBefore: number;
			firstKeptEntryId?: string | undefined;
		}> = [];
		const compactor = new SessionCompactor(
			dependencies(() => [], {
				persistCompaction: (summary, tokensBefore, firstKeptEntryId) =>
					persisted.push({ summary, tokensBefore, firstKeptEntryId }),
			}),
		);

		await compactor.recordCompaction(
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
			Promise.resolve(12_345),
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
		let revision = 1;
		let persistenceCalls = 0;
		const replacement: Message[] = [{ role: "user", content: "new history" }];
		const compactor = new SessionCompactor(
			dependencies(() => current, {
				historyRevision: () => revision,
				beforeCompact: async () => {
					current = replacement;
					revision++;
					return { summary: "summary" };
				},
				commitHistory: (expectedRevision, compacted) => {
					if (revision !== expectedRevision) return false;
					current = compacted;
					revision++;
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
