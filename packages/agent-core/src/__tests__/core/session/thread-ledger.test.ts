import { describe, expect, test } from "bun:test";
import { ThreadLedger } from "../../../core/session/thread-ledger.ts";

describe("ThreadLedger", () => {
	test("keeps replacements as append-only provenance", () => {
		const ledger = new ThreadLedger();
		ledger.append([{ role: "user", content: "original" }]);
		ledger.replace([{ role: "user", content: "compacted" }], "compaction");

		expect(ledger.messages).toEqual([{ role: "user", content: "compacted" }]);
		expect(ledger.items().map(item => item.type)).toEqual([
			"message",
			"projection",
		]);
	});

	test("returns copies rather than mutable ledger state", () => {
		const ledger = new ThreadLedger();
		ledger.append([{ role: "user", content: "safe" }]);
		const messages = ledger.messages;
		messages[0].content = "mutated";
		expect(ledger.messages[0].content).toBe("safe");
	});
});
