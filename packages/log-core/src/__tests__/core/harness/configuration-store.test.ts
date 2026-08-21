import { describe, expect, test } from "bun:test";
import { ConfigurationStore } from "../../../core/configuration/configuration-store.ts";

describe("ConfigurationStore", () => {
	test("publishes immutable monotonic snapshots", () => {
		const store = new ConfigurationStore(
			{ model: "a", values: [1] },
			{ clone: value => ({ ...value, values: [...value.values] }) },
		);
		const initial = store.snapshot();
		const updated = store.update({ model: "b" });

		expect(initial).toEqual({
			revision: 0,
			value: { model: "a", values: [1] },
		});
		expect(updated.revision).toBe(1);
		expect(updated.value.model).toBe("b");
		expect(initial.value.model).toBe("a");
	});

	test("rejects an invalid revision without publishing it", () => {
		const store = new ConfigurationStore(
			{ count: 1 },
			{
				clone: value => ({ ...value }),
				validate: value => (value.count < 0 ? ["count must be positive"] : []),
			},
		);
		expect(() => store.update({ count: -1 })).toThrow("count must be positive");
		expect(store.snapshot()).toEqual({ revision: 0, value: { count: 1 } });
	});
});
