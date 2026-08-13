import { expect, test } from "bun:test";
import { calculateTotal } from "./src/calculator.ts";

test("totals quantities and applies a percentage discount", () => {
	expect(
		calculateTotal(
			[
				{ price: 12.5, quantity: 2 },
				{ price: 4, quantity: 3 },
			],
			10,
		),
	).toBe(33.3);
});

test("does not mutate input", () => {
	const items = [{ price: 7, quantity: 2 }];
	calculateTotal(items);
	expect(items).toEqual([{ price: 7, quantity: 2 }]);
});
