import { expect, test } from "bun:test";
import { parseMetricLines } from "./metrics.ts";

test("parseMetricLines accepts finite named metrics and rejects unsafe values", () => {
	const metrics = parseMetricLines(
		[
			"METRIC accuracy=0.98",
			"METRIC latency.p95=12.4",
			"METRIC loss=NaN",
			"METRIC __proto__=1",
			"noise",
		].join("\n"),
	);

	expect(Object.fromEntries(metrics)).toEqual({
		accuracy: 0.98,
		"latency.p95": 12.4,
	});
});
