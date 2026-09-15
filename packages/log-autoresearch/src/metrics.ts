/** Parses `METRIC name=value` lines that an experiment command prints to stdout/stderr. */

const METRIC_LINE_PREFIX = "METRIC";

const DENIED_METRIC_NAMES = new Set(["__proto__", "constructor", "prototype"]);

export function parseMetricLines(output: string): Map<string, number> {
	const metrics = new Map<string, number>();
	const regex = new RegExp(
		`^${METRIC_LINE_PREFIX}\\s+([\\w.µ]+)=(\\S+)\\s*$`,
		"gm",
	);
	for (const match of output.matchAll(regex)) {
		const name = match[1];
		if (DENIED_METRIC_NAMES.has(name)) continue;
		const value = Number(match[2]);
		if (Number.isFinite(value)) {
			metrics.set(name, value);
		}
	}
	return metrics;
}
