/**
 * Experiment result/state shapes and the pure math over them: best/baseline
 * metric lookup and a MAD-based confidence score. No I/O — reconstructState
 * (state-reconstruction.ts) rebuilds this from the JSONL log on disk.
 */

/** Actionable Side Information (ASI) — free-form diagnostics per experiment run. */
export interface ASI {
	[key: string]: unknown;
}

export interface ExperimentResult {
	commit: string;
	metric: number;
	/** Additional tracked metrics: { name: value } */
	metrics: Record<string, number>;
	status: "keep" | "discard" | "crash" | "checks_failed";
	description: string;
	timestamp: number;
	/** Segment index — increments on each config header */
	segment: number;
	/** Session-level confidence score */
	confidence: number | null;
	/** Actionable Side Information */
	asi?: ASI;
}

export interface MetricDef {
	name: string;
	unit: string;
}

export interface ExperimentState {
	results: ExperimentResult[];
	/** Baseline primary metric */
	bestMetric: number | null;
	bestDirection: "lower" | "higher";
	metricName: string;
	metricUnit: string;
	/** Secondary metrics definitions */
	secondaryMetrics: MetricDef[];
	name: string | null;
	/** Current segment index */
	currentSegment: number;
	/** Maximum number of experiments before auto-stopping */
	maxExperiments: number | null;
	/** Session confidence score */
	confidence: number | null;
}

export function createExperimentState(): ExperimentState {
	return {
		results: [],
		bestMetric: null,
		bestDirection: "lower",
		metricName: "metric",
		metricUnit: "",
		secondaryMetrics: [],
		name: null,
		currentSegment: 0,
		maxExperiments: null,
		confidence: null,
	};
}

export function cloneExperimentState(state: ExperimentState): ExperimentState {
	return {
		...state,
		results: state.results.map(result => ({
			...result,
			metrics: { ...result.metrics },
		})),
		secondaryMetrics: state.secondaryMetrics.map(metric => ({ ...metric })),
	};
}

export function isBetter(
	current: number,
	best: number,
	direction: "lower" | "higher",
): boolean {
	return direction === "lower" ? current < best : current > best;
}

function sortedMedian(values: number[]): number {
	if (values.length === 0) return 0;
	const sorted = [...values].sort((a, b) => a - b);
	const mid = Math.floor(sorted.length / 2);
	return sorted.length % 2 === 0
		? (sorted[mid - 1] + sorted[mid]) / 2
		: sorted[mid];
}

export function computeConfidence(
	results: ExperimentResult[],
	segment: number,
	direction: "lower" | "higher",
): number | null {
	const cur = results.filter(r => r.segment === segment && r.metric > 0);
	if (cur.length < 3) return null;

	const values = cur.map(r => r.metric);
	const median = sortedMedian(values);
	const deviations = values.map(v => Math.abs(v - median));
	const mad = sortedMedian(deviations);

	if (mad === 0) return null;

	let baseline: number | null = null;
	for (const r of cur) {
		if (r.segment === segment) {
			baseline = r.metric;
			break;
		}
	}
	if (baseline === null) return null;

	let bestKept: number | null = null;
	for (const r of cur) {
		if (r.status === "keep" && r.metric > 0) {
			if (bestKept === null || isBetter(r.metric, bestKept, direction)) {
				bestKept = r.metric;
			}
		}
	}
	if (bestKept === null || bestKept === baseline) return null;

	const delta = Math.abs(bestKept - baseline);
	return delta / mad;
}

export function currentResults(
	results: ExperimentResult[],
	segment: number,
): ExperimentResult[] {
	return results.filter(r => r.segment === segment);
}

export function findBaselineMetric(
	results: ExperimentResult[],
	segment: number,
): number | null {
	const cur = currentResults(results, segment);
	return cur.length > 0 ? cur[0].metric : null;
}

export function findBestMetric(
	results: ExperimentResult[],
	segment: number,
	direction: "lower" | "higher",
): number | null {
	const kept = currentResults(results, segment)
		.filter(r => r.status === "keep")
		.map(r => r.metric);
	if (kept.length === 0) return null;
	return direction === "lower" ? Math.min(...kept) : Math.max(...kept);
}

export function formatNum(value: number | null, unit: string): string {
	if (value === null) return "—";
	const u = unit || "";
	if (value === Math.round(value)) {
		return value.toLocaleString() + u;
	}
	return value.toFixed(2) + u;
}
