#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
tui_dir="$repo_dir/tui"
benchmark="packages/tui/src/__tests__/benchmark-keystroke.ts"
runs="${AUTORESEARCH_RUNS:-3}"

if ! [[ "$runs" =~ ^[1-9][0-9]*$ ]]; then
	printf 'AUTORESEARCH_RUNS must be a positive integer\n' >&2
	exit 2
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

cd "$tui_dir"

# Run the real InputBar -> layout -> viewport render -> frame diff path. Each
# process gets a fresh heap; taking the median of independent runs reduces JIT,
# GC, and scheduler noise without hiding tail latency inside a single run.
for ((run = 1; run <= runs; run++)); do
	COLUMNS=120 LINES=40 npx --no-install tsx "$benchmark" --json \
		>"$tmp_dir/run-$run.json"
done

node - "$tmp_dir" "$runs" <<'NODE'
const fs = require("node:fs");
const path = require("node:path");

const directory = process.argv[2];
const runCount = Number(process.argv[3]);
const reports = Array.from({ length: runCount }, (_, index) =>
	JSON.parse(fs.readFileSync(path.join(directory, `run-${index + 1}.json`), "utf8")),
);

function bucket(report, turns) {
	const result = report.results.find(candidate => candidate.turns === turns);
	if (!result) throw new Error(`benchmark result is missing the ${turns}-turn bucket`);
	return result;
}

function median(values) {
	const sorted = [...values].sort((a, b) => a - b);
	const middle = Math.floor(sorted.length / 2);
	return sorted.length % 2
		? sorted[middle]
		: (sorted[middle - 1] + sorted[middle]) / 2;
}

function metric(name, values) {
	console.log(`METRIC ${name}=${median(values).toFixed(3)}`);
}

// 150 turns guarantees a content-heavy, fully occupied 120x40 viewport while
// remaining representative of an ordinary long coding session. The empty and
// 2500-turn buckets distinguish viewport cost from backlog-scaling regressions.
metric("keystroke_full_p50_ms", reports.map(report => bucket(report, 150).p50Ms));
metric("keystroke_full_p95_ms", reports.map(report => bucket(report, 150).p95Ms));
metric("keystroke_full_p99_ms", reports.map(report => bucket(report, 150).p99Ms));
metric("keystroke_empty_p50_ms", reports.map(report => bucket(report, 0).p50Ms));
metric("keystroke_marathon_p50_ms", reports.map(report => bucket(report, 2500).p50Ms));
metric(
	"keystroke_backlog_scaling_x",
	reports.map(report => bucket(report, 2500).p50Ms / Math.max(bucket(report, 0).p50Ms, 0.001)),
);
NODE
