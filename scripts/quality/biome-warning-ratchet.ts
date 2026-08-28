import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

interface BiomeDiagnostic {
	severity: string;
	category?: string;
	location?: { path?: string };
}

interface BiomeReport {
	summary?: { diagnosticsNotPrinted?: number };
	diagnostics?: BiomeDiagnostic[];
}

const workspaceRoot = resolve(import.meta.dir, "../..");
const baselinePath = resolve(
	workspaceRoot,
	"quality/biome-warning-baseline.json",
);
const baseline = JSON.parse(readFileSync(baselinePath, "utf8")) as Record<
	string,
	number
>;

const result = spawnSync(
	"bun",
	[
		"x",
		"biome",
		"check",
		"packages/",
		"packages/blocks",
		"apps/",
		"--reporter=json",
		"--max-diagnostics=1000",
	],
	{
		cwd: workspaceRoot,
		encoding: "utf8",
		maxBuffer: 20 * 1024 * 1024,
	},
);

if (result.error) throw result.error;
if (result.status !== 0) {
	process.stderr.write(result.stderr);
	process.stdout.write(result.stdout);
	process.exit(result.status ?? 1);
}

let report: BiomeReport;
try {
	report = JSON.parse(result.stdout) as BiomeReport;
} catch (error) {
	throw new Error("Biome warning ratchet could not parse the JSON report", {
		cause: error,
	});
}

if ((report.summary?.diagnosticsNotPrinted ?? 0) > 0) {
	throw new Error(
		"Biome warning ratchet reached its diagnostic limit; raise --max-diagnostics",
	);
}

const counts = new Map<string, number>();
for (const diagnostic of report.diagnostics ?? []) {
	if (diagnostic.severity !== "warning") continue;
	const category = diagnostic.category;
	const file = diagnostic.location?.path;
	if (!category || !file) {
		throw new Error("Biome returned a warning without a category or path");
	}
	const scope = /__tests__|\.test\.|benchmark|profile-/.test(file)
		? "test"
		: "production";
	const key = `${scope}|${category}`;
	counts.set(key, (counts.get(key) ?? 0) + 1);
}

const regressions: string[] = [];
const improvements: string[] = [];
for (const key of new Set([...Object.keys(baseline), ...counts.keys()])) {
	const allowed = baseline[key] ?? 0;
	const actual = counts.get(key) ?? 0;
	if (actual > allowed) regressions.push(`${key}: ${actual} > ${allowed}`);
	else if (actual < allowed)
		improvements.push(`${key}: ${actual} < ${allowed}`);
}

if (regressions.length > 0) {
	console.error("Biome warning baseline regressed:");
	for (const regression of regressions.sort()) console.error(`  ${regression}`);
	process.exit(1);
}

const warningCount = [...counts.values()].reduce(
	(total, count) => total + count,
	0,
);
console.log(`Biome warning ratchet passed (${warningCount} warnings).`);
if (improvements.length > 0) {
	console.log("Baseline improvements available:");
	for (const improvement of improvements.sort())
		console.log(`  ${improvement}`);
}
