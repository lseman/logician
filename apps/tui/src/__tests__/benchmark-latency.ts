#!/usr/bin/env tsx
// Cold-start latency benchmark for the TUI
import { spawnSync } from "node:child_process";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const benchmarkDir = dirname(__filename);
const srcDir = join(benchmarkDir, "..");
const appRoot = join(benchmarkDir, "..", "..");
const repoRoot = join(appRoot, "..", "..");

function countFiles(dir: string): {
	files: number;
	totalLines: number;
	maxFile: string;
	biggestSize: number;
} {
	let files = 0,
		lines = 0,
		biggest = "",
		biggestSize = 0;
	function walk(p: string) {
		for (const entry of readdirSync(p, { withFileTypes: true })) {
			const path = join(p, entry.name);
			if (entry.isDirectory()) {
				if (!entry.name.startsWith("__")) walk(path);
			} else if (
				entry.name.endsWith(".ts") &&
				!entry.name.endsWith(".test.ts")
			) {
				files++;
				const content = readFileSync(path, "utf-8");
				const cLines = content.split("\n").length;
				lines += cLines;
				const bytes = Buffer.byteLength(content);
				if (bytes > biggestSize) {
					biggestSize = bytes;
					biggest = path;
				}
			}
		}
	}
	walk(dir);
	return { files, totalLines: lines, maxFile: biggest, biggestSize };
}

const info = countFiles(srcDir);

console.log(`=== TUI Latency Benchmark ===`);
console.log(`Source stats:`);
console.log(`  TS files (excl tests): ${info.files}`);
console.log(`  Total source lines:    ~${info.totalLines.toLocaleString()}`);
console.log(
	`  Largest file:          ${(info.maxFile || "").split("/").pop()} (${info.biggestSize} bytes)`,
);

// Run cold-start benchmark via tsx
console.log(`\n--- Cold Start Benchmarks ---`);

const results = [
	{
		label: "tsx parse index.ts only",
		command: "npx",
		args: [
			"tsx",
			"--eval",
			'import("./apps/tui/src/index.js").catch(() => {})',
		],
		cwd: repoRoot,
	},
];

for (const r of results) {
	const start = Date.now();
	const result = spawnSync(r.command, r.args, {
		cwd: r.cwd,
		timeout: 30000,
		env: { ...process.env, FORCE_COLOR: "0" },
	});
	const elapsed = Date.now() - start;
	console.log(
		`  ${r.label.padEnd(40)} ${elapsed} ms${result.error ? ` (${result.error.message})` : ""}`,
	);
}

// Run the compiled binary
console.log(`\n--- Binary Benchmarks ---`);

const binResults = [
	{
		label: "Binary: --doctor",
		command: join(appRoot, "dist", "logician"),
		args: ["--doctor"],
	},
];

for (const r of binResults) {
	const start = Date.now();
	const result = spawnSync(r.command, r.args, {
		timeout: 30000,
		env: { ...process.env, LOGICIAN_TRUST: "always" },
	});
	const elapsed = Date.now() - start;
	console.log(
		`  ${r.label.padEnd(40)} ${elapsed} ms${result.error ? ` (${result.error.message})` : ""}`,
	);
}

console.log("\nDone.");
