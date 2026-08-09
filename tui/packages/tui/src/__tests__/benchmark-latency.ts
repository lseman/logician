#!/usr/bin/env tsx
// Cold-start latency benchmark for the TUI
import { spawnSync } from "node:child_process";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const srcDir = join(__dirname, "..");

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
		cmd: `cd /home/seman/logician/tui && timeout 30 npx tsx --eval 'import("./packages/tui/src/index.js").catch(()=>{})' 2>&1; echo $?`,
	},
];

for (const r of results) {
	const start = Date.now();
	const _result = spawnSync("bash", ["-c", r.cmd], {
		timeout: 30000,
		env: { ...process.env, FORCE_COLOR: "0" },
	});
	const elapsed = Date.now() - start;
	console.log(`  ${r.label.padEnd(40)} ${elapsed} ms`);
}

// Run the compiled binary
console.log(`\n--- Binary Benchmarks ---`);

const binResults = [
	{
		label: "Binary: --doctor",
		cmd: `/home/seman/logician/tui/dist/logician --doctor 2>&1 || true`,
	},
];

for (const r of binResults) {
	const start = Date.now();
	const _result = spawnSync("bash", ["-c", r.cmd], {
		timeout: 30000,
		env: { ...process.env, LOGICIAN_TRUST: "always" },
	});
	const elapsed = Date.now() - start;
	console.log(`  ${r.label.padEnd(40)} ${elapsed} ms`);
}

console.log("\nDone.");
