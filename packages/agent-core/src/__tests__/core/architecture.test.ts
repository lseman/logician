import { test } from "bun:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const sourceRoot = path.resolve(
	path.dirname(fileURLToPath(import.meta.url)),
	"../..",
);
const coreRoot = path.join(sourceRoot, "core");

function sourceFiles(directory: string): string[] {
	return readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
		const file = path.join(directory, entry.name);
		return entry.isDirectory()
			? sourceFiles(file)
			: entry.isFile() && file.endsWith(".ts")
				? [file]
				: [];
	});
}

void test("core does not depend on application, adapters, capabilities, or infrastructure", () => {
	const violations: string[] = [];
	const importPattern = /(?:from\s+|import\s*)["']([^"']+)["']/g;
	for (const file of sourceFiles(coreRoot)) {
		const source = readFileSync(file, "utf8");
		for (const match of source.matchAll(importPattern)) {
			const specifier = match[1];
			if (!specifier.startsWith(".")) continue;
			const target = path.resolve(path.dirname(file), specifier);
			if (target !== coreRoot && !target.startsWith(`${coreRoot}${path.sep}`)) {
				violations.push(
					`${path.relative(sourceRoot, file)} -> ${path.relative(sourceRoot, target)}`,
				);
			}
		}
	}
	assert.deepEqual(violations, []);
});

void test("implementation modules import defining files instead of internal facades", () => {
	const violations: string[] = [];
	const importPattern = /(?:from\s+|import\s*)["']([^"']+\/index\.ts)["']/g;
	const piAdapter = path.join(sourceRoot, "adapters", "pi", "index.ts");

	for (const file of sourceFiles(sourceRoot)) {
		if (file.includes(`${path.sep}__tests__${path.sep}`)) continue;
		if (path.basename(file) === "index.ts") continue;
		const source = readFileSync(file, "utf8");
		for (const match of source.matchAll(importPattern)) {
			const specifier = match[1];
			if (!specifier.startsWith(".")) continue;
			const target = path.resolve(path.dirname(file), specifier);
			// The Pi adapter is currently implemented in its facade file.
			if (target === piAdapter) continue;
			violations.push(
				`${path.relative(sourceRoot, file)} -> ${path.relative(sourceRoot, target)}`,
			);
		}
	}

	assert.deepEqual(violations, []);
});
