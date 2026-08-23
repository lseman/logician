import { describe, expect, test } from "bun:test";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";

async function sourceFiles(root: string): Promise<string[]> {
	const entries = await readdir(root, { withFileTypes: true });
	return (
		await Promise.all(
			entries.map(entry => {
				const target = path.join(root, entry.name);
				return entry.isDirectory()
					? sourceFiles(target)
					: Promise.resolve(entry.name.endsWith(".ts") ? [target] : []);
			}),
		)
	).flat();
}

describe("core architecture boundaries", () => {
	test("source contains only foundational modules (flat layout)", async () => {
		const sourceRoot = path.resolve(import.meta.dir, "../");
		const entries = await readdir(sourceRoot, { withFileTypes: true });
		const directories = entries
			.filter(entry => entry.isDirectory() && entry.name !== "__tests__")
			.map(entry => entry.name)
			.sort();
		expect(directories).toEqual([
			"capabilities",
			"control",
			"runtime",
			"system",
		]);
	});

	test("source does not import workspace feature packages", async () => {
		const sourceRoot = path.resolve(import.meta.dir, "../");
		const offenders: string[] = [];
		for (const file of await sourceFiles(sourceRoot)) {
			if (file.includes(`${path.sep}__tests__${path.sep}`)) continue;
			if (/from\s+["']@logician\//.test(await readFile(file, "utf8"))) {
				offenders.push(path.relative(sourceRoot, file));
			}
		}
		expect(offenders).toEqual([]);
	});

	test("protocol types are self-contained (no workspace deps)", async () => {
		const sourceRoot = path.resolve(import.meta.dir, "../");
		const protocolFile = path.resolve(
			sourceRoot,
			"system/types/types-protocol.ts",
		);
		const eventFile = path.resolve(sourceRoot, "system/types/types-events.ts");
		const protocolSources = [protocolFile, eventFile];
		const offenders: string[] = [];
		for (const file of protocolSources) {
			const source = await readFile(file, "utf8");
			if (/from\s+["']@logician\//.test(source)) {
				offenders.push(path.relative(sourceRoot, file));
			}
		}
		expect(offenders).toEqual([]);
	});

	test("every public export resolves to a source file", async () => {
		const packageRoot = path.resolve(import.meta.dir, "../..");
		const packageJson = JSON.parse(
			await readFile(path.join(packageRoot, "package.json"), "utf8"),
		) as { exports?: Record<string, string> };
		const missing: string[] = [];
		for (const [name, target] of Object.entries(packageJson.exports ?? {})) {
			try {
				await readFile(path.resolve(packageRoot, target));
			} catch {
				missing.push(`${name} -> ${target}`);
			}
		}
		expect(missing).toEqual([]);
	});

	test("removed compatibility seams cannot be reintroduced", async () => {
		const sourceRoot = path.resolve(import.meta.dir, "../");
		const packageRoot = path.resolve(sourceRoot, "..");
		const packageJson = JSON.parse(
			await readFile(path.join(packageRoot, "package.json"), "utf8"),
		) as { exports?: Record<string, string> };
		expect(packageJson.exports?.["./adapters/pi"]).toBeUndefined();

		const offenders: string[] = [];
		for (const file of await sourceFiles(sourceRoot)) {
			if (file.includes(`${path.sep}__tests__${path.sep}`)) continue;
			const source = await readFile(file, "utf8");
			if (
				/\bEventCallback\b|compatibilityAdapters|piCompatibilityAdapter/.test(
					source,
				)
			) {
				offenders.push(path.relative(sourceRoot, file));
			}
		}
		expect(offenders).toEqual([]);
	});
});
