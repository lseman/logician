import { describe, expect, test } from "bun:test";
import { readdir, readFile } from "node:fs/promises";
import path from "node:path";

const MODULE_DEPTH = new Map([
	["system", 0],
	["capabilities", 1],
	["control", 2],
	["runtime", 3],
]);

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

function relativeImports(source: string): string[] {
	return [...source.matchAll(/(?:from\s+|import\s*)["'](\.[^"']+)["']/g)].map(
		match => match[1],
	);
}

async function scannedImports(source: string): Promise<string[]> {
	const scan = await new Bun.Transpiler({ loader: "ts" }).scan(source);
	return scan.imports.map(item => item.path);
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

	test("foundational modules never depend on orchestration modules", async () => {
		const sourceRoot = path.resolve(import.meta.dir, "../");
		const violations: string[] = [];
		for (const file of await sourceFiles(sourceRoot)) {
			if (file.includes(`${path.sep}__tests__${path.sep}`)) continue;
			const sourceModule = path.relative(sourceRoot, file).split(path.sep)[0];
			const sourceDepth = MODULE_DEPTH.get(sourceModule);
			if (sourceDepth === undefined) continue;

			for (const specifier of await scannedImports(
				await readFile(file, "utf8"),
			)) {
				if (!specifier.startsWith(".")) continue;
				const target = path.resolve(path.dirname(file), specifier);
				const targetModule = path
					.relative(sourceRoot, target)
					.split(path.sep)[0];
				const targetDepth = MODULE_DEPTH.get(targetModule);
				if (targetDepth === undefined || targetDepth <= sourceDepth) continue;
				violations.push(
					`${path.relative(sourceRoot, file)} -> ${specifier} (${sourceModule} -> ${targetModule})`,
				);
			}
		}
		expect(violations.sort()).toEqual([]);
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

	test("every production module is reachable from a public export", async () => {
		const sourceRoot = path.resolve(import.meta.dir, "../");
		const packageRoot = path.resolve(sourceRoot, "..");
		const files = (await sourceFiles(sourceRoot)).filter(
			file => !file.includes(`${path.sep}__tests__${path.sep}`),
		);
		const knownFiles = new Set(files);
		const dependencies = new Map<string, string[]>();
		for (const file of files) {
			const imports = relativeImports(await readFile(file, "utf8"));
			dependencies.set(
				file,
				imports
					.map(specifier => {
						const resolved = path.resolve(path.dirname(file), specifier);
						return resolved.endsWith(".ts") ? resolved : `${resolved}.ts`;
					})
					.filter(candidate => knownFiles.has(candidate)),
			);
		}

		const packageJson = JSON.parse(
			await readFile(path.join(packageRoot, "package.json"), "utf8"),
		) as { exports?: Record<string, string> };
		const reachable = new Set<string>();
		const visit = (file: string): void => {
			if (reachable.has(file)) return;
			reachable.add(file);
			for (const dependency of dependencies.get(file) ?? []) visit(dependency);
		};
		for (const target of Object.values(packageJson.exports ?? {})) {
			visit(path.resolve(packageRoot, target));
		}

		const unreachable = files
			.filter(file => !reachable.has(file))
			.map(file => path.relative(sourceRoot, file))
			.sort();
		expect(unreachable).toEqual([]);
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
