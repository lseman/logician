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

function packageName(specifier: string): string {
	return specifier.split("/").slice(0, 2).join("/");
}

describe("workspace package architecture", () => {
	test("runtime declares every workspace package imported by production source", async () => {
		const packageRoot = path.resolve(import.meta.dir, "../..");
		const manifest = JSON.parse(
			await readFile(path.join(packageRoot, "package.json"), "utf8"),
		) as { dependencies?: Record<string, string> };
		const imported = new Set<string>();
		for (const file of await sourceFiles(path.join(packageRoot, "src"))) {
			if (file.includes(`${path.sep}__tests__${path.sep}`)) continue;
			const source = await readFile(file, "utf8");
			for (const match of source.matchAll(
				/from\s+["'](@logician\/[^"']+)["']/g,
			)) {
				imported.add(packageName(match[1]));
			}
		}
		const undeclared = [...imported]
			.filter(name => !(name in (manifest.dependencies ?? {})))
			.sort();
		expect(undeclared).toEqual([]);
	});

	test("workspace package dependency graph is acyclic", async () => {
		const packagesRoot = path.resolve(import.meta.dir, "../../..");
		const entries = await readdir(packagesRoot, { withFileTypes: true });
		const graph = new Map<string, string[]>();
		for (const entry of entries) {
			if (!entry.isDirectory()) continue;
			try {
				const manifest = JSON.parse(
					await readFile(
						path.join(packagesRoot, entry.name, "package.json"),
						"utf8",
					),
				) as { name: string; dependencies?: Record<string, string> };
				graph.set(
					manifest.name,
					Object.keys(manifest.dependencies ?? {}).filter(name =>
						name.startsWith("@logician/"),
					),
				);
			} catch {
				// Not a workspace package.
			}
		}

		const cycles: string[] = [];
		const visit = (name: string, path: string[]): void => {
			const cycleStart = path.indexOf(name);
			if (cycleStart >= 0) {
				cycles.push([...path.slice(cycleStart), name].join(" -> "));
				return;
			}
			for (const dependency of graph.get(name) ?? []) {
				if (graph.has(dependency)) visit(dependency, [...path, name]);
			}
		};
		for (const name of graph.keys()) visit(name, []);
		expect([...new Set(cycles)].sort()).toEqual([]);
	});
});
