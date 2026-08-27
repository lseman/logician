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

async function workspacePackageRoots(workspaceRoot: string): Promise<string[]> {
	const roots: string[] = [];
	for (const parent of [
		path.join(workspaceRoot, "packages"),
		path.join(workspaceRoot, "packages", "blocks"),
		path.join(workspaceRoot, "apps"),
	]) {
		for (const entry of await readdir(parent, { withFileTypes: true })) {
			if (!entry.isDirectory() || entry.name === "blocks") continue;
			roots.push(path.join(parent, entry.name));
		}
	}
	roots.push(path.join(workspaceRoot, "ecosystem", "memoriam"));
	return roots;
}

async function workspaceDependencyGraph(
	workspaceRoot: string,
): Promise<Map<string, string[]>> {
	const graph = new Map<string, string[]>();
	for (const packageRoot of await workspacePackageRoots(workspaceRoot)) {
		const manifest = JSON.parse(
			await readFile(path.join(packageRoot, "package.json"), "utf8"),
		) as { name: string; dependencies?: Record<string, string> };
		graph.set(
			manifest.name,
			Object.keys(manifest.dependencies ?? {}).filter(name =>
				name.startsWith("@logician/"),
			),
		);
	}
	return graph;
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
		const workspaceRoot = path.resolve(import.meta.dir, "../../../..");
		const graph = await workspaceDependencyGraph(workspaceRoot);

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

	test("workspace dependencies resolve to declared workspaces", async () => {
		const workspaceRoot = path.resolve(import.meta.dir, "../../../..");
		const graph = await workspaceDependencyGraph(workspaceRoot);
		const missing = [...graph.entries()].flatMap(([name, dependencies]) =>
			dependencies
				.filter(dependency => !graph.has(dependency))
				.map(dependency => `${name} -> ${dependency}`),
		);
		expect(missing.sort()).toEqual([]);
	});

	test("core and feature blocks do not depend on the runtime", async () => {
		const workspaceRoot = path.resolve(import.meta.dir, "../../../..");
		const graph = await workspaceDependencyGraph(workspaceRoot);
		const blockNames = new Set<string>();
		for (const packageRoot of await workspacePackageRoots(workspaceRoot)) {
			if (
				!packageRoot.includes(`${path.sep}packages${path.sep}blocks${path.sep}`)
			) {
				continue;
			}
			const manifest = JSON.parse(
				await readFile(path.join(packageRoot, "package.json"), "utf8"),
			) as { name: string };
			blockNames.add(manifest.name);
		}
		const forbidden = [...graph.entries()]
			.filter(
				([name, dependencies]) =>
					(name === "@logician/log-core" || blockNames.has(name)) &&
					dependencies.includes("@logician/log-runtime"),
			)
			.map(([name]) => name)
			.sort();
		expect(forbidden).toEqual([]);
	});
});
