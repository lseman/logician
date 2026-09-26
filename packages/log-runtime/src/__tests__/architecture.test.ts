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
		path.join(workspaceRoot, "apps"),
	]) {
		for (const entry of await readdir(parent, { withFileTypes: true })) {
			if (!entry.isDirectory() || entry.name === "blocks") continue;
			roots.push(path.join(parent, entry.name));
		}
	}
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

/**
 * Source modules: top-level folders, with each capability and adapter its own
 * module. A new top-level folder has to be added here deliberately.
 */
const TOP_LEVEL_MODULES = [
	"adapters",
	"agent",
	"capabilities",
	"config",
	"context",
	"diagnostics",
	"events",
	"resources",
	"session",
	"shared",
	"tools",
	"transcript",
	"trust",
];

function moduleOf(relative: string): string | undefined {
	const parts = relative.split(path.sep);
	if (parts.length === 1) return undefined; // public entry-point files
	if (
		(parts[0] === "capabilities" || parts[0] === "adapters") &&
		parts.length > 2
	) {
		return `${parts[0]}/${parts[1]}`;
	}
	return parts[0];
}

describe("runtime source layout", () => {
	const sourceRoot = path.resolve(import.meta.dir, "..");

	test("source contains only the known top-level modules", async () => {
		const directories = (await readdir(sourceRoot, { withFileTypes: true }))
			.filter(entry => entry.isDirectory() && entry.name !== "__tests__")
			.map(entry => entry.name)
			.sort();
		expect(directories).toEqual(TOP_LEVEL_MODULES);
	});

	test("module dependency graph is acyclic (runtime imports)", async () => {
		const transpiler = new Bun.Transpiler({ loader: "ts" });
		const graph = new Map<string, Set<string>>();
		for (const file of await sourceFiles(sourceRoot)) {
			if (file.includes(`${path.sep}__tests__${path.sep}`)) continue;
			const from = moduleOf(path.relative(sourceRoot, file));
			if (!from) continue;
			for (const { path: specifier } of transpiler.scan(
				await readFile(file, "utf8"),
			).imports) {
				if (!specifier.startsWith(".")) continue;
				const to = moduleOf(
					path.relative(
						sourceRoot,
						path.resolve(path.dirname(file), specifier),
					),
				);
				if (!to || to === from) continue;
				const edges = graph.get(from) ?? new Set<string>();
				edges.add(to);
				graph.set(from, edges);
			}
		}
		const cycles = new Set<string>();
		const visit = (name: string, stack: string[]): void => {
			const start = stack.indexOf(name);
			if (start >= 0) {
				cycles.add([...stack.slice(start), name].join(" -> "));
				return;
			}
			for (const next of graph.get(name) ?? []) visit(next, [...stack, name]);
		};
		for (const name of graph.keys()) visit(name, []);
		expect([...cycles].sort()).toEqual([]);
	});

	test("every public export resolves to a source file", async () => {
		const packageRoot = path.resolve(sourceRoot, "..");
		const manifest = JSON.parse(
			await readFile(path.join(packageRoot, "package.json"), "utf8"),
		) as { exports?: Record<string, string> };
		const missing: string[] = [];
		for (const [name, target] of Object.entries(manifest.exports ?? {})) {
			try {
				await readFile(path.resolve(packageRoot, target));
			} catch {
				missing.push(`${name} -> ${target}`);
			}
		}
		expect(missing).toEqual([]);
	});
});
