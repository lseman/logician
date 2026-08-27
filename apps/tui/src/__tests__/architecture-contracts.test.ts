import { describe, expect, test } from "bun:test";
import { readdir, readFile } from "node:fs/promises";
import { dirname, relative, resolve, sep } from "node:path";

const TUI_ROOT = resolve(import.meta.dir, "../..");
const SOURCE_ROOT = resolve(TUI_ROOT, "src");
const WORKSPACE_ROOT = resolve(TUI_ROOT, "../..");

const PROTECTED_LAYERS = new Set([
	"footer",
	"input",
	"overlays",
	"rendering",
	"state",
	"status",
	"terminal",
]);

const APPLICATION_RUNTIME_SURFACES = [
	"@logician/log-runtime/application",
	"@logician/log-runtime/configuration",
	"@logician/log-runtime/context",
	"@logician/log-runtime/developer-tools",
	"@logician/log-runtime/reasoning",
	"@logician/log-runtime/tools",
	"@logician/log-runtime/trust",
];

interface SourceImport {
	file: string;
	line: number;
	specifier: string;
}

interface PackageManifest {
	name: string;
	exports?: Record<string, unknown> | string;
}

async function sourceFiles(directory: string): Promise<string[]> {
	const files: string[] = [];
	for (const entry of await readdir(directory, { withFileTypes: true })) {
		if (entry.name === "__tests__" || entry.name === "testing") continue;
		const path = resolve(directory, entry.name);
		if (entry.isDirectory()) files.push(...(await sourceFiles(path)));
		else if (entry.name.endsWith(".ts")) files.push(path);
	}
	return files;
}

async function moduleSpecifiers(
	file: string,
	source: string,
): Promise<SourceImport[]> {
	const scan = await new Bun.Transpiler({ loader: "ts" }).scan(source);
	return scan.imports.map(item => {
		const offset = source.indexOf(item.path);
		const line = offset < 0 ? 1 : source.slice(0, offset).split("\n").length;
		return { file, line, specifier: item.path };
	});
}

async function productionImports(): Promise<SourceImport[]> {
	const imports: SourceImport[] = [];
	for (const file of await sourceFiles(SOURCE_ROOT)) {
		imports.push(
			...(await moduleSpecifiers(file, await readFile(file, "utf8"))),
		);
	}
	return imports;
}

function sourceLabel(item: SourceImport): string {
	return `${relative(SOURCE_ROOT, item.file)}:${item.line} -> ${item.specifier}`;
}

function workspacePackageName(specifier: string): string | null {
	if (!specifier.startsWith("@logician/")) return null;
	return specifier.split("/").slice(0, 2).join("/");
}

async function workspaceManifests(): Promise<Map<string, PackageManifest>> {
	const manifests = new Map<string, PackageManifest>();
	const roots = ["packages", "packages/blocks", "apps", "ecosystem"];
	for (const root of roots) {
		const rootPath = resolve(WORKSPACE_ROOT, root);
		for (const entry of await readdir(rootPath, { withFileTypes: true })) {
			if (!entry.isDirectory()) continue;
			try {
				const manifest = JSON.parse(
					await readFile(resolve(rootPath, entry.name, "package.json"), "utf8"),
				) as PackageManifest;
				if (manifest.name) manifests.set(manifest.name, manifest);
			} catch {
				// Non-package ecosystem directories are outside this contract.
			}
		}
	}
	return manifests;
}

function exportedSubpath(
	manifest: PackageManifest,
	specifier: string,
): boolean {
	const subpath = specifier.slice(manifest.name.length);
	const key = subpath ? `.${subpath}` : ".";
	if (!manifest.exports) return key === ".";
	if (typeof manifest.exports === "string") return key === ".";
	return Object.hasOwn(manifest.exports, key);
}

describe("TUI dependency contracts", () => {
	test("presentation and foundation layers do not reach into application internals", async () => {
		const violations: string[] = [];
		for (const item of await productionImports()) {
			const sourceLayer = relative(SOURCE_ROOT, item.file).split(sep)[0];
			if (!PROTECTED_LAYERS.has(sourceLayer)) continue;

			if (item.specifier.startsWith(".")) {
				const target = resolve(dirname(item.file), item.specifier);
				const targetLayer = relative(SOURCE_ROOT, target).split(sep)[0];
				if (targetLayer === "app") violations.push(sourceLabel(item));
				continue;
			}

			if (
				APPLICATION_RUNTIME_SURFACES.some(
					surface =>
						item.specifier === surface ||
						item.specifier.startsWith(`${surface}/`),
				)
			) {
				violations.push(sourceLabel(item));
			}
		}

		expect(violations).toEqual([]);
	});

	test("rendering depends only on rendering and terminal TUI layers", async () => {
		const violations = (await productionImports())
			.filter(
				item => relative(SOURCE_ROOT, item.file).split(sep)[0] === "rendering",
			)
			.filter(item => item.specifier.startsWith("."))
			.filter(item => {
				const target = resolve(dirname(item.file), item.specifier);
				const layer = relative(SOURCE_ROOT, target).split(sep)[0];
				return layer !== "rendering" && layer !== "terminal";
			})
			.map(sourceLabel);

		expect(violations).toEqual([]);
	});

	test("workspace imports use declared dependencies and exported package seams", async () => {
		const tuiManifest = JSON.parse(
			await readFile(resolve(TUI_ROOT, "package.json"), "utf8"),
		) as {
			dependencies?: Record<string, string>;
			devDependencies?: Record<string, string>;
		};
		const declared = new Set([
			...Object.keys(tuiManifest.dependencies ?? {}),
			...Object.keys(tuiManifest.devDependencies ?? {}),
		]);
		const manifests = await workspaceManifests();
		const violations: string[] = [];

		for (const item of await productionImports()) {
			const packageName = workspacePackageName(item.specifier);
			if (!packageName) continue;
			if (!declared.has(packageName)) {
				violations.push(`${sourceLabel(item)} (undeclared package)`);
				continue;
			}
			const manifest = manifests.get(packageName);
			if (!manifest) {
				violations.push(`${sourceLabel(item)} (unknown workspace package)`);
				continue;
			}
			if (!exportedSubpath(manifest, item.specifier)) {
				violations.push(`${sourceLabel(item)} (unexported package subpath)`);
			}
		}

		expect(violations).toEqual([]);
	});
});
