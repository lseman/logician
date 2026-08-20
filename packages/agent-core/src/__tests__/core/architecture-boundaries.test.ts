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
	test("source layers follow the declared dependency direction", async () => {
		const sourceRoot = path.resolve(import.meta.dir, "../../");
		const allowed: Record<string, ReadonlySet<string>> = {
			core: new Set(["core"]),
			capabilities: new Set(["core", "capabilities"]),
			infrastructure: new Set(["core", "infrastructure"]),
			adapters: new Set(["core", "infrastructure", "adapters"]),
			application: new Set([
				"core",
				"capabilities",
				"infrastructure",
				"adapters",
				"application",
			]),
		};
		const offenders: string[] = [];
		for (const [sourceLayer, permitted] of Object.entries(allowed)) {
			const layerRoot = path.join(sourceRoot, sourceLayer);
			for (const file of await sourceFiles(layerRoot)) {
				const source = await readFile(file, "utf8");
				for (const match of source.matchAll(/from\s+["'](\.[^"']+)["']/g)) {
					const target = path.resolve(path.dirname(file), match[1]);
					const relative = path.relative(sourceRoot, target);
					if (relative.startsWith("..")) continue;
					const targetLayer = relative.split(path.sep)[0];
					if (!permitted.has(targetLayer)) {
						offenders.push(`${path.relative(sourceRoot, file)} -> ${relative}`);
					}
				}
			}
		}
		expect(offenders).toEqual([]);
	});

	test("core does not import product feature packages", async () => {
		const coreRoot = path.resolve(import.meta.dir, "../../core");
		const offenders: string[] = [];
		for (const file of await sourceFiles(coreRoot)) {
			if ((await readFile(file, "utf8")).includes("@logician/agent-blocks")) {
				offenders.push(path.relative(coreRoot, file));
			}
		}
		expect(offenders).toEqual([]);
	});

	test("client protocol is independent and application events use it", async () => {
		const sourceRoot = path.resolve(import.meta.dir, "../../");
		const protocolRoot = path.resolve(sourceRoot, "../../agent-protocol/src");
		const protocolOffenders: string[] = [];
		for (const file of await sourceFiles(protocolRoot)) {
			const source = await readFile(file, "utf8");
			if (/from\s+["']@logician\//.test(source)) {
				protocolOffenders.push(path.relative(protocolRoot, file));
			}
		}
		expect(protocolOffenders).toEqual([]);

		const applicationRoot = path.join(sourceRoot, "application");
		const legacyEventImports: string[] = [];
		for (const file of await sourceFiles(applicationRoot)) {
			if (
				(await readFile(file, "utf8")).includes("core/types/runtime-events")
			) {
				legacyEventImports.push(path.relative(applicationRoot, file));
			}
		}
		expect(legacyEventImports).toEqual([]);
	});

	test("removed compatibility seams cannot be reintroduced", async () => {
		const sourceRoot = path.resolve(import.meta.dir, "../../");
		const packageRoot = path.resolve(sourceRoot, "..");
		const packageJson = JSON.parse(
			await readFile(path.join(packageRoot, "package.json"), "utf8"),
		) as { exports?: Record<string, string> };
		expect(packageJson.exports?.["./events"]).toBeUndefined();
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
