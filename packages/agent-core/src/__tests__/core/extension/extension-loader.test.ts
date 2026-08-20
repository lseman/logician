import { test } from "bun:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadExtensions } from "../../../core/extension/loader.ts";

test("discovers native global and project extension locations", () => {
	const root = mkdtempSync(join(tmpdir(), "logician-extension-loader-"));
	const globalDir = join(root, "home", ".logician", "extensions");
	const projectDir = join(root, "project");
	const projectExtensionDir = join(projectDir, ".logician", "extensions");
	mkdirSync(globalDir, { recursive: true });
	mkdirSync(projectExtensionDir, { recursive: true });
	writeFileSync(join(globalDir, "global.ts"), "export default () => {};");
	writeFileSync(
		join(projectExtensionDir, "project.ts"),
		"export default () => {};",
	);

	const result = loadExtensions({ userDir: globalDir, projectDir });
	assert.deepEqual(
		result.extensions.map(extension => extension.name),
		["global", "project"],
	);
});

test("accepts an explicit extension file", () => {
	const root = mkdtempSync(join(tmpdir(), "logician-extension-file-"));
	const file = join(root, "standalone.ts");
	writeFileSync(file, "export default () => {};");

	const result = loadExtensions({ explicitPaths: [file] });
	assert.equal(result.extensions.length, 1);
	assert.equal(result.extensions[0].path, file);
});
