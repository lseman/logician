import { test } from "bun:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import {
	createPostEditDiagnosticHooks,
	diagnoseEditedFile,
} from "../../infrastructure/developer-tools/post-edit-diagnostics.ts";

function workspace(): string {
	return mkdtempSync(path.join(tmpdir(), "logician-diagnostics-"));
}

void test("diagnoseEditedFile reports bounded TypeScript syntax errors", async () => {
	const cwd = workspace();
	writeFileSync(path.join(cwd, "broken.ts"), "const value = ;\n", "utf8");

	const diagnostics = await diagnoseEditedFile(cwd, "broken.ts");

	assert.ok(diagnostics.length > 0);
	assert.equal(diagnostics[0]?.line, 1);
	assert.match(diagnostics[0]?.message ?? "", /Expression expected/);
});

void test("diagnoseEditedFile reports malformed JSON", async () => {
	const cwd = workspace();
	writeFileSync(path.join(cwd, "broken.json"), '{"ok": true,}', "utf8");

	const diagnostics = await diagnoseEditedFile(cwd, "broken.json");

	assert.ok(diagnostics.length > 0);
	assert.equal(diagnostics[0]?.line, 1);
});

void test("diagnoseEditedFile honors configured paths outside CWD", async () => {
	const cwd = workspace();
	const allowed = workspace();
	const file = path.join(allowed, "broken.json");
	writeFileSync(file, '{"ok": true,}', "utf8");

	const diagnostics = await diagnoseEditedFile(cwd, file, [allowed]);

	assert.ok(diagnostics.length > 0);
	assert.equal(diagnostics[0]?.line, 1);
});

void test("diagnoseEditedFile honors allowAllPaths outside CWD", async () => {
	const cwd = workspace();
	const outside = workspace();
	const file = path.join(outside, "broken.json");
	writeFileSync(file, '{"ok": true,}', "utf8");

	const diagnostics = await diagnoseEditedFile(cwd, file, undefined, true);

	assert.ok(diagnostics.length > 0);
	assert.equal(diagnostics[0]?.line, 1);
});

void test("diagnoseEditedFile uses the nearest project for semantic errors", async () => {
	const cwd = workspace();
	writeFileSync(
		path.join(cwd, "tsconfig.json"),
		JSON.stringify({ compilerOptions: { strict: true, noEmit: true } }),
		"utf8",
	);
	writeFileSync(
		path.join(cwd, "semantic.ts"),
		"const value: string = 1;\n",
		"utf8",
	);

	const diagnostics = await diagnoseEditedFile(cwd, "semantic.ts");

	assert.ok(diagnostics.some(item => item.code === 2322));
});

void test("post-edit hook appends diagnostics after a successful edit", async () => {
	const cwd = workspace();
	writeFileSync(path.join(cwd, "broken.ts"), "const value = ;\n", "utf8");
	const hook = createPostEditDiagnosticHooks(cwd).afterToolCall;
	assert.ok(hook);

	const result = await hook(
		{
			toolCall: { id: "call_1", name: "edit_file", arguments: "{}" },
			args: { path: "broken.ts" },
			result: "Successfully replaced 1 block(s) in broken.ts.",
			isError: false,
			iteration: 1,
		},
		undefined,
	);

	assert.match(result?.content ?? "", /post_edit_diagnostics/);
	assert.match(result?.content ?? "", /broken\.ts:1:/);
});

void test("post-edit hook separates LSP source and symbolic code", async () => {
	const cwd = workspace();
	writeFileSync(path.join(cwd, "broken.cpp"), "broken", "utf8");
	const lspManager = {
		diagnosticsFor: async () => [
			{
				line: 4,
				column: 2,
				message: "No matching function for call.",
				source: "clang",
				code: "ovl_no_viable_function_in_call",
			},
		],
	};
	const hook = createPostEditDiagnosticHooks(
		cwd,
		() => true,
		lspManager as never,
	).afterToolCall;
	assert.ok(hook);

	const result = await hook(
		{
			toolCall: { id: "call_cpp", name: "edit_file", arguments: "{}" },
			args: { path: "broken.cpp" },
			result: "Successfully replaced 1 block(s) in broken.cpp.",
			isError: false,
			iteration: 1,
		},
		undefined,
	);

	assert.match(
		result?.content ?? "",
		/broken\.cpp:4:2 clang ovl_no_viable_function_in_call:/,
	);
});

void test("post-edit hook stays silent for valid and failed edits", async () => {
	const cwd = workspace();
	writeFileSync(path.join(cwd, "valid.ts"), "const value = 1;\n", "utf8");
	const hook = createPostEditDiagnosticHooks(cwd).afterToolCall;
	assert.ok(hook);
	const base = {
		toolCall: { id: "call_1", name: "edit_file", arguments: "{}" },
		args: { path: "valid.ts" },
		iteration: 1,
	};

	assert.equal(
		await hook(
			{
				...base,
				result: "Successfully replaced 1 block(s) in valid.ts.",
				isError: false,
			},
			undefined,
		),
		undefined,
	);
	assert.equal(
		await hook(
			{ ...base, result: "Could not edit file.", isError: true },
			undefined,
		),
		undefined,
	);
});

void test("post-edit hook can be toggled without rebuilding it", async () => {
	const cwd = workspace();
	writeFileSync(path.join(cwd, "broken.ts"), "const value = ;\n", "utf8");
	let enabled = false;
	const hook = createPostEditDiagnosticHooks(cwd, () => enabled).afterToolCall;
	assert.ok(hook);
	const input = {
		toolCall: { id: "call_toggle", name: "edit_file", arguments: "{}" },
		args: { path: "broken.ts" },
		result: "Successfully replaced 1 block(s) in broken.ts.",
		isError: false,
		iteration: 1,
	};
	assert.equal(await hook(input, undefined), undefined);
	enabled = true;
	assert.match(
		(await hook(input, undefined))?.content ?? "",
		/post_edit_diagnostics/,
	);
});
