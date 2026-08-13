import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { runProcess } from "./process.ts";
import type { GraderResult, GraderSpec } from "./types.ts";

function inside(workspace: string, candidate: string): string {
	const root = path.resolve(workspace);
	const resolved = path.resolve(root, candidate);
	if (resolved !== root && !resolved.startsWith(`${root}${path.sep}`))
		throw new Error(`grader path escapes workspace: ${candidate}`);
	return resolved;
}

function pathAllowed(file: string, prefixes: string[]): boolean {
	return prefixes.some(
		prefix => file === prefix || file.startsWith(`${prefix}/`),
	);
}

export async function grade(
	spec: GraderSpec,
	workspace: string,
): Promise<GraderResult> {
	const started = performance.now();
	try {
		if (spec.type === "command") {
			const result = await runProcess(spec.command, spec.args ?? [], {
				cwd: workspace,
				timeoutMs: spec.timeoutMs ?? 120_000,
			});
			return {
				id: spec.id,
				type: spec.type,
				passed: result.exitCode === 0 && !result.timedOut,
				durationMs: result.durationMs,
				summary: result.timedOut
					? "command timed out"
					: `exit ${result.exitCode}`,
				evidence: `${result.stdout}${result.stderr}`.trim().slice(-4000),
			};
		}
		if (spec.type === "file_contains") {
			const file = inside(workspace, spec.path);
			const content = existsSync(file) ? readFileSync(file, "utf8") : "";
			const passed = content.includes(spec.pattern);
			return {
				id: spec.id,
				type: spec.type,
				passed,
				durationMs: Math.round(performance.now() - started),
				summary: passed ? "pattern found" : "pattern not found",
				evidence: spec.path,
			};
		}
		if (spec.type === "file_absent") {
			const passed = !existsSync(inside(workspace, spec.path));
			return {
				id: spec.id,
				type: spec.type,
				passed,
				durationMs: Math.round(performance.now() - started),
				summary: passed ? "file absent" : "unexpected file exists",
				evidence: spec.path,
			};
		}
		const diff = await runProcess(
			"git",
			["diff", "--name-only", "--no-renames", spec.baseRef, "--"],
			{ cwd: workspace, timeoutMs: 30_000 },
		);
		const files = diff.stdout
			.split("\n")
			.map(value => value.trim())
			.filter(Boolean);
		const unexpected = files.filter(
			file => !pathAllowed(file, spec.allowedPaths),
		);
		const overLimit =
			spec.maxChangedFiles !== undefined && files.length > spec.maxChangedFiles;
		const passed = diff.exitCode === 0 && unexpected.length === 0 && !overLimit;
		return {
			id: spec.id,
			type: spec.type,
			passed,
			durationMs: diff.durationMs,
			summary: passed
				? `${files.length} changed file(s) in scope`
				: "diff escaped allowed scope",
			evidence: unexpected.length
				? `unexpected: ${unexpected.join(", ")}`
				: files.join("\n"),
		};
	} catch (error) {
		return {
			id: spec.id,
			type: spec.type,
			passed: false,
			durationMs: Math.round(performance.now() - started),
			summary: error instanceof Error ? error.message : String(error),
		};
	}
}
