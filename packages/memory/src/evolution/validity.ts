import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync, realpathSync } from "node:fs";
import { isAbsolute, relative, resolve } from "node:path";
import type { ClaimValidityPredicate } from "../types.js";

export function sha256(value: string | Uint8Array): string {
	return createHash("sha256").update(value).digest("hex");
}

function workspaceFile(
	workspace: string,
	requestedPath: string,
): string | null {
	const root = realpathSync(workspace);
	const candidate = resolve(root, requestedPath);
	if (!existsSync(candidate)) return null;
	const actual = realpathSync(candidate);
	const rel = relative(root, actual);
	if (rel.startsWith("..") || isAbsolute(rel)) return null;
	return actual;
}

function configValue(value: unknown, key: string): unknown {
	return key
		.split(".")
		.filter(Boolean)
		.reduce<unknown>((current, part) => {
			if (!current || typeof current !== "object" || Array.isArray(current))
				return undefined;
			return (current as Record<string, unknown>)[part];
		}, value);
}

/** Execute a bounded, side-effect-free validity assertion in the claim workspace. */
export function evaluateValidityPredicate(
	workspace: string,
	predicate: ClaimValidityPredicate,
): boolean {
	try {
		if (predicate.type === "git_revision") {
			const revision = execFileSync("git", ["rev-parse", "HEAD"], {
				cwd: workspace,
				encoding: "utf8",
				stdio: ["ignore", "pipe", "ignore"],
				timeout: 2_000,
			}).trim();
			return revision === predicate.revision;
		}
		const path = workspaceFile(workspace, predicate.path);
		if (!path) return false;
		if (predicate.type === "file_hash")
			return sha256(readFileSync(path)) === predicate.sha256;
		const parsed = JSON.parse(readFileSync(path, "utf8")) as unknown;
		return (
			sha256(JSON.stringify(configValue(parsed, predicate.key))) ===
			predicate.sha256
		);
	} catch {
		return false;
	}
}

export function predicatesAreValid(
	workspace: string,
	predicates: ClaimValidityPredicate[],
): boolean {
	return predicates.every(predicate =>
		evaluateValidityPredicate(workspace, predicate),
	);
}
