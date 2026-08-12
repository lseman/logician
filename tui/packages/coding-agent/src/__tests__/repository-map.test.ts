import { test } from "bun:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { RepositoryMap } from "../application/repository-map.ts";

function repository(): string {
	const cwd = mkdtempSync(path.join(tmpdir(), "logician-repo-map-"));
	execFileSync("git", ["init", "-q"], { cwd });
	return cwd;
}

void test("repository map extracts symbols and imports within its token budget", () => {
	const cwd = repository();
	writeFileSync(
		path.join(cwd, "auth.ts"),
		'import { db } from "./db";\nexport class AuthService {}\n',
	);
	writeFileSync(path.join(cwd, "db.ts"), "export function connect() {}\n");
	const rendered = new RepositoryMap(cwd, { maxTokens: 128 }).render(
		"auth service",
	);
	assert.match(rendered, /auth\.ts/);
	assert.match(rendered, /AuthService/);
	assert.match(rendered, /\.\/db/);
	assert.ok(rendered.length <= 512);
});

void test("repository map refreshes changed files and removes deleted symbols", async () => {
	const cwd = repository();
	const file = path.join(cwd, "worker.ts");
	writeFileSync(file, "export function oldWorker() {}\n");
	const map = new RepositoryMap(cwd);
	assert.match(map.render(), /oldWorker/);
	await new Promise(resolve => setTimeout(resolve, 5));
	writeFileSync(file, "export function newWorker() {}\n");
	const refreshed = map.render();
	assert.match(refreshed, /newWorker/);
	assert.doesNotMatch(refreshed, /oldWorker/);
});
