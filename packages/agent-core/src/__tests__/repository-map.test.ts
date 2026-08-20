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
	const map = new RepositoryMap(cwd, { maxTokens: 128 });
	const rendered = map.render("fix auth service");
	assert.match(rendered, /auth\.ts/);
	assert.match(rendered, /AuthService/);
	assert.match(rendered, /\.\/db/);
	assert.ok(rendered.length <= 512);
	assert.equal(map.render("hi"), "");
	assert.equal(map.render("tell me something interesting"), "");
});

void test("repository map refreshes changed files and removes deleted symbols", async () => {
	const cwd = repository();
	const file = path.join(cwd, "worker.ts");
	writeFileSync(file, "export function oldWorker() {}\n");
	const map = new RepositoryMap(cwd);
	assert.match(map.render("worker"), /oldWorker/);
	await new Promise(resolve => setTimeout(resolve, 5));
	writeFileSync(file, "export function newWorker() {}\n");
	const refreshed = map.render("worker");
	assert.match(refreshed, /newWorker/);
	assert.doesNotMatch(refreshed, /oldWorker/);
});

void test("repository map requires meaningful matches for repository intent", () => {
	const cwd = repository();
	writeFileSync(
		path.join(cwd, "billing.ts"),
		"export class InvoiceLedger {}\n",
	);
	const map = new RepositoryMap(cwd);
	assert.equal(map.render("fix authentication retries"), "");
	assert.match(map.render("fix invoice ledger"), /InvoiceLedger/);
	assert.match(map.render("InvoiceLedger"), /billing\.ts/);
});
