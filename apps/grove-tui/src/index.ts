#!/usr/bin/env node
import { GroveApp } from "./app.ts";
import { LogicianGroveRepository } from "./repository.ts";

const app = new GroveApp(new LogicianGroveRepository(), process.cwd());
try {
	app.start();
} catch (error: unknown) {
	const message = error instanceof Error ? error.message : String(error);
	process.stderr.write(`${message}\n`);
	process.exitCode = 1;
}

process.once("SIGTERM", () => app.stop());
