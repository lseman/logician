import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { test } from "node:test";
import { sandbox } from "../tools/sandbox.ts";

const bwrapAvailable =
	process.platform === "linux" &&
	spawnSync("bwrap", ["--version"], { stdio: "ignore" }).status === 0;

void test(
	"bubblewrap keeps its synthetic home bind source alive until exit",
	{ skip: !bwrapAvailable },
	async () => {
		const result = await sandbox.execute(
			{
				command: "test -d /home && printf sandbox-home-ok",
				profile: "code",
				timeout: 5,
			},
			{ cwd: process.cwd() },
		);

		assert.equal(typeof result, "object");
		if (typeof result === "object") {
			assert.equal(result.content, "sandbox-home-ok");
			assert.equal(result.details?.bwrapAvailable, true);
		}
	},
);
