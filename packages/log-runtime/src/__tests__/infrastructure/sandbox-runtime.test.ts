import { test } from "bun:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
	getDefaultSandboxProfile,
	sandbox,
	setDefaultSandboxProfile,
} from "../../capabilities/tools/sandbox.ts";

const bwrapAvailable =
	process.platform === "linux" &&
	spawnSync("bwrap", ["--version"], { stdio: "ignore" }).status === 0;

const bwrapTest = bwrapAvailable ? test : test.skip;

void bwrapTest(
	"bubblewrap keeps its synthetic home bind source alive until exit",
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

void test("model arguments cannot weaken the user-controlled sandbox profile", async () => {
	const previous = getDefaultSandboxProfile();
	setDefaultSandboxProfile("code");
	try {
		const result = await sandbox.execute(
			{ command: "printf should-not-run-unsandboxed", profile: "none" },
			{ cwd: process.cwd() },
		);
		assert.equal(typeof result, "object");
		if (typeof result === "object") {
			assert.equal(result.details?.profile, "code");
			if (!bwrapAvailable) assert.equal(result.isError, true);
		}
	} finally {
		setDefaultSandboxProfile(previous);
	}
});
