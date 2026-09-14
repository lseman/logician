import { afterEach, expect, test } from "bun:test";
import { execFile } from "node:child_process";
import {
	mkdirSync,
	mkdtempSync,
	readFileSync,
	rmSync,
	symlinkSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";
import type { Tool, ToolResult } from "@logician/log-core";
import { ToolRegistry } from "@logician/log-core/runtime";
import { createDefaultTools } from "../../capabilities/tools/default-tools.ts";
import { createReadTool } from "../../capabilities/tools/read-file.ts";
import { hasBeenRead } from "../../capabilities/tools/support/read-tracker.ts";
import {
	DEFAULT_MAX_BYTES,
	DEFAULT_MAX_LINES,
} from "../../capabilities/tools/support/utils/truncate.ts";
import { XdDeviceRegistry } from "../../capabilities/tools/support/xd-device-registry.ts";
import { createWriteTool } from "../../capabilities/tools/write-file.ts";
import { LocalProtocolHandler } from "../../runtime/bridge/support/internal-urls/local-protocol.ts";
import { ArtifactRegistry } from "../../runtime/bridge/support/internal-urls/artifact-manager.ts";
import { LogProtocolHandler } from "../../runtime/bridge/support/internal-urls/log-protocol.ts";
import { InternalUrlRouter } from "../../runtime/bridge/support/internal-urls/router.ts";
import { SkillProtocolHandler } from "../../runtime/bridge/support/internal-urls/skill-protocol.ts";
import { ToolRouter } from "../../runtime/bridge/support/tool-router.ts";

const dirs: string[] = [];
afterEach(() => {
	for (const dir of dirs.splice(0))
		rmSync(dir, { recursive: true, force: true });
});
function temp(): string {
	const dir = mkdtempSync(path.join(tmpdir(), "resource-tools-"));
	dirs.push(dir);
	return dir;
}
function result(value: string | ToolResult): ToolResult {
	return typeof value === "string" ? { content: value } : value;
}
function call(name: string, args: Record<string, unknown>) {
	return { id: "call-1", name, arguments: JSON.stringify(args) };
}
function deviceCall(
	content = '{"text":"hello"}',
	url = "xd://example",
	extra = {},
) {
	return call("write", { path: url, content, ...extra });
}
function device(overrides: Partial<Tool> = {}): Tool {
	return {
		name: "example",
		description: "Example device",
		parameters: {
			type: "object",
			properties: { text: { type: "string" } },
			required: ["text"],
		},
		execute: async args => String(args.text),
		...overrides,
	};
}
function setup(target = device()) {
	const devices = new XdDeviceRegistry();
	devices.mount(target);
	const urls = new InternalUrlRouter();
	urls.register(devices);
	const read = createReadTool(urls);
	const registry = new ToolRegistry({ cwd: temp() });
	registry.registerMany([read, createWriteTool(devices), target]);
	return { devices, urls, read, registry };
}

test("file and resource pagination share numbering, but only files have edit anchors", async () => {
	const cwd = temp();
	writeFileSync(path.join(cwd, "a.txt"), "one\ntwo\nthree\nfour\n");
	const urls = new InternalUrlRouter();
	urls.register({
		scheme: "sample",
		immutable: true,
		resolve: async url => ({
			url: url.href,
			content: "one\ntwo\nthree\nfour\n",
			sourcePath: path.join(cwd, "virtual.txt"),
		}),
	});
	const read = createReadTool(urls);
	const file = result(
		await read.execute({ path: "a.txt", offset: 2, limit: 2 }, { cwd }),
	);
	const virtual = result(
		await read.execute(
			{ path: "SaMpLe://Mixed:Name/nested?x#y", offset: 2, limit: 2 },
			{ cwd },
		),
	);
	expect(file.content).toMatch(/^\[a.txt#[a-f0-9]{4}\]\n2:two\n3:three/);
	expect(virtual.content).toStartWith(
		"[sample://Mixed:Name/nested?x#y]\n2:two\n3:three",
	);
	for (const output of [file, virtual])
		expect(output.content).toContain("Use offset=4 to continue");
	expect(hasBeenRead(path.join(cwd, "a.txt"))).toBe(true);
	expect(hasBeenRead(path.join(cwd, "virtual.txt"))).toBe(false);
});

test("empty files, empty directories, directory pages and invalid ranges are explicit", async () => {
	const cwd = temp();
	mkdirSync(path.join(cwd, "empty"));
	writeFileSync(path.join(cwd, "blank"), "");
	const read = createReadTool();
	expect(
		result(await read.execute({ path: "blank" }, { cwd })).content,
	).toContain("Empty resource");
	expect(
		result(await read.execute({ path: "empty" }, { cwd })).content,
	).toContain("Empty directory");
	expect(
		result(await read.execute({ path: ".", offset: 2, limit: 1 }, { cwd }))
			.content,
	).toContain("2:empty/");
	for (const invalid of [0, -1, 1.5, "2", NaN, Infinity]) {
		expect(
			result(await read.execute({ path: "blank", offset: invalid }, { cwd }))
				.isError,
		).toBe(true);
	}
	expect(
		result(await read.execute({ path: "blank", offset: 2 }, { cwd })).isError,
	).toBe(true);
});

test("URL resources obey line and byte caps with usable continuation offsets", async () => {
	const urls = new InternalUrlRouter();
	urls.register({
		scheme: "many",
		immutable: true,
		resolve: async url => ({
			url: url.href,
			content: Array.from(
				{ length: DEFAULT_MAX_LINES + 2 },
				(_, i) => `line${i + 1}`,
			).join("\n"),
		}),
	});
	urls.register({
		scheme: "bytes",
		immutable: true,
		resolve: async url => ({
			url: url.href,
			content: Array.from({ length: 10 }, () =>
				"é".repeat(DEFAULT_MAX_BYTES / 8),
			).join("\n"),
		}),
	});
	const read = createReadTool(urls);
	const page = result(await read.execute({ path: "many://" }, {}));
	expect(page.content).toContain(`Use offset=${DEFAULT_MAX_LINES + 1}`);
	expect(
		result(
			await read.execute(
				{ path: "many://", offset: DEFAULT_MAX_LINES + 1 },
				{},
			),
		).content,
	).toContain(`${DEFAULT_MAX_LINES + 2}:line${DEFAULT_MAX_LINES + 2}`);
	const bytes = result(await read.execute({ path: "bytes://" }, {}));
	expect(bytes.content).toContain("Use offset=4");
	expect(Buffer.byteLength(bytes.content)).toBeLessThan(
		DEFAULT_MAX_BYTES + 250,
	);
});

test("unsupported URLs never fall through to disk and literal URL-shaped paths remain readable", async () => {
	const cwd = temp();
	mkdirSync(path.join(cwd, "unknown:"));
	writeFileSync(path.join(cwd, "unknown:", "file"), "literal");
	const read = createReadTool(new InternalUrlRouter());
	expect(
		result(await read.execute({ path: "unknown://file" }, { cwd })).isError,
	).toBe(true);
	expect(
		result(await read.execute({ path: "./unknown://file" }, { cwd })).content,
	).toContain("1:literal");
	expect(
		result(await read.execute({ path: "../outside" }, { cwd })).isError,
	).toBe(true);
});

test("skill URLs preserve names and receive the caller's loaded resources", async () => {
	const urls = new InternalUrlRouter();
	urls.register(new SkillProtocolHandler());
	const read = createReadTool(urls);
	const value = result(
		await read.execute(
			{ path: "SKILL://Plugin:Name", offset: 2 },
			{
				skills: [
					{
						name: "Plugin:Name",
						path: "/elsewhere/SKILL.md",
						content: "heading\nbody",
					},
				],
			},
		),
	);
	expect(value.content).toBe("[skill://Plugin:Name]\n2:body");
});

test("device catalogs and docs update after mount/unmount without cached stale reads", async () => {
	const { devices, registry } = setup();
	const root = call("read", { path: "xd://" });
	expect((await registry.execute(root)).content).toContain("xd://example");
	expect(
		(await registry.execute(call("read", { path: "XD://example" }))).content,
	).toContain('"required"');
	devices.unmount("example");
	expect((await registry.execute(root)).content).not.toContain("xd://example");
	expect(
		(await registry.execute(call("read", { path: "xd://example" }))).isError,
	).toBe(true);
	devices.mount(device({ name: "second" }));
	expect((await registry.execute(root)).content).toContain("xd://second");
});

test("device dispatch uses target preparation, context, updates, and structured result", async () => {
	const updates: string[] = [];
	const { registry } = setup(
		device({
			prepareArguments: raw => ({
				text: String((raw as Record<string, unknown>).text).toUpperCase(),
			}),
			execute: async (args, ctx) => {
				expect(ctx.allowedPaths).toEqual(["/test-allowed"]);
				expect(ctx.signal).toBeDefined();
				ctx.onUpdate?.("progress");
				return {
					content: String(args.text),
					details: { cwd: ctx.cwd },
					isError: true,
					terminate: true,
				};
			},
		}),
	);
	const output = await registry.execute(deviceCall(), {
		cwd: "/test-cwd",
		allowedPaths: ["/test-allowed"],
		onUpdate: update => {
			updates.push(update);
		},
	});
	expect(output).toEqual({
		content: "HELLO",
		details: { cwd: "/test-cwd" },
		isError: true,
		terminate: true,
	});
	expect(updates).toEqual(["progress"]);
});

test("malformed JSON, non-object payloads, unknown devices and append never execute", async () => {
	let runs = 0;
	const { registry } = setup(
		device({
			execute: async () => {
				runs++;
				return "ran";
			},
		}),
	);
	for (const content of ["{", "null", "[]", '"hello"', "42", "true"]) {
		expect((await registry.execute(deviceCall(content))).isError).toBe(true);
	}
	expect(
		(await registry.execute(deviceCall("{}", "xd://missing"))).isError,
	).toBe(true);
	expect(
		(await registry.execute(deviceCall("{}", "xd://example", { append: true })))
			.isError,
	).toBe(true);
	expect(runs).toBe(0);
});

test("device calls use the target timeout", async () => {
	const { registry } = setup(
		device({ timeoutMs: 5, execute: async () => new Promise(() => {}) }),
	);
	expect((await registry.execute(deviceCall())).content).toContain("timed out");
});

test("cancellation reaches resource handlers and cancelled reads fail", async () => {
	const controller = new AbortController();
	const urls = new InternalUrlRouter();
	let calls = 0;
	urls.register({
		scheme: "cancel",
		immutable: true,
		resolve: async (url, ctx) => {
			calls++;
			expect(ctx?.signal).toBe(controller.signal);
			controller.abort();
			return { url: url.href, content: "not delivered" };
		},
	});
	const read = createReadTool(urls);
	expect(
		result(
			await read.execute({ path: "cancel://" }, { signal: controller.signal }),
		).isError,
	).toBe(true);
	expect(
		result(
			await read.execute({ path: "cancel://" }, { signal: controller.signal }),
		).isError,
	).toBe(true);
	expect(calls).toBe(1);
});

test("session device registries are isolated, honor xdev, and follow capability toggles", async () => {
	function router(enabled: boolean) {
		return new ToolRouter({
			cwd: temp(),
			sessionId: "resources",
			projectTrusted: true,
			autoStartMcp: false,
			xdevEnabled: enabled,
			emit: () => {},
			onContextChanged: () => {},
			onToolAdded: () => {},
		});
	}
	const a = router(true);
	const b = router(false);
	const readA = a.getDefaultTools().find(tool => tool.name === "read");
	const readB = b.getDefaultTools().find(tool => tool.name === "read");
	if (!readA || !readB) throw new Error("Missing read tool");
	expect(result(await readA.execute({ path: "xd://" }, {})).content).toContain(
		"xd://git",
	);
	expect(result(await readB.execute({ path: "xd://" }, {})).content).toContain(
		"No devices",
	);
	a.setGraphicianEnabled(false);
	a.setGraphicianEnabled(true);
	// graphician is a core tool but not xd://discoverable (clean native interface)
	expect(
		result(await readA.execute({ path: "xd://graphician" }, {})).isError,
	).toBe(true);
	expect(
		result(await readB.execute({ path: "xd://graphician" }, {})).isError,
	).toBe(true);
});

test("standalone default tools also have a working device catalog", async () => {
	const registry = new ToolRegistry();
	registry.registerMany(createDefaultTools());
	expect(
		(await registry.execute(call("read", { path: "xd://" }))).content,
	).toContain("xd://git");
	expect(
		registry.prepare(deviceCall('{"command":"status"}', "xd://git")).call.name,
	).toBe("git");
});

async function hasAstGrepCli(): Promise<boolean> {
	for (const bin of ["sg", "ast-grep"]) {
		try {
			await promisify(execFile)(bin, ["--version"]);
			return true;
		} catch {}
	}
	return false;
}
const astGrepTest = (await hasAstGrepCli()) ? test : test.skip;

astGrepTest(
	"ast_edit's staged preview is applied to disk via xd://resolve",
	async () => {
		const cwd = temp();
		const filePath = path.join(cwd, "sample.ts");
		writeFileSync(filePath, 'console.log("hi");\n');
		const registry = new ToolRegistry({ cwd });
		registry.registerMany(createDefaultTools());

		const staged = await registry.execute(
			call("ast_edit", {
				ops: [{ pat: "console.log($X)", out: "logger.info($X)" }],
				paths: ["sample.ts"],
			}),
		);
		expect(staged.isError).toBeFalsy();
		expect(readFileSync(filePath, "utf-8")).toBe('console.log("hi");\n');

		const resolved = await registry.execute(
			call("write", {
				path: "xd://resolve",
				content: '{"reason":"looks right"}',
			}),
		);
		expect(resolved.isError).toBeFalsy();
		const applied = readFileSync(filePath, "utf-8");
		expect(applied).toContain('logger.info("hi")');
		expect(applied).not.toContain("console.log");
	},
);

astGrepTest(
	"xd://reject discards ast_edit's staged preview without touching disk",
	async () => {
		const cwd = temp();
		const filePath = path.join(cwd, "sample.ts");
		writeFileSync(filePath, 'console.log("hi");\n');
		const registry = new ToolRegistry({ cwd });
		registry.registerMany(createDefaultTools());

		await registry.execute(
			call("ast_edit", {
				ops: [{ pat: "console.log($X)", out: "logger.info($X)" }],
				paths: ["sample.ts"],
			}),
		);
		const rejected = await registry.execute(
			call("write", {
				path: "xd://reject",
				content: '{"reason":"not needed"}',
			}),
		);
		expect(rejected.isError).toBeFalsy();
		expect(readFileSync(filePath, "utf-8")).toBe('console.log("hi");\n');
	},
);

test("local and documentation URLs cannot escape their roots through siblings or symlinks", async () => {
	const cwd = temp();
	const outside = temp();
	writeFileSync(path.join(outside, "secret"), "private");
	const urls = new InternalUrlRouter();
	urls.register(new LocalProtocolHandler());
	urls.register(new LogProtocolHandler());
	const read = createReadTool(urls);
	for (const [scheme, directory] of [
		["local", ".logician/artifacts"],
		["log", "docs"],
	] as const) {
		const root = path.join(cwd, directory);
		mkdirSync(root, { recursive: true });
		mkdirSync(`${root}-other`);
		writeFileSync(path.join(`${root}-other`, "secret"), "private");
		symlinkSync(outside, path.join(root, "escape"));
		for (const url of [
			`${scheme}://../${path.basename(root)}-other/secret`,
			`${scheme}://escape/secret`,
		]) {
			expect(result(await read.execute({ path: url }, { cwd })).isError).toBe(
				true,
			);
		}
		writeFileSync(path.join(root, "ok"), "visible");
		expect(
			result(await read.execute({ path: `${scheme}://ok` }, { cwd })).content,
		).toContain("1:visible");
	}
});

test("a catalog entry cannot execute a target absent from the execution registry", async () => {
	const { registry } = setup();
	registry.unregister("example");
	expect((await registry.execute(deviceCall())).isError).toBe(true);
});

test("independent protocol routers do not replace each other's handlers", async () => {
	const a = new InternalUrlRouter();
	const b = new InternalUrlRouter();
	a.register({
		scheme: "test",
		immutable: true,
		resolve: async url => ({ url: url.href, content: "A" }),
	});
	b.register({
		scheme: "test",
		immutable: true,
		resolve: async url => ({ url: url.href, content: "B" }),
	});
	expect(
		result(await createReadTool(a).execute({ path: "test://" }, {})).content,
	).toBe("[test://]\n1:A");
	expect(
		result(await createReadTool(b).execute({ path: "test://" }, {})).content,
	).toBe("[test://]\n1:B");
});

test("read and write are the only registered names and preserve file IO", async () => {
	const registry = new ToolRegistry({ cwd: temp() });
	registry.registerMany(createDefaultTools({ xdevEnabled: false }));
	expect(registry.has("read")).toBe(true);
	expect(registry.has("write")).toBe(true);
	expect(registry.has("read_file")).toBe(false);
	expect(registry.has("write_file")).toBe(false);
	const created = await registry.execute(
		call("write", { path: "example.txt", content: "first" }),
	);
	expect(created.content).toContain("Created");
	expect(
		(await registry.execute(call("read", { path: "example.txt" }))).content,
	).toContain("1:first");
	await registry.execute(
		call("write", { path: "example.txt", content: "second" }),
	);
	expect(
		(await registry.execute(call("read", { path: "example.txt" }))).content,
	).toContain("1:second");
});

// ── Router: immutable stamping, write dispatch, completion ────────────────────

test("resolve() stamps the handler's immutable default, but a resource's own value wins", async () => {
	const urls = new InternalUrlRouter();
	urls.register({
		scheme: "ro",
		immutable: true,
		resolve: async url => ({ url: url.href, content: "x" }),
	});
	urls.register({
		scheme: "rw-but-flagged",
		immutable: false,
		resolve: async url => ({ url: url.href, content: "x", immutable: true }),
	});
	expect((await urls.resolve("ro://a")).immutable).toBe(true);
	expect((await urls.resolve("rw-but-flagged://a")).immutable).toBe(true);
});

test("write() rejects unknown schemes and schemes without a write handler, and dispatches to ones that have it", async () => {
	const urls = new InternalUrlRouter();
	const written: Array<{ url: string; content: string }> = [];
	urls.register({
		scheme: "ro",
		immutable: true,
		resolve: async url => ({ url: url.href, content: "x" }),
	});
	urls.register({
		scheme: "rw",
		immutable: false,
		resolve: async url => ({ url: url.href, content: "x" }),
		write: async (url, content) => {
			written.push({ url: url.href, content });
		},
	});
	await expect(urls.write("missing://a", "x")).rejects.toThrow(
		/Unsupported internal URL scheme/,
	);
	await expect(urls.write("ro://a", "x")).rejects.toThrow(
		/ro:\/\/ is read-only for write/,
	);
	await urls.write("rw://a", "hello");
	expect(written).toEqual([{ url: "rw://a", content: "hello" }]);
});

test("completionSchemes() only lists handlers with complete(), and complete() returns null for the rest", async () => {
	const urls = new InternalUrlRouter();
	urls.register({
		scheme: "no-complete",
		immutable: true,
		resolve: async url => ({ url: url.href, content: "x" }),
	});
	urls.register({
		scheme: "has-complete",
		immutable: true,
		resolve: async url => ({ url: url.href, content: "x" }),
		complete: async query => [{ value: `${query}-match` }],
	});
	expect(urls.completionSchemes()).toEqual(["has-complete"]);
	expect(await urls.complete("no-complete", "q")).toBeNull();
	expect(await urls.complete("has-complete", "q")).toEqual([
		{ value: "q-match" },
	]);
});

// ── local:// writes ─────────────────────────────────────────────────────────

test("local:// round-trips a write through read, rejects append, and requires a target", async () => {
	const cwd = temp();
	const urls = new InternalUrlRouter();
	urls.register(new LocalProtocolHandler());
	const readTool = createReadTool(urls);
	const writeTool = createWriteTool(undefined, urls);

	const written = result(
		await writeTool.execute(
			{ path: "local://foo", content: "hello world" },
			{ cwd },
		),
	);
	expect(written.content).toContain("Wrote local://foo");
	expect(
		result(await readTool.execute({ path: "local://foo" }, { cwd })).content,
	).toContain("1:hello world");

	expect(
		result(
			await writeTool.execute(
				{ path: "local://foo", content: "x", append: true },
				{ cwd },
			),
		).content,
	).toContain("append is not supported");

	expect(
		result(await writeTool.execute({ path: "local://", content: "x" }, { cwd }))
			.content,
	).toContain("Error");
});

test("local:// write rejects path traversal and writing over an existing directory", async () => {
	const cwd = temp();
	const urls = new InternalUrlRouter();
	urls.register(new LocalProtocolHandler());
	const writeTool = createWriteTool(undefined, urls);

	expect(
		result(
			await writeTool.execute(
				{ path: "local://../escape", content: "x" },
				{ cwd },
			),
		).content,
	).toContain("Error");

	mkdirSync(path.join(cwd, ".logician", "artifacts", "adir"), {
		recursive: true,
	});
	expect(
		result(
			await writeTool.execute({ path: "local://adir", content: "x" }, { cwd }),
		).content,
	).toContain("Error");
});

test("local:// read rejects a binary file and an oversized text file instead of corrupting/blowing up output", async () => {
	const cwd = temp();
	const urls = new InternalUrlRouter();
	urls.register(new LocalProtocolHandler());
	const readTool = createReadTool(urls);
	const artifactDir = path.join(cwd, ".logician", "artifacts");
	mkdirSync(artifactDir, { recursive: true });

	// Binary content (embedded NUL byte) should be rejected, not read as text.
	writeFileSync(
		path.join(artifactDir, "binary.bin"),
		Buffer.from([0x00, 0x01, 0x02, 0xff, 0xfe]),
	);
	const binaryResult = result(
		await readTool.execute({ path: "local://binary.bin" }, { cwd }),
	);
	expect(binaryResult.isError).toBe(true);
	expect(binaryResult.content).toContain("binary");

	// A file over the 1 MiB text-resource cap should be rejected with a clear
	// message rather than materialized into the model's context wholesale.
	writeFileSync(
		path.join(artifactDir, "huge.txt"),
		"a".repeat(1024 * 1024 + 1),
	);
	const hugeResult = result(
		await readTool.execute({ path: "local://huge.txt" }, { cwd }),
	);
	expect(hugeResult.isError).toBe(true);
	expect(hugeResult.content).toContain("exceeding");
});

test("write to an immutable scheme surfaces the read-only-for-write message", async () => {
	const urls = new InternalUrlRouter();
	urls.register(new SkillProtocolHandler());
	const writeTool = createWriteTool(undefined, urls);
	const output = result(
		await writeTool.execute(
			{ path: "skill://name", content: "x" },
			{
				skills: [{ name: "name", path: "/elsewhere/SKILL.md", content: "y" }],
			},
		),
	);
	expect(output.isError).not.toBe(false);
	expect(output.content).toContain("read-only for write");
});

// ── local:// artifact pathOnly ─────────────────────────────────────────────

test("local:// artifact pathOnly resolves sourcePath without reading full content", async () => {
	const cwd = temp();
	ArtifactRegistry.resetForTests();
	ArtifactRegistry.instance().init({ cwd, sessionId: "path-only" });
	const id = await ArtifactRegistry.instance().save(
		"full artifact body",
		"tool",
	);
	if (id === null) throw new Error("artifact save failed");

	const urls = new InternalUrlRouter();
	urls.register(new LocalProtocolHandler());
	const withContent = await urls.resolve(`local://${id}`);
	expect(withContent.content).toBe("full artifact body");

	const pathOnly = await urls.resolve(`local://${id}`, { pathOnly: true });
	expect(pathOnly.content).toBe("");
	expect(pathOnly.sourcePath).toBeDefined();
});
