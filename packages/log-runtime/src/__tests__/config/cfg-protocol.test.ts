import { afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { RuntimeSettingsPatch } from "../../agent/types.ts";
import type { ConfigProvenance } from "../../config/config-provenance.ts";
import { saveConfigPath } from "../../config/config-store.ts";
import {
	type CfgApproval,
	type CfgChangeRequest,
	type CfgHost,
	CfgProtocolHandler,
} from "../../resources/cfg-protocol.ts";
import { parseInternalUrl } from "../../resources/parse.ts";

function fakeHost(options: {
	config?: Record<string, unknown>;
	provenance?: ConfigProvenance;
	answers?: CfgApproval[];
}): CfgHost & {
	patches: RuntimeSettingsPatch[];
	saved: Array<[string, unknown]>;
	requests: CfgChangeRequest[];
} {
	const answers = [...(options.answers ?? [])];
	const host = {
		patches: [] as RuntimeSettingsPatch[],
		saved: [] as Array<[string, unknown]>,
		requests: [] as CfgChangeRequest[],
		resolve: () => ({
			config: options.config ?? {},
			provenance: options.provenance ?? [],
		}),
		applyLive: (patch: RuntimeSettingsPatch) => {
			host.patches.push(patch);
		},
		save: (path: string, value: unknown) => {
			host.saved.push([path, value]);
			return true;
		},
		approve: async (request: CfgChangeRequest) => {
			host.requests.push(request);
			return answers.shift() ?? "deny";
		},
	};
	return host;
}

const url = (href: string) => parseInternalUrl(href);

describe("cfg:// reads", () => {
	test("a leaf shows value, type, default, source, and liveness", async () => {
		const handler = new CfgProtocolHandler(
			fakeHost({
				config: { temperature: 0.2 },
				provenance: [{ key: "temperature", value: 0.2, layer: "project" }],
			}),
		);
		const { content } = await handler.resolve(url("cfg://temperature"));
		expect(content).toContain("temperature: 0.2");
		expect(content).toContain("type: number (>=0, <=2)");
		expect(content).toContain("default: 0.5");
		expect(content).toContain("source: project config");
		expect(content).toContain("applies: live");
	});

	test("a namespace renders a tree; dots and slashes are interchangeable", async () => {
		const handler = new CfgProtocolHandler(
			fakeHost({ config: { ttsr: { judge: false } } }),
		);
		const bySlash = await handler.resolve(url("cfg://ttsr"));
		const byDot = await handler.resolve(url("cfg://TTSR"));
		expect(bySlash.content).toBe(byDot.content);
		expect(bySlash.content).toContain("judge: false");
		expect(bySlash.content).toContain(
			"interruptMode: unset  # always|prose-only|tool-only|never",
		);
	});

	test("unknown settings suggest similar keys", async () => {
		const handler = new CfgProtocolHandler(fakeHost({}));
		await expect(handler.resolve(url("cfg://ttsr/judg"))).rejects.toThrow(
			/Unknown setting: ttsr\.judg[\s\S]*Similar: .*ttsr\.judge/,
		);
	});

	test("completion lists settings as slash paths", async () => {
		const handler = new CfgProtocolHandler(fakeHost({}));
		const values = (await handler.complete("ttsr/")).map(c => c.value);
		expect(values).toContain("ttsr/judge");
		expect(values.every(value => value.startsWith("ttsr/"))).toBe(true);
	});
});

describe("cfg:// writes", () => {
	test("an approved session write applies live and reads back as a session override", async () => {
		const host = fakeHost({ config: { temperature: 0.5 }, answers: ["once"] });
		const handler = new CfgProtocolHandler(host);
		const result = await handler.write(url("cfg://temperature"), "0.1");
		expect(result).toContain(
			"Set temperature = 0.1 for this session (was 0.5)",
		);
		expect(host.patches).toEqual([{ temperature: 0.1 }]);
		expect(host.saved).toEqual([]);
		const { content } = await handler.resolve(url("cfg://temperature"));
		expect(content).toContain("source: session override");
	});

	test("a denied write changes nothing", async () => {
		const host = fakeHost({ answers: ["deny"] });
		const handler = new CfgProtocolHandler(host);
		const result = await handler.write(url("cfg://maxIterations"), "12");
		expect(result).toContain("declined");
		expect(host.patches).toEqual([]);
	});

	test("values are validated like a config file before anyone is asked", async () => {
		const host = fakeHost({ answers: ["once"] });
		const handler = new CfgProtocolHandler(host);
		await expect(handler.write(url("cfg://temperature"), "7")).rejects.toThrow(
			/temperature/,
		);
		await expect(
			handler.write(url("cfg://ttsr/interruptMode/save"), "sometimes"),
		).rejects.toThrow(/must be one of: always, prose-only, tool-only, never/);
		await expect(
			handler.write(url("cfg://guardsEnabled"), "maybe"),
		).rejects.toThrow(/boolean/);
		expect(host.requests).toEqual([]);
	});

	test("startup-only settings refuse session writes and point at /save", async () => {
		const handler = new CfgProtocolHandler(fakeHost({ answers: ["once"] }));
		await expect(
			handler.write(url("cfg://ttsr/judge"), "false"),
		).rejects.toThrow(/read at startup[\s\S]*cfg:\/\/ttsr\/judge\/save/);
	});

	test("/save persists, names a shadowing layer, and applies live when possible", async () => {
		const host = fakeHost({
			provenance: [{ key: "ttsr.judge", value: true, layer: "project" }],
			answers: ["once", "once"],
		});
		const handler = new CfgProtocolHandler(host);
		const saved = await handler.write(url("cfg://ttsr/judge/save"), "false");
		expect(host.saved).toEqual([["ttsr.judge", false]]);
		expect(host.requests[0]?.shadowedBy).toContain("project config");
		expect(saved).toContain("takes effect in the next session");
		expect(saved).toContain("wins over the saved global value");

		await handler.write(url("cfg://compaction/enabled/save"), "true");
		expect(host.patches).toEqual([{ proactiveCompactionEnabled: true }]);
	});

	test("a session grant skips later prompts; a session-only grant doesn't cover saves", async () => {
		const host = fakeHost({ answers: ["session", "once"] });
		const handler = new CfgProtocolHandler(host);
		await handler.write(url("cfg://temperature"), "0.3");
		await handler.write(url("cfg://maxTokens"), "2048");
		expect(host.requests).toHaveLength(1);
		await handler.write(url("cfg://maxTokens/save"), "4096");
		expect(host.requests).toHaveLength(2);
	});

	test("namespaces, model lists, and credentials are not writable", async () => {
		const handler = new CfgProtocolHandler(fakeHost({ answers: ["once"] }));
		await expect(handler.write(url("cfg://ttsr"), "{}")).rejects.toThrow(
			/namespace/,
		);
		await expect(handler.write(url("cfg://models/save"), "[]")).rejects.toThrow(
			/model definitions/,
		);
	});

	test("token-count settings are not mistaken for credentials", async () => {
		const host = fakeHost({ config: { maxTokens: 1000 }, answers: ["once"] });
		const handler = new CfgProtocolHandler(host);
		const { content } = await handler.resolve(url("cfg://maxTokens"));
		expect(content).toContain("maxTokens: 1000");
		await handler.write(url("cfg://maxTokens"), "2048");
		expect(host.patches).toEqual([{ maxTokens: 2048 }]);
	});
});

describe("saveConfigPath", () => {
	const previousHome = process.env.HOME;
	const previousPersist = process.env.LOGICIAN_PERSIST_CONFIG;
	let home: string | undefined;

	afterEach(() => {
		process.env.HOME = previousHome;
		if (previousPersist === undefined)
			delete process.env.LOGICIAN_PERSIST_CONFIG;
		else process.env.LOGICIAN_PERSIST_CONFIG = previousPersist;
		if (home) rmSync(home, { recursive: true, force: true });
	});

	test("sets nested paths and prunes emptied parents", () => {
		home = mkdtempSync(join(tmpdir(), "cfg-save-"));
		process.env.HOME = home;
		process.env.LOGICIAN_PERSIST_CONFIG = "1";
		const file = join(home, ".logician", "settings.json");

		expect(saveConfigPath("ttsr.judge", false)).toBe(true);
		expect(JSON.parse(readFileSync(file, "utf8"))).toEqual({
			ttsr: { judge: false },
		});

		writeFileSync(file, JSON.stringify({ model: "m", ttsr: { judge: false } }));
		expect(saveConfigPath("ttsr.judge", undefined)).toBe(true);
		expect(JSON.parse(readFileSync(file, "utf8"))).toEqual({ model: "m" });
	});
});

describe("ttsr config layering", () => {
	test("project ttsr keys merge over global ones instead of replacing them", async () => {
		const { mergeRuntimeConfigLayers } = await import(
			"../../config/runtime-config.ts"
		);
		const merged = mergeRuntimeConfigLayers(
			{ ttsr: { judge: false, builtinRules: true } },
			{ ttsr: { builtinRules: false } },
		);
		expect(merged.ttsr).toEqual({ judge: false, builtinRules: false });
	});
});
