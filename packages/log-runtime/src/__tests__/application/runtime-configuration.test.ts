import { describe, expect, test } from "bun:test";
import type { AgentConfig } from "@logician/log-core";
import type { RuntimeEvent } from "@logician/log-core/events";
import {
	RuntimeConfiguration,
	type RuntimeSettingsMemory,
	type RuntimeSettingsSession,
	type RuntimeSettingsTools,
} from "../../runtime/bridge/application/runtime-configuration.ts";

class FakeSession implements RuntimeSettingsSession {
	patches: Partial<AgentConfig>[] = [];
	thinkingLevels: string[] = [];
	autoCompaction: boolean[] = [];
	models = {
		setThinkingLevel: (level: NonNullable<AgentConfig["thinkingLevel"]>) => {
			this.thinkingLevels.push(level);
		},
	};

	configure(patch: Partial<AgentConfig>): void {
		this.patches.push(patch);
	}

	enableAutoCompaction(enabled: boolean): void {
		this.autoCompaction.push(enabled);
	}
}

class FakeTools implements RuntimeSettingsTools {
	graphician: boolean[] = [];
	fffgrep: boolean[] = [];
	setGraphicianEnabled(enabled: boolean): void {
		this.graphician.push(enabled);
	}
	setFffgrepEnabled(enabled: boolean): void {
		this.fffgrep.push(enabled);
	}
}

class FakeMemory implements RuntimeSettingsMemory {
	enabled: Array<{ enabled: boolean; sessionId: string }> = [];
	getStore(): unknown {
		return {};
	}
	setEnabled(enabled: boolean, sessionId: string): void {
		this.enabled.push({ enabled, sessionId });
	}
}

function createRuntimeConfiguration() {
	const config = {
		baseUrl: "http://localhost",
		model: "test",
		tools: [{ name: "base" }],
		cwd: "/tmp",
		maxIterations: 3,
	} as AgentConfig;
	const session = new FakeSession();
	const tools = new FakeTools();
	const memory = new FakeMemory();
	const backendLevels: string[] = [];
	const events: RuntimeEvent[] = [];
	let legroomCloses = 0;
	let legroomEnabled = true;
	const runtime = new RuntimeConfiguration({
		config,
		backend: {
			setDefaultThinkingLevel: level => backendLevels.push(level),
		},
		session: () => session,
		sessionId: () => "session-1",
		tools,
		interactions: { mode: "acceptEdits" },
		memory,
		legroom: {
			isEnabled: () => legroomEnabled,
			setEnabled: enabled => {
				legroomEnabled = enabled;
				if (!enabled) legroomCloses++;
			},
		},
		memoriam: {
			isEnabled: () => true,
			setEnabled: () => {},
		},
		defaultTools: () => config.tools,
		setReasoner: () => {},
		emit: event => events.push(event),
		postEditDiagnostics: true,
	});
	return {
		runtime,
		config,
		session,
		tools,
		memory,
		backendLevels,
		events,
		legroomCloses: () => legroomCloses,
	};
}

describe("RuntimeConfiguration", () => {
	test("propagates model settings to config, session, and backend", () => {
		const state = createRuntimeConfiguration();
		state.runtime.update({ thinkingLevel: "high", temperature: 0.2 });
		expect(state.config.thinkingLevel).toBe("high");
		expect(state.config.temperature).toBe(0.2);
		expect(state.session.thinkingLevels).toEqual(["high"]);
		expect(state.backendLevels).toEqual(["high"]);
		expect(state.session.patches).toContainEqual({ temperature: 0.2 });
	});

	test("updates tool availability and reconfigures the live session", () => {
		const state = createRuntimeConfiguration();
		state.runtime.update({ graphicianEnabled: false, fffgrepEnabled: true });
		expect(state.tools.graphician).toEqual([false]);
		expect(state.tools.fffgrep).toEqual([true]);
		expect(state.session.patches.at(-1)?.tools).toEqual(state.config.tools);
	});

	test("owns feature lifecycle side effects and notices", () => {
		const state = createRuntimeConfiguration();
		state.runtime.update({ memoryEnabled: false, legroomEnabled: false });
		expect(state.memory.enabled).toEqual([
			{ enabled: false, sessionId: "session-1" },
		]);
		expect(state.legroomCloses()).toBe(1);
		expect(state.events).toHaveLength(2);
		expect(state.runtime.read().legroomEnabled).toBe(false);
	});
});
