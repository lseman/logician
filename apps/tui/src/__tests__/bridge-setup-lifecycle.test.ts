import { expect, test } from "bun:test";
import { RuntimeEventBus } from "@logician/log-runtime/application";
import {
	type BridgeEventHandlerCtx,
	setupBridge,
} from "../app/bridge-event-handler.ts";

test("disposing bridge setup suppresses subscriptions and late initialization", async () => {
	const events = new RuntimeEventBus();
	let resolveInitialization: (state: Record<string, unknown>) => void =
		() => {};
	const initialization = new Promise<Record<string, unknown>>(resolve => {
		resolveInitialization = resolve;
	});
	const statusUpdates: unknown[] = [];
	const systemMessages: string[] = [];
	const ctx = {
		bridge: {
			events,
			init: () => initialization,
			getSandboxMode: () => "workspace-write",
			getSettingsData: () => ({ memoriamEnabled: false }),
		},
		statusPanel: { update: (value: unknown) => statusUpdates.push(value) },
		transcript: {
			addSystemMessage: (message: string) => systemMessages.push(message),
			getTurns: () => [],
		},
		transcriptDisplay: { setTurns: () => {} },
		tui: { requestRender: () => {} },
	} as unknown as BridgeEventHandlerCtx;

	const dispose = setupBridge(ctx);
	dispose();
	events.notifyError(new Error("after disposal"));
	resolveInitialization({ context_tokens: 12, context_max_tokens: 100 });
	await initialization;
	await Promise.resolve();

	expect(statusUpdates).toEqual([]);
	expect(systemMessages).toEqual([]);
});
