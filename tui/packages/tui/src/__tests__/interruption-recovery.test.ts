import assert from "node:assert/strict";
import test from "node:test";
import { Transcript } from "@logician/coding-agent/sessions";
import { LogicianTUI } from "../app/tui.ts";

interface CancellationTestTui {
	cancellationPending: boolean;
	pendingPermission: { toolCallId: string; toolName: string } | null;
	bridge: {
		isActive(): boolean;
		cancel(): Promise<{
			clearedSteering: string[];
			clearedFollowUp: string[];
			clearedNextTurn: string[];
		}>;
	};
	transcript: Transcript;
	inputBar: { valueText: string };
	statusPanel: {
		update(info: { phase: string }): void;
		startAnimation(): void;
		stopAnimation(): void;
	};
	notifications: { show(message: string, level: string): void };
	transcriptDisplay: { setTurns(turns: unknown[]): void };
	tui: { requestRender(): void };
	cancelActiveTurn(): Promise<void>;
}

void test("interruption waits for settlement and restores the active prompt", async () => {
	let settle!: () => void;
	const settled = new Promise<void>((resolve) => {
		settle = resolve;
	});
	const phases: string[] = [];
	const transcript = new Transcript();
	transcript.addTurn("repair the renderer");
	const instance = Object.create(
		LogicianTUI.prototype,
	) as CancellationTestTui;
	instance.cancellationPending = false;
	instance.pendingPermission = { toolCallId: "tool-1", toolName: "bash" };
	instance.transcript = transcript;
	instance.inputBar = { valueText: "" };
	instance.bridge = {
		isActive: () => true,
		cancel: async () => {
			await settled;
			return {
				clearedSteering: ["change it"],
				clearedFollowUp: [],
				clearedNextTurn: [],
			};
		},
	};
	instance.statusPanel = {
		update: ({ phase }) => phases.push(phase),
		startAnimation: () => {},
		stopAnimation: () => {},
	};
	instance.notifications = { show: () => {} };
	instance.transcriptDisplay = { setTurns: () => {} };
	instance.tui = { requestRender: () => {} };

	const cancellation = instance.cancelActiveTurn();
	assert.equal(instance.pendingPermission, null);
	assert.equal(instance.inputBar.valueText, "");
	assert.deepEqual(phases, ["cancelling"]);

	settle();
	await cancellation;

	assert.equal(instance.inputBar.valueText, "repair the renderer");
	assert.deepEqual(phases, ["cancelling", "ready"]);
	assert.match(
		transcript.getTurns().at(-1)?.userMessage?.content ?? "",
		/Turn interrupted safely.*prompt was restored.*Cleared 1 queued message/s,
	);
});

void test("interruption never overwrites text entered while cancellation settles", async () => {
	const transcript = new Transcript();
	transcript.addTurn("original prompt");
	const instance = Object.create(
		LogicianTUI.prototype,
	) as CancellationTestTui;
	instance.cancellationPending = false;
	instance.pendingPermission = null;
	instance.transcript = transcript;
	instance.inputBar = { valueText: "" };
	instance.bridge = {
		isActive: () => true,
		cancel: async () => {
			instance.inputBar.valueText = "new draft";
			return {
				clearedSteering: [],
				clearedFollowUp: [],
				clearedNextTurn: [],
			};
		},
	};
	instance.statusPanel = {
		update: () => {},
		startAnimation: () => {},
		stopAnimation: () => {},
	};
	instance.notifications = { show: () => {} };
	instance.transcriptDisplay = { setTurns: () => {} };
	instance.tui = { requestRender: () => {} };

	await instance.cancelActiveTurn();

	assert.equal(instance.inputBar.valueText, "new draft");
});
