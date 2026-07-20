import assert from "node:assert/strict";
import { test } from "node:test";
import { MessageDeliveryManager } from "../message-queue/manager.ts";

void test("one-at-a-time steering drains without dropping later messages", () => {
	const manager = new MessageDeliveryManager({ steeringMode: "one-at-a-time" });
	manager.queue.steering("one");
	manager.queue.steering("two");

	assert.deepEqual(manager.afterTurn().map((message) => message.content), ["one"]);
	assert.deepEqual(manager.queue.getSteering().map((message) => message.content), ["two"]);
	assert.deepEqual(manager.afterTurn().map((message) => message.content), ["two"]);
});

void test("one-at-a-time idle drain preserves remaining steering and follow-ups", () => {
	const manager = new MessageDeliveryManager({
		steeringMode: "one-at-a-time",
		followUpMode: "one-at-a-time",
	});
	manager.queue.steering("steer-one");
	manager.queue.steering("steer-two");
	manager.queue.followUp("follow-one");
	manager.queue.followUp("follow-two");

	assert.deepEqual(manager.onIdle().map((message) => message.content), [
		"steer-one",
		"follow-one",
	]);
	assert.deepEqual(manager.queue.getSteering().map((message) => message.content), ["steer-two"]);
	assert.deepEqual(manager.queue.getFollowUp().map((message) => message.content), ["follow-two"]);
});
