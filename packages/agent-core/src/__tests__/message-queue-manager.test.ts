import { test } from "bun:test";
import assert from "node:assert/strict";
import { MessageQueue } from "../runtime/queue/queue.ts";

void test("one-at-a-time steering drains without dropping later messages", () => {
	const manager = new MessageQueue({ steeringMode: "one-at-a-time" });
	manager.steering("one");
	manager.steering("two");

	assert.deepEqual(
		manager.afterTurn().map(message => message.content),
		["one"],
	);
	assert.deepEqual(
		manager.getSteering().map(message => message.content),
		["two"],
	);
	assert.deepEqual(
		manager.afterTurn().map(message => message.content),
		["two"],
	);
});

void test("one-at-a-time idle drain preserves remaining steering and follow-ups", () => {
	const manager = new MessageQueue({
		steeringMode: "one-at-a-time",
		followUpMode: "one-at-a-time",
	});
	manager.steering("steer-one");
	manager.steering("steer-two");
	manager.followUp("follow-one");
	manager.followUp("follow-two");

	assert.deepEqual(
		manager.onIdle().map(message => message.content),
		["steer-one", "follow-one"],
	);
	assert.deepEqual(
		manager.getSteering().map(message => message.content),
		["steer-two"],
	);
	assert.deepEqual(
		manager.getFollowUp().map(message => message.content),
		["follow-two"],
	);
});
