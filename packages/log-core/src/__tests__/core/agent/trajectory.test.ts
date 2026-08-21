import { test } from "bun:test";
import assert from "node:assert/strict";
import {
	gradeTrajectory,
	summarizeTrajectory,
} from "../../../core/evaluation/trajectory.ts";
import type { AgentEvent } from "../../../core/types/types-messages.ts";

void test("trajectory summary measures outcomes and autonomous friction", () => {
	const events = [
		{ type: "turn_start", turnId: "1" },
		{
			type: "tool_execution_start",
			toolCallId: "c",
			toolName: "bash",
			args: {},
		},
		{
			type: "tool_execution_end",
			toolCallId: "c",
			toolName: "bash",
			result: "denied",
			isError: true,
		},
		{
			type: "tool_permission_decision",
			toolCallId: "c",
			toolName: "bash",
			decision: "deny",
			source: "rule",
		},
		{ type: "acceptance_complete", status: "failed" },
		{ type: "agent_end", messages: [], status: "failed" },
	] satisfies AgentEvent[];
	assert.deepEqual(summarizeTrajectory(events), {
		status: "failed",
		turns: 1,
		toolCalls: 1,
		toolErrors: 1,
		permissionDenials: 1,
		continuations: 0,
		interventions: 0,
		compactions: 0,
		verificationPassed: false,
	});
});

void test("trajectory grading catches outcome and autonomy regressions", () => {
	const events = [
		{ type: "turn_start", turnId: "1" },
		{ type: "turn_start", turnId: "2" },
		{ type: "acceptance_complete", status: "failed" },
		{ type: "agent_end", messages: [], status: "failed" },
	] satisfies AgentEvent[];
	const grade = gradeTrajectory(events, {
		status: "completed",
		verificationPassed: true,
		maxTurns: 1,
	});
	assert.equal(grade.passed, false);
	assert.deepEqual(grade.failures, [
		"status: expected completed, got failed",
		"verification: expected true, got false",
		"turns: maximum 1, got 2",
	]);
});
