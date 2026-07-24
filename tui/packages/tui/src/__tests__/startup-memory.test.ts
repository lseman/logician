import assert from "node:assert/strict";
import { test } from "node:test";
import { formatStartupMemory } from "../layers/presentation/startup-memory.ts";

void test("formats persisted observational memory for the startup transcript", () => {
	const lines = formatStartupMemory({
		observational_memory: {
			observation_count: 2,
			active_observation_count: 1,
			reflection_count: 1,
			dropped_count: 1,
			observations: [
				{
					id: "aaaaaaaaaaaa",
					content: "Remember   this project decision.",
					relevance: "high",
				},
			],
			reflections: [
				{
					id: "bbbbbbbbbbbb",
					content: "The project has a stable memory policy.",
				},
			],
		},
	});

	assert.deepEqual(lines, [
		"",
		"## Observational memory",
		"1 active observations · 1 reflections · 1 archived",
		"",
		"### Recent observations",
		"- [high] Remember this project decision. (aaaaaaaaaaaa)",
		"",
		"### Recent reflections",
		"- The project has a stable memory policy. (bbbbbbbbbbbb)",
	]);
});

void test("omits an empty observational memory section", () => {
	assert.deepEqual(
		formatStartupMemory({
			observational_memory: {
				observation_count: 0,
				active_observation_count: 0,
				reflection_count: 0,
				dropped_count: 0,
				observations: [],
				reflections: [],
			},
		}),
		[],
	);
});
