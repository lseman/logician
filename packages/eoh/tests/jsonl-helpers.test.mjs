import { test } from "node:test";
import assert from "node:assert";
import {
	parseJsonlEntry,
	isEohConfigEntry,
	isEohRunEntry,
	reconstructEohState,
	extractEohSessionName,
	hasEohConfigHeader,
} from "../src/jsonl.ts";

test("parseJsonlEntry parses valid JSON", () => {
	const entry = parseJsonlEntry('{"run": 1, "fitness": 0.5}');
	assert.ok(entry);
	assert.equal(entry.run, 1);
	assert.equal(entry.fitness, 0.5);
});

test("parseJsonlEntry returns null for invalid JSON", () => {
	assert.equal(parseJsonlEntry("not json"), null);
	assert.equal(parseJsonlEntry(""), null);
	assert.equal(parseJsonlEntry("[]"), null);
});

test("isEohConfigEntry identifies config entries", () => {
	assert.ok(isEohConfigEntry({ type: "eoh_config", name: "test" }));
	assert.ok(!isEohConfigEntry({ type: "run", run: 1 }));
	assert.ok(!isEohConfigEntry(null));
});

test("isEohRunEntry identifies run entries", () => {
	assert.ok(isEohRunEntry({ run: 1, fitness: 0.5 }));
	assert.ok(!isEohRunEntry({ type: "eoh_config" }));
});

test("hasEohConfigHeader detects config header", () => {
	const withConfig = '{"type": "eoh_config", "name": "test"}\n{"run": 1}';
	assert.ok(hasEohConfigHeader(withConfig));
	assert.ok(!hasEohConfigHeader('{"run": 1}'));
});

test("extractEohSessionName extracts name from config", () => {
	const withName = '{"type": "eoh_config", "name": "my-eoh"}\n{"run": 1}';
	assert.equal(extractEohSessionName(withName), "my-eoh");
	assert.equal(extractEohSessionName('{"run": 1}'), "EoH");
});

test("reconstructEohState rebuilds state from JSONL", () => {
	const jsonl = [
		'{"type": "eoh_config", "name": "test", "populationSize": 5, "maxGenerations": 10}',
		'{"run": 1, "fitness": 0.3, "generation": 1, "createdBy": "init", "status": "keep", "description": "first", "timestamp": 1000}',
		'{"run": 2, "fitness": 0.7, "generation": 2, "createdBy": "m1_improve", "status": "keep", "description": "improved", "timestamp": 2000}',
	].join("\n");

	const state = reconstructEohState(jsonl);
	assert.equal(state.name, "test");
	assert.equal(state.populationSize, 5);
	assert.equal(state.maxGenerations, 10);
	assert.equal(state.currentSegment, 0);
	assert.equal(state.results.length, 2);
	assert.equal(state.results[0].fitness, 0.3);
	assert.equal(state.results[1].fitness, 0.7);
	assert.equal(state.results[1].createdBy, "m1_improve");
});

test("reconstructEohState handles multiple segments", () => {
	const jsonl = [
		'{"type": "eoh_config", "name": "seg1"}',
		'{"run": 1, "fitness": 0.3, "generation": 1, "createdBy": "init", "status": "keep", "description": "seg1", "timestamp": 1000}',
		'{"type": "eoh_config", "name": "seg2"}',
		'{"run": 2, "fitness": 0.5, "generation": 2, "createdBy": "init", "status": "keep", "description": "seg2", "timestamp": 2000}',
	].join("\n");

	const state = reconstructEohState(jsonl);
	assert.equal(state.currentSegment, 1);
	assert.equal(state.results.length, 2);
	assert.equal(state.results[0].segment, 0);
	assert.equal(state.results[1].segment, 1);
});

test("reconstructEohState handles empty JSONL", () => {
	const state = reconstructEohState("");
	assert.equal(state.name, null);
	assert.equal(state.results.length, 0);
	assert.equal(state.currentSegment, 0);
});
