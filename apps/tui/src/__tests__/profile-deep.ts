#!/usr/bin/env tsx

// Deep profiler — breaks down keystroke latency into finer buckets.
// Usage: cd apps/tui && npx --no-install tsx src/__tests__/profile-deep.ts

import { performance } from "node:perf_hooks";
import type {
	AssistantChunk,
	AssistantMessage,
	ToolExecution,
	Turn,
	UserMessage,
} from "@logician/agent-core/sessions";
import { InputBar } from "../input/input-bar.ts";
import { Flex } from "../rendering/flex.ts";
import { renderLayoutFrame } from "../rendering/layout.ts";
import { ScrollView } from "../rendering/scroll-view.ts";
import { TranscriptDisplay } from "../rendering/transcript/display.ts";
import { StatusBar } from "../status/status-bar.ts";
import { Container, CURSOR_MARKER } from "../terminal/core.ts";
import { initTheme } from "../terminal/theme.ts";

initTheme("dark");

const width = 120;
const termHeight = 40;
const NUM_KEYSTROKES = 300;

function genRandomText(length: number, seed: number): string {
	const words = [
		"the",
		"function",
		"variable",
		"component",
		"render",
		"update",
		"handle",
		"process",
		"compute",
	];
	let text = "";
	let n = seed;
	while (text.length < length) {
		n = (n * 1103515245 + 12345) & 0x7fffffff;
		text += `${words[n % words.length]} `;
	}
	return text.slice(0, length);
}

function genTurn(turnIdx: number, chunkSize: number): Turn {
	const chunks: AssistantChunk[] = [];
	for (let ci = 0; ci < 3 + (turnIdx % 4); ci++) {
		chunks.push({
			seq: ci,
			type: ci === 0 ? "thinking" : ci < 3 ? "content" : "tool",
			contentText: genRandomText(chunkSize, turnIdx + ci),
			isComplete: true,
		});
	}
	const toolResult: ToolExecution = {
		tool_name: ["read_file", "bash", "grep"][turnIdx % 3],
		args: {},
		result: genRandomText(chunkSize, turnIdx),
		isError: false,
		isComplete: true,
		durationMs: 10,
	};
	chunks.push({
		seq: chunks.length,
		type: "tool",
		tool: toolResult as any,
		isComplete: true,
	});
	return {
		id: `turn-${turnIdx}`,
		userMessage: {
			type: "user",
			content: `Task ${turnIdx}: do the thing.`,
		} as UserMessage,
		assistantMessage: {
			type: "assistant",
			chunks,
			isComplete: true,
		} as AssistantMessage,
		isComplete: true,
	};
}

function buildFrame(numTurns: number, chunkSize: number) {
	const transcriptDisplay = new TranscriptDisplay({
		thinkingMode: "collapsed",
	});
	const turns: Turn[] = [];
	for (let i = 0; i < numTurns; i++) turns.push(genTurn(i, chunkSize));
	transcriptDisplay.setTurns(turns);

	const inputBar = new InputBar();
	const statusBar = new StatusBar();

	const transcriptScroll = new ScrollView(transcriptDisplay, {
		follow: "end",
		primary: true,
		overscroll: "chain",
		scrollbar: "always",
	});
	transcriptDisplay.setScrollView(transcriptScroll);

	const dock = new Container();
	dock.addChild(inputBar);
	dock.addChild(statusBar);

	const root = new Flex([
		{ component: transcriptScroll, basis: 0, grow: 1, shrink: 1, minSize: 1 },
		{ component: dock, basis: "auto", grow: 0, shrink: 1, minSize: 1 },
	]);

	return { root, inputBar };
}

const KEY_SEQUENCE = "abcdefghijklmnopqrstuvwxyz          .,".split("");
const TOTAL_COLS = width - 1;

// Full _commitFrame diff (matches real commitFrameDiffCost in benchmark)
function fullCommitFrameDiff(prevLines: string[], newLines: string[]): void {
	const _renderWidth = Math.max(1, TOTAL_COLS);
	for (let row = 0; row < termHeight; row++) {
		const prevLine = prevLines[row];
		const newLine =
			row < newLines.length ? newLines[row] : " ".repeat(termHeight);
		if (prevLine === newLine) continue;
		const cleanPrev = prevLine?.replace(CURSOR_MARKER, "") ?? "";
		const cleanNew = newLine.replace(CURSOR_MARKER, "");
		if (cleanPrev === cleanNew) continue;
	}
}

function measureBlock(_name: string, fn: () => void): number {
	const t0 = performance.now();
	fn();
	return performance.now() - t0;
}

function main() {
	for (const { label, numTurns, chunkSize } of [
		{ label: "empty", numTurns: 0, chunkSize: 0 },
		{ label: "5 turns", numTurns: 5, chunkSize: 300 },
		{ label: "150 turns", numTurns: 150, chunkSize: 600 },
	]) {
		console.log(`\n=== ${label} (${numTurns} turns) ===`);
		const { root, inputBar } = buildFrame(numTurns, chunkSize);

		let frame = renderLayoutFrame(root, TOTAL_COLS, termHeight, () => {});
		const prevLines: string[] = [...frame.lines];

		for (let i = 0; i < 10; i++) {
			inputBar.handleInput(KEY_SEQUENCE[i % KEY_SEQUENCE.length]);
			frame = renderLayoutFrame(root, TOTAL_COLS, termHeight, () => {});
			prevLines.length = 0;
			prevLines.push(...frame.lines);
		}
		inputBar.handleInput("\x15");

		const buckets = {
			handleInput: [] as number[],
			layout: [] as number[],
			diff: [] as number[],
			fullKeystroke: [] as number[],
		};

		for (let i = 0; i < NUM_KEYSTROKES; i++) {
			const key = KEY_SEQUENCE[i % KEY_SEQUENCE.length];

			// Full keystroke timing
			const tFullStart = performance.now();

			// handleInput
			buckets.handleInput.push(
				measureBlock("hi", () => inputBar.handleInput(key)),
			);

			// renderLayoutFrame (includes flex layout + all component renders)
			buckets.layout.push(
				measureBlock("layout", () => {
					frame = renderLayoutFrame(root, TOTAL_COLS, termHeight, () => {});
				}),
			);

			// full diff
			buckets.diff.push(
				measureBlock("diff", () => fullCommitFrameDiff(prevLines, frame.lines)),
			);

			prevLines.length = 0;
			prevLines.push(...frame.lines);

			buckets.fullKeystroke.push(performance.now() - tFullStart);
		}

		function stats(arr: number[]) {
			const sorted = [...arr].sort((a, b) => a - b);
			const p50 = sorted[Math.floor(sorted.length * 0.5)];
			const p95 = sorted[Math.ceil(sorted.length * 0.95) - 1];
			const avg = arr.reduce((s, t) => s + t, 0) / arr.length;
			return { p50: p50.toFixed(4), p95: p95.toFixed(4), avg: avg.toFixed(4) };
		}

		console.log(
			`  ${"handleInput".padEnd(16)}: p50=${stats(buckets.handleInput).p50}ms p95=${stats(buckets.handleInput).p95}ms avg=${stats(buckets.handleInput).avg}ms`,
		);
		console.log(
			`  ${"renderLayoutFrame".padEnd(16)}: p50=${stats(buckets.layout).p50}ms p95=${stats(buckets.layout).p95}ms avg=${stats(buckets.layout).avg}ms`,
		);
		console.log(
			`  ${"fullDiff".padEnd(16)}: p50=${stats(buckets.diff).p50}ms p95=${stats(buckets.diff).p95}ms avg=${stats(buckets.diff).avg}ms`,
		);
		console.log(
			`  ${"TOTAL".padEnd(16)}: p50=${stats(buckets.fullKeystroke).p50}ms p95=${stats(buckets.fullKeystroke).p95}ms avg=${stats(buckets.fullKeystroke).avg}ms`,
		);

		const layoutSum = buckets.layout.reduce((s, t) => s + t, 0);
		const diffSum = buckets.diff.reduce((s, t) => s + t, 0);
		const inputSum = buckets.handleInput.reduce((s, t) => s + t, 0);
		const totalSum = layoutSum + diffSum + inputSum;
		console.log(
			`  Distribution: handleInput=${((inputSum / totalSum) * 100).toFixed(1)}% renderLayout=${((layoutSum / totalSum) * 100).toFixed(1)}% diff=${((diffSum / totalSum) * 100).toFixed(1)}%`,
		);

		// TranscriptDisplay breakdown
		const td = new TranscriptDisplay({ thinkingMode: "collapsed" });
		const turns: Turn[] = [];
		for (let i = 0; i < numTurns; i++) turns.push(genTurn(i, chunkSize));
		td.setTurns(turns);

		// Measure how long td.render takes with different widths
		const tdTimes = Array.from({ length: 200 }, () =>
			measureBlock("td_render", () => td.render(TOTAL_COLS)),
		);
		console.log(
			`  TranscriptDisplay.render(): p50=${stats(tdTimes).p50}ms avg=${stats(tdTimes).avg}ms`,
		);
	}
}

main();
