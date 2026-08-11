#!/usr/bin/env tsx

// Profiler — breaks down keystroke latency by phase.
// Usage: cd tui && npx --no-install tsx packages/tui/src/__tests__/profile-keystroke.ts

import { performance } from "node:perf_hooks";
import type { Turn, AssistantMessage, UserMessage, AssistantChunk } from "@logician/coding-agent/sessions";
import { InputBar } from "../input/input-bar.ts";
import { Flex } from "../rendering/flex.ts";
import { renderLayoutFrame, type LayoutFrame } from "../rendering/layout.ts";
import { ScrollView } from "../rendering/scroll-view.ts";
import { TranscriptDisplay } from "../rendering/transcript/display.ts";
import { StatusBar } from "../status/status-bar.ts";
import { Container } from "../terminal/core.ts";
import { initTheme } from "../terminal/theme.ts";

initTheme("dark");

const width = 120;
const termHeight = 40;
const NUM_KEYSTROKES = 300;
const SIZES = [
	{ label: "empty", turns: 0, chunkSize: 0 },
	{ label: "5 turns", turns: 5, chunkSize: 300 },
	{ label: "150 turns", turns: 150, chunkSize: 600 },
];

function genRandomText(length: number, seed: number): string {
	const words = ["the","function","variable","component","render","update","handle","process"];
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
	const toolResult = {
		tool_name: ["read_file","bash","grep"][turnIdx % 3],
		args: {},
		result: genRandomText(chunkSize, turnIdx),
		isError: false,
		isComplete: true,
		durationMs: 10,
	};
	chunks.push({ seq: chunks.length, type: "tool", tool: toolResult as any, isComplete: true });
	return {
		id: `turn-${turnIdx}`,
		userMessage: { type: "user", content: `Task ${turnIdx}: do the thing.` } as UserMessage,
		assistantMessage: { type: "assistant", chunks, isComplete: true } as AssistantMessage,
		isComplete: true,
	};
}

function buildFrame(numTurns: number, chunkSize: number) {
	const transcriptDisplay = new TranscriptDisplay({ thinkingMode: "collapsed" });
	const turns: Turn[] = [];
	for (let i = 0; i < numTurns; i++) turns.push(genTurn(i, chunkSize));
	transcriptDisplay.setTurns(turns);

	const inputBar = new InputBar();
	const statusBar = new StatusBar();

	const transcriptScroll = new ScrollView(transcriptDisplay, {
		follow: "end", primary: true, overscroll: "chain", scrollbar: "always",
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

function main() {
	for (const size of SIZES) {
		console.log(`\n=== ${size.label} (${size.turns} turns) ===`);
		const { root, inputBar } = buildFrame(size.turns, size.chunkSize);

		// Warm up frame
		let frame = renderLayoutFrame(root, TOTAL_COLS, termHeight, () => {});
		const prevLines: string[] = [...frame.lines];

		// JIT warmup keystrokes
		for (let i = 0; i < 10; i++) {
			inputBar.handleInput(KEY_SEQUENCE[i % KEY_SEQUENCE.length]);
			frame = renderLayoutFrame(root, TOTAL_COLS, termHeight, () => {});
			prevLines.length = 0;
			prevLines.push(...frame.lines);
		}
		inputBar.handleInput("\x15"); // Ctrl-U clear

		const phaseTimes = { handleInput: [] as number[], layout: [] as number[], diff: [] as number[] };

		for (let i = 0; i < NUM_KEYSTROKES; i++) {
			const key = KEY_SEQUENCE[i % KEY_SEQUENCE.length];
			
			// Phase 1: handleInput
			const tStart = performance.now();
			inputBar.handleInput(key);
			phaseTimes.handleInput.push(performance.now() - tStart);

			// Phase 2: renderLayoutFrame
			const tLayoutStart = performance.now();
			frame = renderLayoutFrame(root, TOTAL_COLS, termHeight, () => {});
			phaseTimes.layout.push(performance.now() - tLayoutStart);

			// Phase 3: diff
			const tDiffStart = performance.now();
			for (let row = 0; row < termHeight; row++) {
				const prevLine = prevLines[row];
				const newLine = row < frame.lines.length ? frame.lines[row] : " ".repeat(termHeight);
				if (prevLine === newLine) continue;
			}
			phaseTimes.diff.push(performance.now() - tDiffStart);

			prevLines.length = 0;
			prevLines.push(...frame.lines);
		}

		for (const [phase, times] of Object.entries(phaseTimes)) {
			const sorted = [...times].sort((a, b) => a - b);
			const p50 = sorted[Math.floor(sorted.length * 0.5)];
			const p95 = sorted[Math.ceil(sorted.length * 0.95) - 1];
			const avg = times.reduce((s, t) => s + t, 0) / times.length;
			console.log(`  ${phase.padEnd(12)}: p50=${p50.toFixed(4)}ms p95=${p95.toFixed(4)}ms avg=${avg.toFixed(4)}ms`);
		}

		const totalAvg = (
			phaseTimes.handleInput.reduce((s, t) => s + t, 0) +
			phaseTimes.layout.reduce((s, t) => s + t, 0) +
			phaseTimes.diff.reduce((s, t) => s + t, 0)
		) / NUM_KEYSTROKES;
		console.log(`  ${"total".padEnd(12)}: avg=${totalAvg.toFixed(4)}ms`);

		const layoutPct = (phaseTimes.layout.reduce((s, t) => s + t, 0) / totalAvg * 100).toFixed(1);
		const diffPct = (phaseTimes.diff.reduce((s, t) => s + t, 0) / totalAvg * 100).toFixed(1);
		const inputPct = (phaseTimes.handleInput.reduce((s, t) => s + t, 0) / totalAvg * 100).toFixed(1);
		console.log(`  Distribution: handleInput=${inputPct}% layout=${layoutPct}% diff=${diffPct}%`);
	}
}

main();
