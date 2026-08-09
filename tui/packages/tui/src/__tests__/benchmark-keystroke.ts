#!/usr/bin/env tsx
// ── Keystroke Latency vs. Transcript Size Benchmark ────────────────────────
// Reproduces the reported symptom: once a lot of text has accumulated in the
// TUI transcript, typing in the input bar starts to feel laggy. This
// isolates exactly one thing — the cost of a single keystroke, end to end —
// and sweeps it across increasing transcript sizes to show whether/how
// latency scales with backlog size.
//
// "End to end" here means the same path a real keypress takes
// (packages/tui/src/terminal/core.ts):
//   stdin data -> handleInput() -> InputBar.handleInput() (mutates state)
//   -> requestRender() -> doRender() -> renderLayoutFrame() (Flex + ScrollView
//      + TranscriptDisplay layout/render) -> _commitFrame() (cell-level diff
//      against the previous frame + escape-sequence generation)
//
// TranscriptDisplay is unbounded by default (maxTurns/maxRenderedLines
// default to Infinity — app/tui.ts only passes an explicit cap through when
// a user sets transcriptMaxTurns/transcriptMaxRenderedLines in
// settings.json). Latency should stay flat regardless of how much history
// exists: painting is clipped to the viewport (paintBox only walks the
// visible rect) and the diff is viewport-sized (termHeight rows), not
// proportional to total transcript size.
//
// The transcript content itself does not change on a keystroke — only the
// input bar does — so any growth in latency as transcript size increases
// would indicate re-walking/re-diffing a bigger tree, not more work that's
// intrinsically necessary.
//
// Run:  npx tsx packages/tui/src/__tests__/benchmark-keystroke.ts [--json]
// Output: table (or JSON) of p50/p95/p99 keystroke latency per transcript size.

import { performance } from "node:perf_hooks";
import { cpus } from "node:os";

import type {
	AssistantChunk,
	AssistantMessage,
	ToolExecution,
	Turn,
	UserMessage,
} from "@logician/coding-agent/sessions";

import { initTheme } from "../terminal/theme.ts";
import { Container } from "../terminal/core.ts";
import { TranscriptDisplay } from "../rendering/transcript/display.ts";
import { ScrollView } from "../rendering/scroll-view.ts";
import { Flex } from "../rendering/flex.ts";
import { renderLayoutFrame, type LayoutFrame } from "../rendering/layout.ts";
import { InputBar } from "../input/input-bar.ts";
import { StatusBar } from "../status/status-bar.ts";

initTheme("dark");

// ── CLI args ──────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const jsonMode = args.includes("--json");
const width = process.stdout.columns ?? 120;
const termHeight = process.stdout.rows ?? 40;
const KEYSTROKES_PER_SIZE = 300; // sample size per transcript-size bucket

// Transcript sizes to sweep, expressed as (turns, chunkSize) so both turn
// count and per-turn content size are covered — "a bunch of text" can mean
// either many short turns or fewer very long ones. The largest buckets model
// long real coding sessions (hundreds-to-thousands of turns, some with large
// tool outputs) to confirm the viewport clip keeps latency flat even at
// extreme backlog sizes, not just small ones.
const SIZES: Array<{ label: string; turns: number; chunkSize: number }> = [
	{ label: "empty", turns: 0, chunkSize: 0 },
	{ label: "small (5 turns)", turns: 5, chunkSize: 300 },
	{ label: "medium (25 turns)", turns: 25, chunkSize: 400 },
	{ label: "large (75 turns)", turns: 75, chunkSize: 500 },
	{ label: "huge (150 turns)", turns: 150, chunkSize: 600 },
	{ label: "extreme (300 turns)", turns: 300, chunkSize: 800 },
	{ label: "long session (600 turns)", turns: 600, chunkSize: 1000 },
	{ label: "very long (1200 turns)", turns: 1200, chunkSize: 1200 },
	{ label: "marathon (2500 turns)", turns: 2500, chunkSize: 1500 },
];

// ── Synthetic transcript generation (same shape as benchmark.ts) ──────────

function genRandomText(length: number, seed: number): string {
	const words = ["the", "function", "variable", "component", "render", "update",
		"handle", "process", "compute", "validate", "transform", "generate",
		"parse", "format", "display", "stream", "buffer", "cache", "state"];
	let text = "";
	let n = seed;
	while (text.length < length) {
		n = (n * 1103515245 + 12345) & 0x7fffffff;
		text += words[n % words.length] + " ";
	}
	return text.slice(0, length);
}

function genToolResult(resultSize: number, turnIdx: number): ToolExecution {
	const toolNames = ["read_file", "write_file", "edit_file", "grep", "bash", "find", "git"];
	const tool = toolNames[turnIdx % toolNames.length];
	return {
		tool_name: tool,
		args: { path: `/src/file${turnIdx}.ts`, command: `echo "hello"` },
		result: genRandomText(resultSize, turnIdx),
		isError: false,
		isComplete: true,
		durationMs: Math.floor(Math.random() * 500) + 10,
	};
}

function genChunks(turnIdx: number, chunkSize: number): AssistantChunk[] {
	const chunks: AssistantChunk[] = [];
	const numChunks = 3 + (turnIdx % 4);
	for (let ci = 0; ci < numChunks; ci++) {
		const type = ci === 0 ? "thinking" : ci < numChunks - 1 ? "content" : "tool";
		if (type === "thinking" || type === "content") {
			chunks.push({
				seq: ci,
				type,
				contentText: genRandomText(chunkSize, turnIdx + ci),
				isComplete: true,
			});
		} else {
			chunks.push({ seq: ci, type: "tool", tool: genToolResult(chunkSize, turnIdx), isComplete: true });
		}
	}
	return chunks;
}

function genTurn(turnIdx: number, chunkSize: number): Turn {
	const userMsg: UserMessage = { type: "user", content: `Task ${turnIdx}: do the thing.` };
	const assistantMsg: AssistantMessage = { type: "assistant", chunks: genChunks(turnIdx, chunkSize), isComplete: true };
	return { id: `turn-${turnIdx}`, userMessage: userMsg, assistantMessage: assistantMsg, isComplete: true };
}

function genTranscript(numTurns: number, chunkSize: number): Turn[] {
	const turns: Turn[] = [];
	for (let i = 0; i < numTurns; i++) turns.push(genTurn(i, chunkSize));
	return turns;
}

// ── Helpers ─────────────────────────────────────────────────────────────

function percentile(values: number[], p: number): number {
	if (values.length === 0) return 0;
	const sorted = values.slice().sort((a, b) => a - b);
	const idx = Math.ceil((p / 100) * sorted.length) - 1;
	return sorted[Math.max(0, idx)];
}

const KEY_SEQUENCE = "abcdefghijklmnopqrstuvwxyz          .,".split("");

// ── Build the same tree app/tui.ts builds ──────────────────────────────────
// (ScrollView(transcriptDisplay) + dock(inputBar, statusBar) inside a Flex
// column) so the benchmark exercises the real layout/diff cost, not a
// simplified stand-in.

function buildFrame(numTurns: number, chunkSize: number) {
	const transcriptDisplay = new TranscriptDisplay({
		thinkingMode: "collapsed",
		maxTurns: Number.POSITIVE_INFINITY,
		maxRenderedLines: Number.POSITIVE_INFINITY,
	});
	const turns = genTranscript(numTurns, chunkSize);
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

	return { root, inputBar, transcriptDisplay };
}

// Mirrors core.ts's _commitFrame: cell/row-level diff against the previous
// frame's lines, producing the escape-sequence buffer that would actually be
// written to the terminal. This is real cost, not a stand-in — a keystroke
// that only changes the input bar still pays for diffing every row because
// the diff loop walks the full frame to find which rows changed.
function commitFrameDiffCost(prevLines: string[], newLines: string[], termWidth: number, termHeight: number): string {
	const renderWidth = Math.max(1, termWidth - 1);
	let changes = "";
	for (let row = 0; row < termHeight; row++) {
		const prevLine = prevLines[row];
		const newLine = row < newLines.length ? newLines[row] : " ".repeat(termWidth);
		if (prevLine === newLine) continue;
		changes += `\x1b[${row + 1};1H\x1b[0m\x1b[2K${newLine.slice(0, renderWidth)}`;
	}
	return changes;
}

// ── One simulated keystroke: the real end-to-end path ──────────────────────

function simulateKeystroke(
	inputBar: InputBar,
	root: Flex,
	prevLines: string[],
	key: string,
): number {
	const start = performance.now();

	// 1. stdin -> focused component handler (real InputBar state mutation)
	inputBar.handleInput(key);

	// 2. requestRender -> layout + render (real layout engine, same call as
	//    core.ts's _doRenderInnerLayoutEngine)
	const frame: LayoutFrame = renderLayoutFrame(root, width - 1, termHeight, () => {});

	// 3. _commitFrame's diff against the previous frame
	commitFrameDiffCost(prevLines, frame.lines, width, termHeight);

	const elapsed = performance.now() - start;

	prevLines.length = 0;
	prevLines.push(...frame.lines);
	return elapsed;
}

// ── Main ────────────────────────────────────────────────────────────────

interface SizeResult {
	label: string;
	turns: number;
	approxChars: number;
	p50Ms: number;
	p95Ms: number;
	p99Ms: number;
	maxMs: number;
	minMs: number;
}

function runBenchmark(): SizeResult[] {
	const results: SizeResult[] = [];
	for (const size of SIZES) {
		const { root, inputBar } = buildFrame(size.turns, size.chunkSize);

		// Warm up: establish prevLines as if the screen was already painted,
		// same as steady-state typing (not a cold first paint).
		const warm = renderLayoutFrame(root, width - 1, termHeight, () => {});
		const prevLines = [...warm.lines];

		// A few throwaway keystrokes to let any JIT warmup happen before we
		// measure, so we're capturing steady-state cost, not compile cost.
		for (let i = 0; i < 10; i++) {
			simulateKeystroke(inputBar, root, prevLines, KEY_SEQUENCE[i % KEY_SEQUENCE.length]);
		}
		inputBar.handleInput("\x15"); // Ctrl-U: clear line, reset for real measurement

		const times: number[] = [];
		for (let i = 0; i < KEYSTROKES_PER_SIZE; i++) {
			const key = KEY_SEQUENCE[i % KEY_SEQUENCE.length];
			times.push(simulateKeystroke(inputBar, root, prevLines, key));
		}

		results.push(summarize(size, times));
	}
	return results;
}

function summarize(size: { label: string; turns: number; chunkSize: number }, times: number[]): SizeResult {
	const approxChars = size.turns * size.chunkSize * 6; // ~6 chunks/turn average
	return {
		label: size.label,
		turns: size.turns,
		approxChars,
		p50Ms: Math.round(percentile(times, 50) * 1000) / 1000,
		p95Ms: Math.round(percentile(times, 95) * 1000) / 1000,
		p99Ms: Math.round(percentile(times, 99) * 1000) / 1000,
		minMs: Math.round(Math.min(...times) * 1000) / 1000,
		maxMs: Math.round(Math.max(...times) * 1000) / 1000,
	};
}

function reportResults(results: SizeResult[]): void {
	console.log(`\n--- Per-keystroke latency ---\n`);
	const header = `  ${"Transcript".padEnd(22)} ${"p50".padStart(8)} ${"p95".padStart(8)} ${"p99".padStart(8)} ${"max".padStart(8)}`;
	console.log(header);
	console.log(`  ${"-".repeat(header.length - 2)}`);
	for (const r of results) {
		console.log(
			`  ${r.label.padEnd(22)} ${(r.p50Ms.toFixed(3) + "ms").padStart(8)} ${(r.p95Ms.toFixed(3) + "ms").padStart(8)} ${(r.p99Ms.toFixed(3) + "ms").padStart(8)} ${(r.maxMs.toFixed(3) + "ms").padStart(8)}`,
		);
	}

	console.log(`\n  Scaling vs. empty transcript:`);
	const baseline = results[0];
	for (const r of results) {
		const factor = baseline.p50Ms > 0 ? r.p50Ms / baseline.p50Ms : 1;
		console.log(`    ${r.label.padEnd(22)} ${factor.toFixed(1)}x`);
	}

	const worst = results[results.length - 1];
	console.log(`\n  Verdict:`);
	console.log(`    Empty:   p50 ${baseline.p50Ms.toFixed(3)}ms, p99 ${baseline.p99Ms.toFixed(3)}ms`);
	console.log(`    Largest: p50 ${worst.p50Ms.toFixed(3)}ms, p99 ${worst.p99Ms.toFixed(3)}ms`);
	if (worst.p50Ms > 16) {
		console.log(`    ⚠ Largest transcript size exceeds the 16ms frame budget — user-perceptible typing lag.`);
	}
	if (worst.p50Ms / Math.max(baseline.p50Ms, 0.001) > 3) {
		console.log(`    ⚠ Latency scales with transcript size — indicates re-walking/re-diffing backlog content`);
		console.log(`      on every keystroke instead of reusing cached, viewport-bounded output.`);
	} else {
		console.log(`    ✓ Latency stays roughly flat as transcript size grows.`);
	}
}

async function main() {
	if (!jsonMode) {
		console.log(`\n=== Keystroke Latency vs. Transcript Size ===`);
		console.log(`Terminal: ${width}x${termHeight}`);
		console.log(`Keystrokes sampled per size: ${KEYSTROKES_PER_SIZE}`);
		console.log(`Node: ${process.version}, Platform: ${process.platform}, Cores: ${cpus().length}`);
	}

	const results = runBenchmark();

	if (jsonMode) {
		console.log(JSON.stringify({ width, termHeight, keystrokesPerSize: KEYSTROKES_PER_SIZE, results }, null, 2));
		return;
	}

	reportResults(results);
	console.log();
}

main().catch(err => {
	console.error("Benchmark failed:", err);
	process.exit(1);
});
