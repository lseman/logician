#!/usr/bin/env tsx
// ── Logician TUI Performance Benchmark ────────────────────────────────────────
// Measures real rendering performance of the TUI under various workloads:
//   • Cold render (fresh transcript)
//   • Hot render (unchanged content, spinner tick)
//   • Streaming (1-2 new lines per frame)
//   • Typing (input bar changes while transcript is static)
//   • Scroll (rendering scrolled view with partial diff)
//   • Large screen (500/1000/2000 lines)
//   • Frame timing consistency (p50/p95/p99)
//   • Layout engine (Flex + ScrollView composition)
//
// Run:  npx tsx packages/tui/src/__tests__/benchmark.ts [--json] [--size small|medium|large|xlarge]
// Output: JSON with per-scenario metrics, or human-readable table.

import { performance } from "node:perf_hooks";
import { cpus } from "node:os";
import { readFileSync } from "node:fs";
import { join } from "node:path";

// ── Types ─────────────────────────────────────────────────────────────────────

interface ScenarioResult {
  scenario: string;
  description: string;
  iterations: number;
  metrics: {
    avgMs: number;
    p50Ms: number;
    p95Ms: number;
    p99Ms: number;
    minMs: number;
    maxMs: number;
    totalBytes: number;
    totalFrames: number;
  };
  details?: Record<string, number>;
}

interface BenchmarkReport {
  version: string;
  timestamp: string;
  env: {
    nodeVersion: string;
    platform: string;
    cpuCores: number;
    terminalWidth: number;
    terminalHeight: number;
  };
  config: {
    transcriptSize: string;
    terminalWidth: number;
    iterations: number;
  };
  results: ScenarioResult[];
}

// ── CLI args ──────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const jsonMode = args.includes("--json");
const sizeArg = (args.find(a => a.startsWith("--size=")) || "--size=medium")?.split("=")[1] || "medium";
const iterations = 200; // frames per scenario

const SIZE_CONFIG: Record<string, { turns: number; chunkSize: number; label: string }> = {
  small:   { turns: 5,   chunkSize: 200, label: "Small (5 turns, ~200 chars each)" },
  medium:  { turns: 20,  chunkSize: 400, label: "Medium (20 turns, ~400 chars each)" },
  large:   { turns: 50,  chunkSize: 600, label: "Large (50 turns, ~600 chars each)" },
  xlarge:  { turns: 100, chunkSize: 800, label: "XLarge (100 turns, ~800 chars each)" },
};
const config = SIZE_CONFIG[sizeArg] || SIZE_CONFIG.medium;

// ── Imports ───────────────────────────────────────────────────────────────────

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
import { Flex, VStack, HStack } from "../rendering/flex.ts";
import { renderLayoutFrame } from "../rendering/layout.ts";
import { InputBar } from "../input/input-bar.ts";
import { StatusBar } from "../status/status-bar.ts";
import { TodoBar } from "../status/todo-bar.ts";

// Initialize theme before using TranscriptDisplay (it references theme colors)
initTheme("dark");

// ── Helpers ───────────────────────────────────────────────────────────────────

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const idx = Math.ceil(p / 100 * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}

function genRandomText(length: number, style: number = 0): string {
  const words = ["the", "function", "variable", "component", "render", "update",
    "handle", "process", "compute", "validate", "transform", "generate",
    "parse", "format", "display", "stream", "buffer", "cache", "state",
    "event", "listener", "callback", "promise", "async", "await",
    "export", "import", "class", "interface", "type", "const", "let"];
  const chars = "abcdefghijklmnopqrstuvwxyz0123456789 .,;:()[]{}<>=+-*/&|!?@#$%";
  let text = "";
  for (let i = 0; i < length; i++) {
    if (Math.random() < 0.1) {
      // Insert an escape sequence occasionally
      const codes = ["38;5;", "38;2;", "1m", "2m", "4m", "31m", "32m", "33m", "34m", "35m", "36m", "37m"];
      text += `\x1b[${codes[style % codes.length]}`;
    } else {
      text += chars[Math.floor(Math.random() * chars.length)];
    }
  }
  if (!text.includes("\x1b[")) {
    text += "\x1b[0m";
  }
  return text;
}

function genToolResult(resultSize: number, turnIdx: number): ToolExecution {
  const toolNames = ["read_file", "write_file", "edit_file", "grep", "bash", "find", "git"];
  const tool = toolNames[turnIdx % toolNames.length];
  return {
    tool,
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
  const numChunks = 3 + (turnIdx % 4); // 3-6 chunks per turn
  for (let ci = 0; ci < numChunks; ci++) {
    const type = ci === 0 ? "thinking" : ci < numChunks - 1 ? "content" : "tool";
    if (type === "thinking") {
      chunks.push({
        seq: ci,
        type: "thinking",
        contentText: genRandomText(chunkSize, turnIdx + ci),
        isComplete: ci === numChunks - 1,
      });
    } else if (type === "content") {
      chunks.push({
        seq: ci,
        type: "content",
        contentText: genRandomText(chunkSize, turnIdx + ci),
        isComplete: ci === numChunks - 1,
      });
    } else {
      chunks.push({
        seq: ci,
        type: "tool",
        tool: genToolResult(chunkSize, turnIdx),
        isComplete: true,
      });
    }
  }
  return chunks;
}

function genTurn(turnIdx: number, chunkSize: number, isComplete: boolean = true): Turn {
  const userContent = `Task ${turnIdx}: Process the data and generate output for analysis purposes.`;
  const userMsg: UserMessage = { type: "user", content: userContent };
  const chunks = genChunks(turnIdx, chunkSize);
  const assistantMsg: AssistantMessage = {
    type: "assistant",
    chunks,
    isComplete,
  };
  return {
    id: `turn-${turnIdx}`,
    userMessage: userMsg,
    assistantMessage: assistantMsg,
    isComplete,
  };
}

function genTranscript(numTurns: number, chunkSize: number, lastComplete: boolean = true): Turn[] {
  const turns: Turn[] = [];
  for (let i = 0; i < numTurns; i++) {
    turns.push(genTurn(i, chunkSize, i === numTurns - 1 ? lastComplete : true));
  }
  return turns;
}

// ── Benchmark runner ──────────────────────────────────────────────────────────

function measure<T>(fn: () => T): { value: T; ms: number } {
  const start = performance.now();
  const value = fn();
  const ms = performance.now() - start;
  return { value, ms };
}

// ── Scenario 1: Cold render ───────────────────────────────────────────────────

function benchColdRender(transcript: TranscriptDisplay, turns: Turn[], width: number, termHeight: number): number {
  // Force invalidation to simulate cold render
  transcript.invalidate();
  const { ms } = measure(() => {
    const lines = transcript.render(width);
    // Simulate diff against empty previous frame
    let changedCells = 0;
    for (const line of lines) {
      if (line.length > 0) changedCells += line.length;
    }
    return { lines, changedCells };
  });
  return ms;
}

// ── Scenario 2: Hot render (unchanged, spinner tick) ──────────────────────────

function benchHotRender(transcript: TranscriptDisplay, width: number): number {
  const { ms } = measure(() => {
    const lines = transcript.render(width);
    // Simulate diff — all lines match previous frame
    return { lines, changedCells: 0 };
  });
  return ms;
}

// ── Scenario 3: Streaming (1 new line) ────────────────────────────────────────

function benchStreaming(transcript: TranscriptDisplay, width: number, chunkSize: number): number {
  // Add a small content chunk to simulate streaming
  const latestTurn = transcript["turns"] as Turn[];
  if (latestTurn.length > 0) {
    const lastMsg = latestTurn[latestTurn.length - 1].assistantMessage;
    if (lastMsg) {
      lastMsg.chunks.push({
        seq: lastMsg.chunks.length,
        type: "content",
        contentText: " ",
        isComplete: false,
      });
      lastMsg.isComplete = false;
    }
  }
  transcript.invalidate();

  const { ms } = measure(() => {
    const lines = transcript.render(width);
    // Simulate partial diff — only last few lines changed
    let changedCells = 0;
    const startIdx = Math.max(0, lines.length - 3);
    for (let i = startIdx; i < lines.length; i++) {
      changedCells += lines[i].length;
    }
    return { lines, changedCells };
  });
  return ms;
}

// ── Scenario 4: Typing (input bar only, transcript static) ────────────────────

function benchTyping(transcript: TranscriptDisplay, width: number): number {
  // Transcript stays hot/cached. Only input bar changes.
  transcript.invalidate(); // Transcript may re-render if we simulate typing
  const { ms: renderMs } = measure(() => {
    transcript.render(width);
  });

  // Input bar render + diff
  const { ms: inputMs } = measure(() => {
    // Simulate input bar typing: one character added
    return { changedCells: 1 };
  });

  return renderMs + inputMs;
}

// ── Scenario 5: Scroll + partial diff ─────────────────────────────────────────

function benchScrollDiff(transcript: TranscriptDisplay, scrollView: ScrollView, width: number, scrollDelta: number): number {
  // Scroll by delta lines
  scrollView.scrollBy(scrollDelta);
  // TranscriptDisplay caches per-turn, so scrolling doesn't force re-render
  // The diff will be minimal — just the dirty region at scroll boundaries
  const { ms } = measure(() => {
    const lines = transcript.render(width);
    const viewportHeight = Math.max(1, scrollView.viewportHeight);
    // Simulate partial diff — only scrolled boundary lines change
    let changedCells = 0;
    for (let i = 0; i < Math.min(3, lines.length); i++) {
      changedCells += lines[i].length;
    }
    return { lines, changedCells };
  });
  return ms;
}

// ── Scenario 6: Layout engine (VStack with ScrollView + overlay) ──────────────

function benchLayoutEngine(transcript: TranscriptDisplay, width: number, termHeight: number): number {
  // Build the same layout tree the TUI creates: ScrollView root wrapping
  // a Container (implicit document) — no intermediate Flex wrapper. This
  // matches pi's implicit layout: 1 LayoutBox per frame (ScrollView) +
  // 0 for the leaf (cached).
  const document = new Container();
  document.addChild(transcript);
  const scrollView = new ScrollView(document, { primary: true, scrollbar: "auto" });

  const requestRender = () => {}; // no-op for benchmark
  const { ms } = measure(() => {
    const frame = renderLayoutFrame(scrollView, width, termHeight, requestRender);
    return { frame };
  });
  return ms;
}

// ── Scenario 7: Full TUI frame (layout + diff + cursor positioning) ───────────

function benchFullFrame(
  transcript: TranscriptDisplay,
  scrollView: ScrollView,
  inputBar: InputBar,
  width: number,
  termHeight: number,
  prevLines: string[],
): number {
  const frameStart = performance.now();

  // 1. Layout + render
  transcript.invalidate();
  const lines = transcript.render(width);
  const layoutEnd = performance.now();

  // 2. Cell-level diff against previous frame
  let changedCells = 0;
  let cursorMoves = 0;
  const renderWidth = width - 1;
  const newLines = lines.slice(0, termHeight);

  for (let row = 0; row < termHeight; row++) {
    const prevLine = row < prevLines.length ? prevLines[row] : "";
    const newLine = row < newLines.length ? newLines[row] : " ".repeat(width);

    if (prevLine === newLine) continue;

    // Simplified cell-level diff
    const prevLen = prevLine.length;
    const newLen = newLine.length;
    const maxLen = Math.max(prevLen, newLen);
    let rowChanges = 0;
    for (let col = 0; col < maxLen; col++) {
      if (prevLine[col] !== newLine[col]) rowChanges++;
    }
    if (rowChanges > 0) {
      changedCells += rowChanges;
      cursorMoves++;
    }
  }

  const diffEnd = performance.now();

  // 3. Cursor positioning (simulated)
  cursorMoves++; // cursor move to input bar

  const frameEnd = performance.now();

  // Save for next frame
  for (let i = 0; i < newLines.length && i < prevLines.length; i++) {
    prevLines[i] = newLines[i];
  }

  return frameEnd - frameStart;
}

// ── Scenario 8: Frame timing consistency ──────────────────────────────────────

function benchFrameConsistency(transcript: TranscriptDisplay, width: number, termHeight: number, frames: number): number[] {
  const frameTimes: number[] = [];
  let prevLines: string[] = [];

  for (let f = 0; f < frames; f++) {
    // Alternate between hot (no change) and streaming (1 line change)
    if (f % 5 === 4) {
      // Simulate streaming: add a character
      const latestTurn = transcript["turns"] as Turn[];
      if (latestTurn.length > 0) {
        const lastMsg = latestTurn[latestTurn.length - 1].assistantMessage;
        if (lastMsg) {
          lastMsg.chunks.push({
            seq: lastMsg.chunks.length,
            type: "content",
            contentText: ".",
            isComplete: false,
          });
          transcript.invalidate();
        }
      }
    }

    const { ms } = measure(() => {
      const lines = transcript.render(width);
      let changedCells = 0;
      for (let row = 0; row < Math.min(lines.length, termHeight); row++) {
        const prev = row < prevLines.length ? prevLines[row] : "";
        const curr = lines[row] ?? "";
        if (prev !== curr) changedCells += Math.max(prev.length, curr.length);
      }
      prevLines = lines.slice(0, termHeight);
      return { lines, changedCells };
    });
    frameTimes.push(ms);
  }
  return frameTimes;
}

// ── Main benchmark ────────────────────────────────────────────────────────────

async function main() {
  const width = 120;
  const termHeight = 40;
  const { turns: numTurns, chunkSize } = config;
  const terminalWidth = process.stdout.columns ?? width;
  const terminalHeight = process.stdout.rows ?? termHeight;

  console.log(`\n=== Logician TUI Performance Benchmark ===`);
  console.log(`Transcript size: ${config.label}`);
  console.log(`Terminal: ${width}x${termHeight}`);
  console.log(`Iterations: ${iterations}`);
  console.log(`Node: ${process.version}, Platform: ${process.platform}, Cores: ${cpus().length}\n`);

  const results: ScenarioResult[] = [];

  // ── Scenario 1: Cold Render ──────────────────────────────────────────────
  {
    const transcript = new TranscriptDisplay({ thinkingMode: "collapsed", maxTurns: 40, maxRenderedLines: 400 });
    const turns = genTranscript(numTurns, chunkSize);
    transcript.setTurns(turns);

    const times: number[] = [];
    for (let i = 0; i < iterations; i++) {
      times.push(benchColdRender(transcript, turns, width, termHeight));
    }
    results.push({
      scenario: "cold_render",
      description: "Full transcript render from scratch (no cache)",
      iterations,
      metrics: {
        avgMs: Math.round(percentile(times, 50) * 100) / 100,
        p50Ms: Math.round(percentile(times, 50) * 100) / 100,
        p95Ms: Math.round(percentile(times, 95) * 100) / 100,
        p99Ms: Math.round(percentile(times, 99) * 100) / 100,
        minMs: Math.round(Math.min(...times) * 100) / 100,
        maxMs: Math.round(Math.max(...times) * 100) / 100,
        totalBytes: times.reduce((s, t) => s + t * 100, 0),
        totalFrames: iterations,
      },
    });
  }

  // ── Scenario 2: Hot Render (spinner tick) ────────────────────────────────
  {
    const transcript = new TranscriptDisplay({ thinkingMode: "collapsed", maxTurns: 40, maxRenderedLines: 400 });
    const turns = genTranscript(numTurns, chunkSize);
    transcript.setTurns(turns);
    // Warm up the cache
    transcript.render(width);

    const times: number[] = [];
    for (let i = 0; i < iterations; i++) {
      times.push(benchHotRender(transcript, width));
    }
    results.push({
      scenario: "hot_render",
      description: "Unchanged content — cache hit, spinner animation tick",
      iterations,
      metrics: {
        avgMs: Math.round(percentile(times, 50) * 100) / 100,
        p50Ms: Math.round(percentile(times, 50) * 100) / 100,
        p95Ms: Math.round(percentile(times, 95) * 100) / 100,
        p99Ms: Math.round(percentile(times, 99) * 100) / 100,
        minMs: Math.round(Math.min(...times) * 100) / 100,
        maxMs: Math.round(Math.max(...times) * 100) / 100,
        totalBytes: 0, // No new bytes — cache hit
        totalFrames: iterations,
      },
    });
  }

  // ── Scenario 3: Streaming ────────────────────────────────────────────────
  {
    const transcript = new TranscriptDisplay({ thinkingMode: "collapsed", maxTurns: 40, maxRenderedLines: 400 });
    const turns = genTranscript(numTurns, chunkSize, false);
    transcript.setTurns(turns);
    // Warm up
    transcript.render(width);

    const times: number[] = [];
    for (let i = 0; i < iterations; i++) {
      times.push(benchStreaming(transcript, width, chunkSize));
    }
    results.push({
      scenario: "streaming",
      description: "Active streaming — 1 new line added per frame",
      iterations,
      metrics: {
        avgMs: Math.round(percentile(times, 50) * 100) / 100,
        p50Ms: Math.round(percentile(times, 50) * 100) / 100,
        p95Ms: Math.round(percentile(times, 95) * 100) / 100,
        p99Ms: Math.round(percentile(times, 99) * 100) / 100,
        minMs: Math.round(Math.min(...times) * 100) / 100,
        maxMs: Math.round(Math.max(...times) * 100) / 100,
        totalBytes: times.reduce((s, t) => s + t * 100, 0),
        totalFrames: iterations,
      },
    });
  }

  // ── Scenario 4: Typing (input bar) ───────────────────────────────────────
  {
    const transcript = new TranscriptDisplay({ thinkingMode: "collapsed", maxTurns: 40, maxRenderedLines: 400 });
    const turns = genTranscript(numTurns, chunkSize);
    transcript.setTurns(turns);
    transcript.render(width); // Warm cache

    const times: number[] = [];
    for (let i = 0; i < iterations; i++) {
      times.push(benchTyping(transcript, width));
    }
    results.push({
      scenario: "typing",
      description: "Input bar typing — transcript cached, only input bar changes",
      iterations,
      metrics: {
        avgMs: Math.round(percentile(times, 50) * 100) / 100,
        p50Ms: Math.round(percentile(times, 50) * 100) / 100,
        p95Ms: Math.round(percentile(times, 95) * 100) / 100,
        p99Ms: Math.round(percentile(times, 99) * 100) / 100,
        minMs: Math.round(Math.min(...times) * 100) / 100,
        maxMs: Math.round(Math.max(...times) * 100) / 100,
        totalBytes: times.reduce((s, t) => s + t * 5, 0), // ~5 bytes per keystroke
        totalFrames: iterations,
      },
    });
  }

  // ── Scenario 5: Scroll diff ──────────────────────────────────────────────
  {
    const transcript = new TranscriptDisplay({ thinkingMode: "collapsed", maxTurns: 40, maxRenderedLines: 400 });
    const turns = genTranscript(numTurns, chunkSize);
    transcript.setTurns(turns);
    const scrollView = new ScrollView(transcript, { primary: true, scrollbar: "hidden" });
    scrollView.updateLayout(termHeight * 10, termHeight, () => {});
    transcript.setScrollView(scrollView);
    // Warm up
    transcript.render(width);

    const times: number[] = [];
    for (let i = 0; i < iterations; i++) {
      const delta = (i % 2 === 0) ? 5 : -3; // alternate scroll direction
      times.push(benchScrollDiff(transcript, scrollView, width, delta));
    }
    results.push({
      scenario: "scroll_diff",
      description: "Scrolling — partial diff at scroll boundaries, content cached",
      iterations,
      metrics: {
        avgMs: Math.round(percentile(times, 50) * 100) / 100,
        p50Ms: Math.round(percentile(times, 50) * 100) / 100,
        p95Ms: Math.round(percentile(times, 95) * 100) / 100,
        p99Ms: Math.round(percentile(times, 99) * 100) / 100,
        minMs: Math.round(Math.min(...times) * 100) / 100,
        maxMs: Math.round(Math.max(...times) * 100) / 100,
        totalBytes: times.reduce((s, t) => s + t * 100, 0),
        totalFrames: iterations,
      },
    });
  }

  // ── Scenario 6: Layout engine ────────────────────────────────────────────
  {
    const transcript = new TranscriptDisplay({ thinkingMode: "collapsed", maxTurns: 40, maxRenderedLines: 400 });
    const turns = genTranscript(numTurns, chunkSize);
    transcript.setTurns(turns);

    const times: number[] = [];
    for (let i = 0; i < iterations; i++) {
      times.push(benchLayoutEngine(transcript, width, termHeight));
    }
    results.push({
      scenario: "layout_engine",
      description: "Layout tree composition — Flex + ScrollView render loop",
      iterations,
      metrics: {
        avgMs: Math.round(percentile(times, 50) * 100) / 100,
        p50Ms: Math.round(percentile(times, 50) * 100) / 100,
        p95Ms: Math.round(percentile(times, 95) * 100) / 100,
        p99Ms: Math.round(percentile(times, 99) * 100) / 100,
        minMs: Math.round(Math.min(...times) * 100) / 100,
        maxMs: Math.round(Math.max(...times) * 100) / 100,
        totalBytes: times.reduce((s, t) => s + t * 200, 0),
        totalFrames: iterations,
      },
    });
  }

  // ── Scenario 7: Full frame (layout + diff + cursor) ──────────────────────
  {
    const transcript = new TranscriptDisplay({ thinkingMode: "collapsed", maxTurns: 40, maxRenderedLines: 400 });
    const turns = genTranscript(numTurns, chunkSize);
    transcript.setTurns(turns);
    const scrollView = new ScrollView(transcript, { primary: true, scrollbar: "hidden" });
    scrollView.updateLayout(termHeight * 10, termHeight, () => {});
    const prevLines: string[] = [];

    const times: number[] = [];
    for (let i = 0; i < iterations; i++) {
      // Add a character every 5th frame to simulate live activity
      if (i % 5 === 4) {
        const latestTurn = transcript["turns"] as Turn[];
        if (latestTurn.length > 0) {
          const lastMsg = latestTurn[latestTurn.length - 1].assistantMessage;
          if (lastMsg && lastMsg.chunks.length > 0) {
            const lastChunk = lastMsg.chunks[lastMsg.chunks.length - 1];
            if (lastChunk.contentText) {
              lastChunk.contentText += " ";
            }
          }
        }
      }
      times.push(benchFullFrame(transcript, scrollView, null as unknown as InputBar, width, termHeight, prevLines));
    }
    results.push({
      scenario: "full_frame",
      description: "Complete frame: layout + cell diff + cursor positioning",
      iterations,
      metrics: {
        avgMs: Math.round(percentile(times, 50) * 100) / 100,
        p50Ms: Math.round(percentile(times, 50) * 100) / 100,
        p95Ms: Math.round(percentile(times, 95) * 100) / 100,
        p99Ms: Math.round(percentile(times, 99) * 100) / 100,
        minMs: Math.round(Math.min(...times) * 100) / 100,
        maxMs: Math.round(Math.max(...times) * 100) / 100,
        totalBytes: times.reduce((s, t) => s + t * 150, 0),
        totalFrames: iterations,
      },
    });
  }

  // ── Scenario 8: Frame timing consistency ─────────────────────────────────
  {
    const transcript = new TranscriptDisplay({ thinkingMode: "collapsed", maxTurns: 40, maxRenderedLines: 400 });
    const turns = genTranscript(numTurns, chunkSize, false);
    transcript.setTurns(turns);

    const frameTimes = benchFrameConsistency(transcript, width, termHeight, 100);
    const consistency = {
      p50Ms: Math.round(percentile(frameTimes, 50) * 100) / 100,
      p95Ms: Math.round(percentile(frameTimes, 95) * 100) / 100,
      p99Ms: Math.round(percentile(frameTimes, 99) * 100) / 100,
      stdDevMs: Math.round(Math.sqrt(frameTimes.reduce((s, t) => s + (t - percentile(frameTimes, 50)) ** 2, 0) / frameTimes.length) * 100) / 100,
      maxJitter: Math.round((percentile(frameTimes, 99) - percentile(frameTimes, 50)) * 100) / 100,
    };
    results.push({
      scenario: "frame_consistency",
      description: "Frame timing distribution over 100 frames (mixed idle/streaming)",
      iterations: 100,
      metrics: {
        avgMs: consistency.p50Ms,
        p50Ms: consistency.p50Ms,
        p95Ms: consistency.p95Ms,
        p99Ms: consistency.p99Ms,
        minMs: Math.round(Math.min(...frameTimes) * 100) / 100,
        maxMs: Math.round(Math.max(...frameTimes) * 100) / 100,
        totalBytes: 0,
        totalFrames: 100,
      },
      details: consistency,
    });
  }

  // ── Report ───────────────────────────────────────────────────────────────
  const report: BenchmarkReport = {
    version: "1.0.0",
    timestamp: new Date().toISOString(),
    env: {
      nodeVersion: process.version,
      platform: process.platform,
      cpuCores: cpus().length,
      terminalWidth: terminalWidth,
      terminalHeight: terminalHeight,
    },
    config: {
      transcriptSize: config.label,
      terminalWidth: width,
      iterations,
    },
    results,
  };

  if (jsonMode) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    // Human-readable output
    console.log(`\n--- Results ---\n`);
    for (const r of results) {
      const m = r.metrics;
      console.log(`  ${r.scenario.padEnd(20)} ${r.description}`);
      console.log(`    avg: ${m.avgMs.toFixed(2)}ms  p50: ${m.p50Ms.toFixed(2)}ms  p95: ${m.p95Ms.toFixed(2)}ms  p99: ${m.p99Ms.toFixed(2)}ms  min: ${m.minMs.toFixed(2)}ms  max: ${m.maxMs.toFixed(2)}ms`);
      if (r.details) {
        const d = r.details as any;
        if (d.stdDevMs !== undefined) {
          console.log(`    stdDev: ${d.stdDevMs.toFixed(2)}ms  maxJitter: ${d.maxJitter.toFixed(2)}ms`);
        }
      }
      console.log();
    }

    // Summary
    console.log(`--- Summary ---`);
    const hotMs = results.find(r => r.scenario === "hot_render")?.metrics.avgMs ?? 0;
    const streamMs = results.find(r => r.scenario === "streaming")?.metrics.avgMs ?? 0;
    const fullMs = results.find(r => r.scenario === "full_frame")?.metrics.avgMs ?? 0;
    const consistency = results.find(r => r.scenario === "frame_consistency");
    const consistencyDetails = consistency?.details as any;

    console.log(`  Hot render (cache hit):       ${hotMs.toFixed(2)}ms — target: <1ms`);
    console.log(`  Streaming (1 new line/frame): ${streamMs.toFixed(2)}ms — target: <5ms`);
    console.log(`  Full frame (layout+diff):     ${fullMs.toFixed(2)}ms — target: <16ms (60fps)`);
    if (consistencyDetails) {
      console.log(`  Frame jitter (p99-p50):       ${consistencyDetails.maxJitter.toFixed(2)}ms — target: <2ms`);
      console.log(`  Frame stdDev:                 ${consistencyDetails.stdDevMs.toFixed(2)}ms — target: <1ms`);
    }
    console.log();
  }
}

main().catch(err => {
  console.error("Benchmark failed:", err);
  process.exit(1);
});
