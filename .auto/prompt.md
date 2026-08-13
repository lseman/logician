# Autoresearch: Full-Screen TUI Keystroke Latency

## Objective
Reduce the steady-state delay between a keystroke and the completed TUI frame while the 120x40 screen is full of transcript content. Keep latency bounded as the off-screen transcript grows. The benchmark follows the real `InputBar.handleInput()` -> layout/render -> frame-diff path used by the application.

## Metrics
- **Primary**: `keystroke_full_p50_ms` (ms, lower is better) — median across independent benchmark processes of p50 end-to-end keystroke latency with 150 mixed-content turns and a full 120x40 viewport.
- **Tail guardrails**: `keystroke_full_p95_ms` and `keystroke_full_p99_ms` (ms, lower is better).
- **Scaling guardrail**: `keystroke_backlog_scaling_x` (ratio, lower is better) — 2500-turn p50 divided by empty-transcript p50. This catches accidental work proportional to off-screen history.
- **Diagnostic buckets**: `keystroke_empty_p50_ms` and `keystroke_marathon_p50_ms`.

Treat improvements smaller than 5% as noise unless they reproduce over more runs. Reject changes that improve p50 by moving work into visible tail spikes, materially worsen the scaling ratio, or change rendered output.

## How to Run
`./.auto/measure.sh` — runs three fresh benchmark processes and outputs only `METRIC name=value` lines. Set `AUTORESEARCH_RUNS=5` for confirmation runs.

For a human-readable diagnostic sweep, run:
`cd apps/tui && COLUMNS=120 LINES=40 npx --no-install tsx src/__tests__/benchmark-keystroke.ts`

## Files in Scope
- **`apps/tui/src/terminal/core.ts`** — TUI rendering engine, frame pacing, diff+write pipeline (1557 lines). The hot path: `doRender()` → `_doRenderInner()` → `_commitFrame()`. Every frame allocates line arrays and calls `visibleWidth()` on every cell.
- **`apps/tui/src/rendering/layout.ts`** — Flexbox constrained layout engine called every frame (675 lines). Key functions: `renderLayoutFrame()`, `paintBox()`.
- **`apps/tui/src/rendering/transcript/display.ts`** — Transcript rendering with per-turn cache. The render loop iterates all turns, checks revisions, rebuilds dirty turns via markdown parsing and text wrapping (4116 lines total for the display module). Key hot path: `render()`, `turnRevisionFor()`, `buildTurnLines()`.
- **`apps/tui/src/terminal/primitives.ts`** — TUI line primitives including `clampLineToWidth()`, `compositeTuiLine()`, `CURSOR_MARKER`, etc. (425 lines).
- **`apps/tui/src/terminal/theme.ts`** — Theme system, called on every render for color lookups. (532 lines)
- **`apps/tui/src/rendering/flex.ts`** — Flex layout component used by the TUI root (295 lines).
- **`apps/tui/src/rendering/scroll-view.ts`** — ScrollView component (236 lines).
- **`apps/tui/src/input/input-bar.ts`** — keystroke handling and input rendering.
- **`apps/tui/src/__tests__/benchmark-keystroke.ts`** — benchmark harness; changes here must improve fidelity, not scores.
- **`tui/package.json`**, **`package.json`** — Dependencies and scripts.

## Off Limits
- Do not change the TUI UI layout, colors, or visual appearance for performance gains.
- Do not remove features (tool cards, thinking display, markdown rendering, overlays).
- Do not add new dependencies.
- Do not change the frame pacing interval (60fps) unless there's a clear win that preserves responsiveness.
- Do not special-case benchmark data, reduce benchmark workload, or weaken percentile reporting.

## Constraints
- Must pass typecheck: `bun run typecheck` in tui/
- Must pass lint: `bun run lint` in root
- Must pass existing tests in `packages/*/src/__tests__/`
- All changes must be strictly performance-oriented — no refactors without measurable benefit.
- Preserve keystroke behavior and byte-for-byte rendered frame semantics.
- The benchmark measures CPU-side handling through escape-sequence generation; it does not claim to measure OS terminal emulator paint time or physical keyboard input latency.

## What's Been Tried
_(Updated as experiments accumulate)_
