# Autoresearch: TUI Latency

## Objective
Reduce the perceived latency of the Logician TUI — cold start time, frame render time during streaming, and layout/diff overhead. The goal is a snappy, responsive terminal interface that feels instantaneous even with large transcripts and complex tool output.

## Metrics
- **Primary**: `module_load_ms` (ms, lower is better) — median of 3 cold module loads via `node --experimental-strip-types`. Measures from process start to first import resolution.
- **Secondary**: `layout_time_ms` (ms), `render_time_ms` (ms), `diff_time_ms` (ms), `write_time_ms` (ms) — per-frame timing breakdowns when rendering a realistic transcript snapshot (30 turns, 60 lines each, mixed content types).

## How to Run
`./.auto/measure.sh` — outputs `METRIC name=value` lines.

## Files in Scope
- **`tui/packages/tui/src/terminal/core.ts`** — TUI rendering engine, frame pacing, diff+write pipeline (1557 lines). The hot path: `doRender()` → `_doRenderInner()` → `_commitFrame()`. Every frame allocates line arrays and calls `visibleWidth()` on every cell.
- **`tui/packages/tui/src/rendering/layout.ts`** — Flexbox constrained layout engine called every frame (675 lines). Key functions: `renderLayoutFrame()`, `paintBox()`.
- **`tui/packages/tui/src/rendering/transcript/display.ts`** — Transcript rendering with per-turn cache. The render loop iterates all turns, checks revisions, rebuilds dirty turns via markdown parsing and text wrapping (4116 lines total for the display module). Key hot path: `render()`, `turnRevisionFor()`, `buildTurnLines()`.
- **`tui/packages/tui/src/terminal/primitives.ts`** — TUI line primitives including `clampLineToWidth()`, `compositeTuiLine()`, `CURSOR_MARKER`, etc. (425 lines).
- **`tui/packages/tui/src/terminal/theme.ts`** — Theme system, called on every render for color lookups. (532 lines)
- **`tui/packages/tui/src/rendering/flex.ts`** — Flex layout component used by the TUI root (295 lines).
- **`tui/packages/tui/src/rendering/scroll-view.ts`** — ScrollView component (236 lines).
- **`tui/packages/tui/src/index.ts`** — Entry point with trust prompt, session loading, bridge initialization. Cold start heavy due to all the auth/trust/bridge setup.
- **`tui/packages/coding-agent/`** — AgentCoreBridge, trust store, session management. Heavy imports in index.ts.
- **`tui/package.json`**, **`package.json`** — Dependencies and scripts.

## Off Limits
- Do not change the TUI UI layout, colors, or visual appearance for performance gains.
- Do not remove features (tool cards, thinking display, markdown rendering, overlays).
- Do not add new dependencies.
- Do not change the frame pacing interval (60fps) unless there's a clear win that preserves responsiveness.

## Constraints
- Must pass typecheck: `bun run typecheck` in tui/
- Must pass lint: `bun run lint` in root
- Must pass existing tests in `tui/packages/*/src/__tests__/`
- All changes must be strictly performance-oriented — no refactors without measurable benefit.

## What's Been Tried
_(Updated as experiments accumulate)_
