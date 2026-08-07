# TUI performance investigation vs. pi

Working notes from a session spent chasing "our TUI feels slower than pi" (pi = `~/logician/repos/pi`, comparing against its **fullscreen** renderer, `packages/tui/src/tui-alt-screen.ts` — not `tui-main-screen.ts`, the flat/no-layout-tree regular mode).

## Fixed this session (confirmed real, landed)

1. **Rendering crash → blank screen.** A stray module-level layout-frame cache in `rendering/layout.ts` compared `cachedLineCount === cachedFrame.lines.length` — always true against itself, so the fast path returned the very first frame forever after startup. Removed. This was the "stuff stopped rendering" regression, unrelated to the perf investigation below but discovered while chasing it.

2. **`visibleWidth` (terminal/primitives.ts)** — ported pi's ASCII fast path (`isPrintableAscii` → `str.length`, skip the regex/`string-width` call entirely) plus a persistent 512-entry FIFO cache for the non-ASCII slow path. Was: unconditional double regex `.replace()` + `string-width` call on every invocation, no cross-frame cache. Measured ~190× faster on mixed ANSI+text content in isolation.

3. **`compositeTuiLine` (terminal/primitives.ts)** — ported pi's single-pass before/overlay/after extraction for ASCII-only lines (falls back to the original multi-pass `clampLineToWidth`/`skipColumns`/`visibleWidth`-heavy path for anything with wide characters, since we deliberately did not port pi's `get-east-asian-width`-based Unicode tables). This one took real iteration — the reference implementation is two *independent* scans (`clampLineToWidth` for "before", `skipColumns` for "after") with different, subtle boundary rules for trailing ANSI codes sitting exactly at a truncation column. Verified against a 2000-trial fuzz test plus 12 hand-picked edge cases (`src/__tests__/composite-tui-line-fast-path.test.ts`).

4. **`TranscriptDisplay.render()` O(all turns) → O(changed turns).** Was re-fingerprinting and re-splicing every cached turn's lines into a fresh array on every call, including every 150ms spinner tick during streaming — even though the per-turn cache (`renderTurn`) was already a correct cache-hit. Added an "assembled prefix" cache: reuse the previously-spliced array up to the first turn whose identity+revision actually changed.

5. **Five leaf components had no self-cache at all** (`Separator`, `NotificationCenter`, `WorkSurface`, `SteerQueue`, `NewOutputIndicator`) — rebuilt their line arrays from scratch on every single frame regardless of whether their own state changed, unlike `StatusBar`/`TodoBar`/`InputBar`/`TranscriptDisplay` which already had the `cachedWidth`/`cachedLines` pattern. Added matching caches to all five (revision counter for `WorkSurface`, since it has many independent mutation points; direct value/ref comparison for the rest).

6. **Added a cross-frame leaf cache in `layoutComponent`** (`rendering/layout.ts`) — when a leaf's own `render()` returns the same array reference as last frame (guaranteed now that #5 is fixed) *and* its layout inputs (x/y/width/height/clip) are unchanged, skip recomputing `allocatedHeight`/`lineOffset`/`clipRect`/`allocateBox` and reuse the previous `LayoutBox`. Explicitly *not* the same bug as #1 — keyed per-component via `WeakMap`, not a single global slot compared against itself.

Net effect of #2–#6, measured via a synthetic 40-turn-transcript keystroke benchmark: `layoutComponent`'s own cost dropped to ~0.02–0.09ms (was the dominant cost before, per an earlier agent's own profiling note that used to sit in a comment here: "24–27ms, ~95% of frame time"). Full test suite green (237/238; the 1 failure is a pre-existing flaky real-pty timing test, confirmed flaky on a clean `main` checkout too, unrelated to any of this).

## Further fixes (flat render cache + single-pass composite)

7. **`renderCache` flat + bounded (layout.ts).** Old design used a nested
   `Map<Component, Map<number, string[]>>` — two hash lookups per access
   plus per-frame `new Map()` that never discarded old inner Maps. After
   N steady-state frames the old inner Maps (and their string-array
   references) accumulated, creating heap pressure that could trigger
   periodic GC. Replaced with a flat `Map<string, string[]>` keyed by
   `"${componentId}|${width}"` (single hash lookup), size-limited to
   2048 entries with LRU-style eviction on hit. Each frame starts with
   a fresh empty cache so no stale data persists across frames.
   **Measured:** layout engine avg dropped from ~7–8ms to ~5.6ms.

8. **Single-pass ASCII composite (primitives.ts).** The old path ran
   `isAsciiOnlyLine()` to scan the entire base line, then called
   `compositeTuiLineAsciiFast()` which scanned the same base line again
   to build before/after segments — a full double-pass on every call.
   Combined into one `compositeTuiLineAsciiSingle()` function that
   verifies ASCII-only while simultaneously building the output; if
   non-ASCII is encountered it returns `null` so the caller falls
   through to the generic path. Eliminates ~30 % of the work for the
   common ASCII-only case.

## Root cause found: `graphemeAt()` was O(n²) via a fresh `Intl.Segmenter` pass per character

The "moving spike" (extra ~13–20ms landing in whichever function happened to be running — `compositeTuiLine`, `paintScrollbar`, even `layoutComponent`) was real but was **the wrong thing to chase under Node**. Two breakthroughs, in order:

**1. The benchmark runtime was wrong.** All profiling up to this point ran the benchmark under `node`/`tsx` (`#!/usr/bin/env node`). The actual app launches via `bun run`. Re-running the identical benchmark with `bun` directly instead of `npx tsx`/`node`:

| | Node/tsx | bun |
|---|---|---|
| ours, steady state avg | ~16ms, spikes every other frame | ~6.5ms, **zero spikes** |
| pi, steady state avg | ~1.4ms, zero spikes | ~0.47ms, zero spikes |

The periodic spike **only exists under Node**. `perf stat -e task-clock,page-faults` showed the process was genuinely CPU-bound the whole time (task-clock ≈ wall-clock) even though V8's own `--cpu-prof` sampler reported 99.9% "idle" — a real Node/tsx-specific profiling+scheduling artifact, most likely interacting with `tsx`'s on-the-fly transform/module-loader machinery. Not investigated further since it doesn't affect the shipped runtime. **Lesson: always benchmark under the runtime the app actually ships with.**

**2. Under bun, a real and large gap remained** (ours ~6.5ms vs pi's ~0.47ms, ~14×, no spikes — clean signal). Used bun's built-in `--cpu-prof --cpu-prof-md` (writes an LLM-friendly markdown profile) and found the entire gap concentrated in one place:

```
50.5%  segment            [native code]   (Intl.Segmenter)
38.7%  [Symbol.iterator]  [native code]
 6.2%  next               [native code]
 1.2%  graphemeAt         primitives.ts:95
 0.8%  graphemeAt         primitives.ts:97
```
97.6% of total time traced back to `clampLineToWidth` → `graphemeAt`.

**The bug** (`terminal/primitives.ts`, `graphemeAt`): called once per character by `clampLineToWidth`'s scanning loop. Each call did `text.slice(offset)` (fresh substring allocation) then ran a **new `Intl.Segmenter.segment()` pass over that entire remaining substring** just to extract the *first* grapheme. For an *n*-character line this is O(n²) — an ever-shrinking suffix re-segmented from scratch on every step — and `Intl.Segmenter` invocation itself is expensive (locale-aware Unicode text segmentation, not a cheap call). `visibleWidth` and `compositeTuiLine` already had ASCII fast paths that bypassed this (landed earlier this session, #2 and #3 above); `clampLineToWidth` — called from both of them *and* directly from `paintScrollbarCell` and elsewhere — had no such fast path and always paid the grapheme-by-grapheme cost, even for 100%-ASCII lines (the overwhelming majority of rendered content).

**The fix**: in `clampLineToWidth`'s scanning loop, check `charCodeAt(i)` is in the printable-ASCII range (`0x20`–`0x7e`) first — if so, treat it as exactly one column and advance one char, matching what `isPrintableAscii`/`isAsciiOnlyLine` already assume elsewhere in this file. Only fall through to `graphemeAt`/`Intl.Segmenter` for anything outside that range (wide chars, combining marks, emoji — where real grapheme-cluster handling is actually needed).

**Verified correct**: 5000-trial differential fuzz test comparing the new fast-path output against the original unconditional-`graphemeAt` reference implementation, across random ASCII text + tabs + interspersed ANSI color codes + random widths — 0 mismatches. Full test suite green (237/238, same pre-existing flaky pty test). pty regression suite (real terminal, real keystrokes) green.

**Measured effect** (bun, same 2000-iteration steady-state benchmark):
- Before: avg 6.5ms/frame, CPU profile total 13.55s for 2000 frames, 97.6% of time in `Intl.Segmenter`.
- After: avg **0.079ms/frame** — **82× faster**, profile total 190.5ms for the same 2000 frames, `Intl.Segmenter` cost gone from the hot path entirely. Now *faster* than pi's ~0.47ms average in this synthetic benchmark.

This appears to be the actual, dominant, real-world explanation for "our TUI feels slower than pi" — not the layout engine, not GC, not allocation patterns (the earlier `renderCache`/leaf-cache work in #4–#8 was real and worth keeping, but was chasing a much smaller effect next to this one).

### Ruled out along the way (for the record, since the Node-side investigation was extensive before the runtime mismatch was caught)

- JIT compilation/deopt — persisted under `--jitless`.
- GC — `--trace-gc` showed only ~0.2–0.3ms scavenge pauses, uncorrelated with spike timing.
- Per-frame `Map` allocation in `renderLayoutFrame` — spike persisted with persistent, cleared Maps instead of fresh ones.
- CPU core migration — spike persisted pinned to a single core via `taskset -c 0`.
- System-wide load/throttling — none present (`nr_throttled=0`, load average <0.25/32 cores).
- Our own `leafCache` addition — spike persisted with it force-disabled.
- Environment-wide effect — a comparable-cost Python loop on the same machine showed zero periodic anomalies at the same moment.

All of the above were true statements about the Node/tsx-specific artifact, which turned out to be a red herring once the benchmark was run under the actual shipping runtime (bun).

## Benchmark scripts

None were kept — all were scratch files under `/tmp/.../scratchpad/` and deleted at the end of each investigation phase per this session's convention. If revisiting, the shape to reconstruct:

```ts
// Synthetic FakeTranscript: 40 turns, ~1100 lines total, self-caches on
// (width) like TranscriptDisplay does — cachedWidth/cachedLines fields,
// return the cached array by reference when width is unchanged.
// Synthetic FakeInput/FakeText: same cachedWidth/cachedLines pattern,
// additionally keyed on `value` for FakeInput.
// Wire them into the real Flex/ScrollView (ours) or VStack/ScrollView
// (pi's) in the same shape as app/tui.ts's buildLayout(): root Flex/VStack
// of [transcriptScrollView, dock], dock = Flex/VStack of
// [separator, pinnedContainer, inputBar, separator, statusPanel].
// Warm up 10 iterations, then time 15+ consecutive renderLayoutFrame()
// calls with zero content changes between them ("steady state" — isolates
// the mechanism from any content-diffing cost).
```

Getting pi's `layout.ts` to run standalone required a `node_modules/get-east-asian-width` symlink inside `repos/pi/packages/tui/` pointing at the copy already resolvable from `~/logician/node_modules/.bun/node_modules/get-east-asian-width` (pi's own repo has no installed dependencies) — remove after use, don't commit it.

**Run benchmarks with `bun <script>.ts` directly, not `npx tsx`/`node`** — the Node-side periodic-spike investigation above cost most of a session chasing an artifact that doesn't exist under the actual shipping runtime.

For CPU profiling, `bun --cpu-prof --cpu-prof-md --cpu-prof-name=<name> <script>.ts` writes `<name>.md` (LLM-friendly self-time table, sorted, with file:line) and `<name>.cpuprofile` (raw, loadable in Chrome DevTools) — this is what actually found the `graphemeAt`/`Intl.Segmenter` bug. `perf stat -e task-clock,page-faults,minor-faults <bun-or-node> <script>` is useful for a quick sanity check on whether a slowdown is CPU-bound work vs. something else (page faults from heavy allocation showed up clearly there before the profiler pinned down the exact function).
