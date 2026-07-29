# Logician TUI Modernization Plan

## Current State

Logician TUI uses a single-column vertical layout with differential ANSI rendering:

```
┌─────────────────────────────────────────────────────────┐
│ Transcript (scrollable, full width)                     │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ User: Fix the bug                                    │ │
│ │ Agent: I'll check the auth middleware...             │ │
│ │ [tool: read_file src/auth.ts]                       │ │
│ │ ```diff                                              │ │
│ │ - if (token < expiry) return false                  │ │
│ │ + if (token <= expiry) return false                 │ │
│ │ ```                                                  │ │
│ └─────────────────────────────────────────────────────┘ │
│ ─────────────────────────────────────────────────────── │
│ Notifications / Todo Bar / Work Surface / Queue         │
│ ─────────────────────────────────────────────────────── │
│ › Ask Logician…                                          │
│ ─────────────────────────────────────────────────────── │
│ ⏸ ready | Qwen | think:off | dir logician | ⎇ main *18 │
└─────────────────────────────────────────────────────────┘
```

**Key facts:**
- Zero external dependencies — pure ANSI/VT100 rendering
- Component interface: `render(width): string[]` + `invalidate()`
- Differential rendering engine (16ms min frame interval)
- ~60 theme color tokens across 8 categories
- Box-drawing popup system with rounded corners
- 100+ tests tightly coupled to render output
- No layout system beyond single Container
- Popups hide transcript entirely
- Status bar: single info-dense line

---

## Design Goals

1. **Better information density** — use horizontal space more effectively
2. **Context visibility** — see code changes while reading agent output
3. **Reduced popup friction** — inline everything possible, overlays only when needed
4. **Progressive enhancement** — works at 80 cols, shines at 160+ cols
5. **Zero breaking changes** — all existing tests pass, Component interface preserved
6. **Configurable** — layout presets, not forced changes

---

## Proposed Layouts

### Layout A: Split Panel (wide terminals, ≥120 cols)

```
┌──────────────────────────┬──────────────────────────┐
│ Transcript               │ File Viewer / Sidebar    │
│ ┌──────────────────────┐ │ ┌────────────────────┐  │
│ │ User: Fix the bug     │ │ │ src/auth.ts        │  │
│ │ Agent: [thinking...]  │ │ │ - line 12          │  │
│ │ [tool: read_file]     │ │ │ + line 12'         │  │
│ │ ```diff               │ │ │                    │  │
│ │ - old                 │ │ │ [file tree]        │  │
│ │ + new                 │ │ │                    │  │
│ │ ```                   │ │ │                    │  │
│ └──────────────────────┘ │ └────────────────────┘  │
│ ─────────────────────────┴───────────────────────── │
│ Notifications / Todo / Work / Queue                 │
│ ─────────────────────────────────────────────────── │
│ › Ask Logician…                                     │
│ ─────────────────────────────────────────────────── │
│ ⏸ ready | Qwen | 49.4k/150k | ⎇ main *18 +1        │
└─────────────────────────────────────────────────────┘
```

### Layout B: Compact (narrow terminals, <120 cols)

Same as current — single column, no changes.

### Layout C: Minimal (for coding focus)

Transcript only, status bar collapsed to one line, no todo/work/queue visible.

---

## Implementation Phases

### Phase 1: Layout System (no breaking changes)

**Goal:** Add a generic layout system that can compose components horizontally and vertically without changing existing rendering.

**Changes:**
- Add `ContainerH` (horizontal container) alongside existing `Container` (vertical)
- Add `SplitPane` — two resizable sub-panes with divider
- Add `LayoutConfig` type for layout presets
- Add layout switcher command (`Ctrl+L` → cycle layouts, or new `Ctrl+Shift+L`)

**Scope:** New files only. No changes to existing components.

**Test impact:** New tests for layout system. Zero changes to existing tests.

**Estimated effort:** 2-3 days

---

### Phase 2: Sidebar Component

**Goal:** A collapsible sidebar showing file tree, sessions, or git status.

**Changes:**
- New `Sidebar` component with tab switching (files / sessions / git)
- Sidebar renders as right panel in Layout A
- Collapsible with `Ctrl+B` (toggle)
- File tree shows modified files with status icons
- Session browser in sidebar (replaces popup)

**Scope:** New files. Integration in `LogicianTUI.buildLayout()`.

**Test impact:** New tests for Sidebar. Zero changes to existing tests.

**Estimated effort:** 3-4 days

---

### Phase 3: Inline Diff Preview

**Goal:** Show diffs inline in the transcript without requiring a separate panel.

**Changes:**
- Enhance `TranscriptDisplay` to render diffs with side-by-side view in wide terminals
- Configurable: `diffStyle: "unified" | "side-by-side" | "inline"`
- Side-by-side only activates when terminal width > 120 cols
- Falls back to unified diff automatically

**Scope:** Changes to `transcript-display.ts` only. Backward compatible.

**Test impact:** Update existing transcript tests to match new diff rendering.

**Estimated effort:** 2-3 days

---

### Phase 4: Status Bar Redesign

**Goal:** Better visual status with progress indicators and clickable sections.

**Changes:**
- Replace single-line status bar with multi-row status area (2-3 rows)
- Visual context token gauge (progress bar)
- Phase indicator with color coding
- Clickable sections (CSI-u support)
- Still compact at narrow widths

**Scope:** Changes to `status-bar.ts` only.

**Test impact:** Update existing status bar tests.

**Estimated effort:** 2 days

---

### Phase 5: Popups → Inline Panels

**Goal:** Reduce popup friction by making common overlays inline.

**Changes:**
- Session browser: inline at bottom (toggleable) instead of popup
- Settings: inline panel instead of overlay
- Model/reasoner selector: inline dropdown in status bar
- Keep permission/choice popups as overlays (they're inherently interruptive)

**Scope:** New inline panel components. Popups remain for interruptive flows.

**Test impact:** New tests for inline panels. Existing popup tests unchanged.

**Estimated effort:** 3-4 days

---

### Phase 6: Theme Enhancements

**Goal:** Better visual hierarchy and modern aesthetics.

**Changes:**
- Add subtle background colors for different content types
- Better code block rendering with line numbers
- Smoother animation transitions (spinner improvements)
- High-contrast mode for accessibility
- Truecolor support validation

**Scope:** Changes to `theme.ts` and component rendering.

**Test impact:** Update existing render tests for color changes.

**Estimated effort:** 2 days

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Test breakage from layout changes | Layout system is opt-in; default layout unchanged |
| Performance regression | Differential rendering preserved; measure frame times |
| SSH terminal compatibility | Test on tmux, screen, plain SSH; no mouse/CSI-u required |
| Popup state management complexity | Keep overlay stack; inline panels use separate state |
| Horizontal space fragmentation | Sidebar auto-collapses at <120 cols |

---

## What NOT to Change

- **Component interface** — `render(width): string[]` stays
- **Differential rendering engine** — core optimization, untouched
- **Theme system** — color tokens stay, new ones added
- **Popup overlay system** — stays for interruptive flows
- **Input handling** — keyboard bindings preserved
- **Existing component behavior** — transcript, input bar, status bar all backward compatible

---

## Testing Strategy

1. **Integration tests first** — test layout composition at the TUI level
2. **Snapshot tests** — capture render output for each layout preset
3. **No component-level test changes** until Phase 3+
4. **Width regression tests** — ensure all layouts work at 80, 120, 160, 200 cols
5. **Performance benchmark** — frame render time < 16ms at all widths

---

## Quick Start for Implementers

```bash
cd tui
npm run typecheck  # verify baseline
npm test           # verify baseline tests pass

# After Phase 1:
npm test           # should pass — no existing tests changed
```

---

## Priorities

| Priority | Phase | Why |
|----------|-------|-----|
| P0 | 1 | Foundation for everything else |
| P1 | 2 | Biggest UX win — context visibility |
| P1 | 3 | Improves reading flow |
| P2 | 4 | Polish, not core |
| P2 | 5 | Convenience |
| P3 | 6 | Visual polish |
