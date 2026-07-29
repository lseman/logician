# Clickable Sections — TUI Implementation Plan

## Problem

Terminal TUIs have no DOM — no click handlers, no hover events, no element queries. We need to map mouse coordinates to logical UI elements and dispatch actions.

## Two-Layer Design

### Layer 1: OSC 8 Hyperlinks (decorative)

Wraps text in `ESC]8;;uriESC\textESC]8;;ESC\`. Terminal shows cursor change on hover. Does NOT fire click events — modern terminals may open external apps on click.

**Use for:** file paths (open in external editor), URLs, model names (link to docs).

### Layer 2: Click Coordinate Mapping (interaction)

Uses the existing SGR mouse protocol infrastructure (`enableMouse`/`disableMouse` in TUI). Mouse clicks arrive as `\x1b[<button;col;row(M)`.

## Architecture

### Click Target Registration

```typescript
// Registered by components during render, consumed by TUI on click
export interface ClickTarget {
  row: number;          // terminal row (0-indexed)
  col: number;          // start column (visible, not including ANSI codes)
  width: number;        // visible width of clickable area
  action: ClickAction;  // what to do on click
  component: Component; // who owns this target
}

export type ClickAction =
  | { type: 'expand'; payload?: unknown }
  | { type: 'collapse'; payload?: unknown }
  | { type: 'toggle'; payload?: unknown }
  | { type: 'focus'; payload?: unknown }
  | { type: 'navigate'; payload?: unknown }
  | { type: 'custom'; name: string; payload?: unknown };
```

### Component Interface Extension

```typescript
export interface Component {
  render(width: number): string[];
  invalidate?(): void;

  // NEW: register clickable regions during render.
  // Called by TUI after render, before dispatch.
  // Components push targets into the array; TUI owns the lookup.
  registerClickTargets?(targets: ClickTarget[]): void;
}
```

### TUI Integration

```
Render cycle:
  1. Each component calls render(width) → string[]
  2. TUI collects rendered lines for the frame
  3. TUI calls registerClickTargets() on each component (if defined)
  4. TUI builds a row-indexed lookup: ClickTarget[]
  5. Mouse event arrives → TUI looks up targets by (row, col)
  6. TUI dispatches action → component handles it
  7. requestRender() → full re-render
```

### Mouse Event Handling

SGR mouse protocol format: `\x1b[<button;col;row(M|m)`

- Button 0 = click down (press)
- Button 1 = press + move (drag)
- Button 2 = click up (release)
- Button 64 = wheel up
- Button 65 = wheel down

We handle button 0 (press) for clicks. Wheel already works (scroll). Drag could add text selection later.

```typescript
// In TUI.handleInput():
if (data.startsWith("\x1b[")) {
  // Check for SGR mouse event
  const m = data.match(/\x1b\[<(\d+);(\d+);(\d+)[Mm]/);
  if (m) {
    const [_, btnStr, colStr, rowStr] = m;
    const btn = parseInt(btnStr, 10);
    const col = parseInt(colStr, 10);
    const row = parseInt(rowStr, 10);

    // Only handle click-down (button 0)
    if (btn === 0) {
      const target = this.lookupClickTarget(row - 1, col - 1);
      if (target) {
        target.component.handleMouseClick?.(target.action);
        this.requestRender();
        return;
      }
    }
  }
}
```

## Component-by-Component Click Targets

### TranscriptDisplay

| Element | Click Action |
|---------|-------------|
| Tool call title | Toggle expand/collapse |
| Thinking block header | Toggle expand/collapse |
| Code fence language | Copy language name |
| File path in tool args | Open in external editor (OSC 8 only) |

### StatusBar

| Element | Click Action |
|---------|-------------|
| Phase indicator | Cycle phases |
| Model name | Open model selector |
| Context gauge | Toggle context details |
| Git branch | Open session manager |
| MCP count | Open MCP manager |
| Thinking level | Cycle thinking levels |

### TodoBar

| Element | Click Action |
|---------|-------------|
| Task item | Expand to show description |
| Status mark | Cycle status (pending → in_progress → completed) |
| Blocked-by count | Show dependency tree |

### WorkSurface

| Element | Click Action |
|---------|-------------|
| File names | Open file in transcript view |
| Evidence count | Expand evidence details |

### NotificationCenter

| Element | Click Action |
|---------|-------------|
| Notification | Dismiss |

## SGR Button Codes Reference

| Code | Meaning | Our Use |
|------|---------|---------|
| 0 | Button press (click down) | **Click action** |
| 1 | Press + move (drag) | Future: text selection |
| 2 | Button release (click up) | Redundant with 0 |
| 3 | Release (drag) | Future: text selection |
| 64 | Wheel up | Already handled (scroll) |
| 65 | Wheel down | Already handled (scroll) |
| 128 | Shift + wheel up | Future: faster scroll |
| 129 | Shift + wheel down | Future: faster scroll |
| 256 | Ctrl + wheel up | Future: page scroll |
| 257 | Ctrl + wheel down | Future: page scroll |

## Implementation Phases

### Phase 1: Core Infrastructure

**Files:**
- `tui/packages/tui/src/layers/core/click-targets.ts` — ClickTarget type + lookup
- `tui/packages/tui/src/layers/core/tui-core.ts` — integrate into TUI class

**Changes:**
- Add `ClickTarget` and `ClickAction` types
- Add `registerClickTargets()` method to Component interface
- Add `clickTargets: ClickTarget[]` array to TUI
- Add `lookupClickTarget(row, col): ClickTarget | null`
- Extend `handleInput()` to parse SGR button 0 events
- Wire mouse click → lookup → dispatch → render

**No behavior changes** — just infrastructure. No existing components need to register targets.

**Test impact:** New tests for click target lookup. Zero changes to existing tests.

### Phase 2: Transcript Click Targets

**Files:**
- `tui/packages/tui/src/components/transcript-display.ts` — register tool/thinking click targets

**Changes:**
- TranscriptDisplay registers ClickTarget for each tool call row
- TranscriptDisplay registers ClickTarget for each thinking block header
- `handleMouseClick(action)` toggles expand/collapse state
- Re-renders with expanded/collapsed content

**Test impact:** Update transcript tests for expanded state rendering.

### Phase 3: Status Bar Click Targets

**Files:**
- `tui/packages/tui/src/components/status-bar.ts` — register clickable sections

**Changes:**
- Status bar registers ClickTarget for each section (model, context, git, etc.)
- `handleMouseClick(action)` dispatches to appropriate handler
- Model click → open model selector
- Context click → toggle details
- Git branch click → open session manager

**Test impact:** Update status bar tests.

### Phase 4: Todo Bar & Other Components

**Files:**
- `tui/packages/tui/src/components/todo/todo-bar.ts`
- `tui/packages/tui/src/components/work-surface.ts`
- `tui/packages/tui/src/components/notification-center.ts`

**Changes:**
- Todo bar: status mark cycles, task item expands
- Work surface: file names clickable
- Notification center: click to dismiss

**Test impact:** New tests for each component.

### Phase 5: Drag Selection (optional)

**Files:**
- `tui/packages/tui/src/layers/core/text-selection.ts` — new file

**Changes:**
- Handle button 1 (drag) events
- Track selection start → end coordinates
- Render selection highlight (reverse video)
- Copy selected text on Ctrl+C

**Test impact:** New tests for selection.

## Mouse Event Format

SGR (enabled with `\x1b[?1006h` — already in TUI):

```
\x1b [ < button ; col ; row ( M | m )
```

- `button`: 0=click, 1=drag, 2=release, 64=wheel up, 65=wheel down
- `col`: mouse column (1-indexed)
- `row`: mouse row (1-indexed)
- `M`: mouse down/move, `m`: mouse up

Example: `\x1b[<0;25;12M` = click at column 25, row 12.

## Edge Cases

### Scrolled transcript

Mouse row coordinates are relative to the terminal viewport. Translated to transcript row by subtracting the transcript's top offset:

```typescript
const transcriptRow = mouseRow - transcriptTopRow + scrollOffset;
```

### Overlays

When an overlay is visible, clicks inside the overlay are handled by the overlay component. Clicks outside the overlay (on transcript) are ignored until the overlay closes.

### Narrow terminals

At <80 cols, some status bar sections may wrap or truncate. Click targets only registered for visible portions.

### OSC 8 vs click layer

OSC 8 hyperlinks are rendered as part of the text content. They don't conflict with click targets — OSC 8 is just text with special escape sequences. Click targets are registered separately by the component.

## Dependencies

- **Terminal support:** SGR mouse protocol (1006) — supported by all modern terminals (iTerm2, Alacritty, Kitty, WezTerm, VSCode terminal, tmux with mouse mode)
- **OSC 8 support:** Most modern terminals (not all)
- **Existing TUI:** Already enables SGR mouse (`enableMouse()`), already handles wheel events

## Testing Strategy

1. **Unit tests:** Click target registration and lookup
2. **Integration tests:** Component → TUI click dispatch
3. **Width tests:** Click targets at 80, 120, 160, 200 cols
4. **Overlay tests:** Clicks ignored when overlay visible
5. **Scroll tests:** Click coordinates translated correctly
