# Logician Ink TUI

An Ink-based terminal TUI for Logician — React for CLI.

## Overview

This is a separate Ink-based TUI implementation for Logician, coexisting alongside the original custom-terminal TUI in `apps/tui/`. It uses:

- **Ink 7** — React for CLI
- **React 19** — UI library
- **Bun** — Package manager and runtime

## Features

- 📝 Transcript display with thinking mode support
- ⌨️ Input bar with keyboard shortcuts
- 📊 Status bar with model, git, and session info
- 🎨 Theme system (dark/light)
- 📋 Session management
- 🤖 Model selector
- ⚙️ Settings panel
- 🔌 Plugin manager
- 📡 MCP server manager
- 🧠 Reasoner selector
- 📬 Steer queue manager
- 🔄 Autoresearch dashboard
- 🧠 Thinking level selector
- 🎯 Inference mode selector
- 🌳 Session tree view

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Enter` | Submit message / confirm selection |
| `Escape` | Close overlay (or interrupt the running turn) |
| `Ctrl+C` | Exit TUI |
| `←` `→` `Ctrl+A` `Ctrl+E` | Move cursor / home / end |
| `Ctrl+U` | Clear input to cursor (pushes to kill ring) |
| `Ctrl+W` | Delete previous word (pushes to kill ring) |
| `Ctrl+K` | Kill to end of line (pushes to kill ring) |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` or `Ctrl+Shift+Z` | Redo |
| `Ctrl+Shift+V` or `Ctrl+_` | Paste from kill ring |
| `Ctrl+←` `Ctrl+→` | Jump by word (left / right) |
| `Ctrl+K` | Kill to end of line |
| `PageUp` / `Ctrl+U` | Scroll up in transcript |
| `PageDown` / `Ctrl+D` | Scroll down (follow mode) |
| `Home` | Toggle follow mode (auto-scroll) |
| `End` | Disable follow mode (scroll manually) |
| `/` | Slash commands (real runtime registry) |
| `@` | File-mention autocomplete |
| `Shift+Tab` | Toggle act / plan workflow mode |
| `Ctrl+S` | Session manager |
| `Ctrl+P` | Model selector |
| `Ctrl+T` | Theme selector |
| `Ctrl+R` | Reasoner selector |
| `Ctrl+A` | Autoresearch dashboard |
| `Ctrl+Q` | Steer-queue manager |
| `Ctrl+O` | Session tree |
| `Ctrl+L` | Settings |
| `Ctrl+G` | File mention |
| `Ctrl+B` | Thinking-level selector |
| `Ctrl+K` | Inference-mode selector |
| `Ctrl+Y` | Plugin manager |

> `Ctrl+M`, `Ctrl+I`, `Ctrl+H`, `Ctrl+J` are indistinguishable from
> `Enter` / `Tab` / `Backspace` / `LF` in a raw terminal and are therefore not
> bound — reach those panels via slash commands (`/model`, `/thinking-steps`, …).

## Running

```bash
# Development
bun run dev

# Production
bun run start
```

**Note:** The Ink TUI requires a real terminal (TTY/PTY) to function. Running via `bun run` in a non-PTY environment will show "Raw mode is not supported" errors. Use a proper terminal emulator to run the TUI.

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LOGICIAN_MODEL` | `local` | Model to use |
| `LOGICIAN_THINKING_LEVEL` | `off` | Thinking level |
| `LOGICIAN_INFERENCE_MODE` | `none` | Inference mode |
| `LOGICIAN_WORKFLOW_MODE` | `act` | Workflow mode (act/plan) |
| `LOGICIAN_EXECUTION_PROFILE` | `minimal` | Execution profile |
| `LOGICIAN_THEME` | `default` | Theme name |
| `LOGICIAN_RTK_PROXY` | `false` | Enable RTK proxy |
| `LOGICIAN_GRAPHICIAN` | `true` | Enable Graphician |
| `LOGICIAN_FFFGREP` | `true` | Enable fffgrep |

## Project Structure

```
apps/tui-ink/
├── src/
│   ├── index.tsx           # Entry point
│   ├── types.ts            # Shared types
│   ├── theme.ts            # Theme system
│   ├── state.ts            # State management
│   ├── utils.ts            # Utilities
│   ├── components/         # Ink components
│   │   ├── App.tsx         # Main app component
│   │   ├── TranscriptDisplay.tsx
│   │   ├── InputBar.tsx
│   │   └── StatusBar.tsx
│   └── overlays/           # Overlay popups
│       ├── SlashPopup.tsx
│       ├── ChoicePopup.tsx
│       ├── PermissionPopup.tsx
│       ├── SessionManager.tsx
│       ├── ModelSelector.tsx
│       ├── ThemeSelector.tsx
│       ├── SettingsSelector.tsx
│       ├── PluginManager.tsx
│       ├── McpManager.tsx
│       ├── ReasonerSelector.tsx
│       ├── QueueManager.tsx
│       ├── AutoresearchDashboard.tsx
│       ├── ThinkingLevelSelector.tsx
│       ├── InferenceModeSelector.tsx
│       └── SessionTree.tsx
├── package.json
└── tsconfig.json
```

## Differences from Original TUI

| Feature | Original TUI (`apps/tui/`) | Ink TUI (`apps/tui-ink/`) |
|---------|---------------------------|---------------------------|
| Framework | Custom ANSI terminal | Ink (React for CLI) |
| Rendering | Manual ANSI escape codes | React component tree |
| State Management | Class-based | React hooks + EventEmitter |
| Layout | Custom flex system | Ink Box flexbox |
| Overlays | Custom terminal widgets | React components |
| Input Handling | Raw stdin mode | Ink useInput hook |
| Dependencies | Minimal | React, Ink, ansi-styles |

## Development

```bash
# Install dependencies
bun install

# Type check
bun run typecheck

# Run in development mode
bun run dev
```

## Status

**Functional MVP.** The end-to-end conversation loop works: live streaming
transcript (content, thinking, tool calls, notices), full input-line editing,
permission / question prompts, and the core overlays wired to real runtime data
(slash commands, model, session, theme, thinking / inference selectors,
settings, steer queue). Session turns persist on `turn_end` and can be reloaded
from the session manager.

### Done

- [x] Connect to the real Logician bridge/runtime
- [x] `TuiState` → React re-render bridge (`useSyncExternalStore`)
- [x] Streaming transcript via `transcript.handleEvent` / `isTranscriptEvent`
- [x] Input bar: cursor movement, word/line kill, `/` and `@` triggers
- [x] Slash popup driven by `createSlashCommands` + `filterSlashCommands`
- [x] Overlays respond to keys (shared `useOverlayInput` hook)
- [x] File-mention autocomplete (shallow project walk)
- [x] Permission (`allow` / `always` / `deny`) and question prompts
- [x] Session persistence + reload
- [x] Transcript scrollback with virtual scrolling, follow-mode toggle,
      and "↓ new output below" indicator
- [x] Markdown rendering: headings, bold/italic, code blocks with syntax
      highlighting (via emphasize), tables, lists, blockquotes, links
- [x] Input bar: undo/redo stack (Ctrl+Z / Ctrl+Y or Ctrl+Shift+Z), kill ring
      (Ctrl+U/W/K push to ring, Ctrl+Shift+V paste), word navigation
      (Ctrl+Left/Right)

### Deferred (parity phase)


- [ ] Subagent chunk rendering
- [ ] Todo bar, work surface, steer-queue status widget
- [ ] Goal runner + plan-mode approval flow
- [ ] Autoresearch dashboard live data; MCP / plugin manager CRUD
- [ ] Undo/redo + kill-ring in the input bar
- [ ] Virtualization for large overlay lists
- [ ] Trust-prompt overlay; headless exec mode

## License

MIT
