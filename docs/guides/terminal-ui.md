---
title: Terminal UI
description: TUI features, keybindings, layouts, and terminal compatibility.
---

# Terminal UI

The Logician TUI is designed for real terminal environments — SSH sessions, tmux, and local terminals.

## Layout

```
┌─────────────────────────────────────────────────┐
│ Logician v0.2.0                    [ask] mode  │ ← Status bar
├─────────────────────────────────────────────────┤
│ > Fix the auth bug in src/middleware.ts         │ ← Input
├─────────────────────────────────────────────────┤
│ 💭 Thinking: analyzing middleware auth check... │
│ 🔧 read src/middleware.ts                       │
│ 🔧 edit src/middleware.ts                       │
│ ✅ Done: Fixed auth check, 2 files changed     │
└─────────────────────────────────────────────────┘
```

## Keybindings

| Key | Action |
|---|---|
| `Enter` | Submit instruction |
| `Ctrl+C` | Cancel current operation |
| `Ctrl+Z` | Pause and return to prompt |
| `Ctrl+R` | Rewind to last checkpoint |
| `Ctrl+B` | Create bookmark |
| `/` | Open command palette |
| `?` | Show help |

## Terminal compatibility

Works with any VT100-compatible terminal:
- `xterm`, `gnome-terminal`, `konsole`, `alacritty`, `kitty`
- SSH sessions (local or remote)
- tmux and screen
- Windows Terminal

## Input modes

- **Normal** — type instructions naturally
- **Multi-line** — press `Shift+Enter` for newlines
- **Paste** — pasted text is processed as a single instruction

## Streaming output

All output streams in real time:
- Reasoning steps appear as they're generated
- Tool calls show before execution
- Results display immediately after completion
