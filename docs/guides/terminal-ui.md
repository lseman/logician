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
| `Esc`, `Esc` | Clear the composer, then safely interrupt the active turn and restore its prompt |
| `Ctrl+C` | Immediately request interruption |
| `Ctrl+O` | Expand or collapse all tool results |
| `Alt+J` / `Alt+K` | Focus the next or previous tool result |
| `Alt+Enter` | Expand or collapse the focused tool result |
| `Ctrl+L` | Open the model selector |
| `Ctrl+S` | Open the session manager |
| `Ctrl+K` | Cycle sandbox mode |
| `Ctrl+P` | Toggle plan/act permission mode |
| `Ctrl+I` | Toggle autonomous/minimal execution policy (enhanced keyboard protocol) |
| `Ctrl+M` / `Alt+M` | Cycle inference mode |
| `/` | Open command palette |
| `/help` | Show the live command reference |

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
