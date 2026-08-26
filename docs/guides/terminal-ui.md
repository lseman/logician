---
title: Terminal UI
description: TUI features, keybindings, layouts, and terminal compatibility.
---

# Terminal UI

The Logician TUI is designed for real terminal environments — SSH sessions,
tmux, and local terminals. Its layout adapts to narrow terminals while keeping
the transcript, composer, and current execution state visible.

## Layout

```text
◆ LOGICIAN

  RESPONSE
  I found the failing boundary and updated the parser.

  COMMAND  bun test packages/log-core
  OUTPUT   42 pass, 0 fail

  COMPOSER             / Enter commands · @ files · Enter send · Ctrl+Enter steer
  › Ask Logician…
  ──────────────────────────────────────────────────────────────────────────────
  ● READY │ model │ cwd │ branch │ thinking mode
```

The main transcript is scrollable. Tool activity is rendered as compact cards
that can be focused and expanded, while longer-lived workflows can open a work
surface or fullscreen overlay without replacing the conversation. An empty
session shows project-aware starting actions; typing `/` opens the searchable
command palette and `@` starts file mention completion.

## Keybindings

| Key | Action |
|---|---|
| `Enter` | Submit instruction |
| `Shift+Enter` | Insert a newline |
| `Ctrl+Enter` | Immediately steer the active turn with the composer text |
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
| `Ctrl+M` / `Alt+M` | Open the inference-mode selector |
| `Ctrl+G` | Jump to a file in the current working set |
| `Ctrl+Shift+T` | Cycle the thinking display mode |
| `Ctrl+A` | Open the autoresearch dashboard |
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
- **Steering** — press `Ctrl+Enter` during a turn to interrupt and steer now
- **Paste** — pasted text is processed as a single instruction

## Streaming output

Provider output and tool progress stream in real time:
- Thinking content appears when the provider exposes it and thinking is enabled
- Tool calls show before execution
- Results display immediately after completion
- Tool cards remain compact by default and can be focused or expanded with the
  keybindings above
