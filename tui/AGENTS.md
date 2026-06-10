# Logician TUI Agent Guide

## Scope

These instructions apply to the `tui` package.

## Workflow

- Keep changes scoped to the TypeScript TUI unless the user explicitly asks for cross-project edits.
- Prefer the local agent-core patterns over introducing new framework layers.
- Use `npx tsc --noEmit` after TypeScript changes.
- Run Prettier with 4-space indentation on touched TypeScript, JSON, and Markdown files.
- Do not start the interactive TUI as a verification step unless the user asks; use focused smoke checks instead.

## UI Conventions

- Preserve the terminal-first interaction model: keyboard shortcuts, compact status, scrollable transcript, and minimal redraw churn.
- Match Pi-inspired behavior where it helps, but adapt it to this package's lighter component system.
- Tool execution should stay compact by default and expand with `Ctrl+O`.
- Keep expanded tool output useful for inspection: show args, written content, edit bodies, command output, and diffs where available.

## Build

- `make build` should produce `dist/logician`.
- The binary entry point is `src/index.ts`.
- Keep generated build output out of source edits unless the user explicitly asks to inspect or commit it.
