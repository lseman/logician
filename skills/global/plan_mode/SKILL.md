---
name: Plan Mode
description: Enter or exit plan mode to block write/execute tools and work in a safe exploration-only state.
triggers:
  - enter plan mode
  - exit plan mode
  - switch to planning mode
  - stop plan mode
  - enable plan mode
  - disable plan mode
aliases:
  - plan_mode
  - planning mode
  - enter_plan_mode
  - exit_plan_mode
preferred_tools:
  - enter_plan_mode
  - exit_plan_mode
when_not_to_use:
  - when you need to actually write or edit files
---

# Plan Mode

While plan mode is active, all write and execution tools are blocked:
`write_file`, `edit_file`, `apply_edit_block`, `smart_edit`, `bash`, `patch`.

Use it to safely explore a codebase, build a plan, and confirm the approach
before making any changes.

## Usage

```
enter_plan_mode()     → activates plan mode
exit_plan_mode()      → deactivates plan mode
```
