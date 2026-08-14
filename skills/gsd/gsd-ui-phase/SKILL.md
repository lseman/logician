---
name: ui-phase
description: "Generate UI design contract (UI-SPEC.md) for frontend phases"
argument-hint: "[phase]"
allowed-tools:
  - Read
  - Write
  - Bash
  - Glob
  - Grep
  - subagent
  - WebFetch
  - ask_user_question
  - mcp__context7__*
---

<objective>
Create a UI design contract (UI-SPEC.md) for a frontend phase.
Orchestrates gsd-ui-researcher and gsd-ui-checker.
Flow: Validate → Research UI → Verify UI-SPEC → Done
</objective>

<execution_context>
@repos/gsd-core/gsd-core/workflows/ui-phase.md
@repos/gsd-core/gsd-core/references/ui-brand.md
</execution_context>

<context>
Phase number: $ARGUMENTS — optional, auto-detects next unplanned phase if omitted.
</context>

<process>
Execute end-to-end.
Preserve all workflow gates.
</process>
