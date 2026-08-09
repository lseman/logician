# GSD Core — Ported to Logician

This directory contains the **GSD Core** framework ported to Logician as a skill bundle.

## What is GSD Core?

GSD Core is a spec-driven context engineering framework for AI coding agents. It enforces a disciplined phase loop:

```
new-project → discuss-phase → plan-phase → execute-phase → verify-work → ship
```

## Installation Status

| Component | Source | Logician Status |
|-----------|--------|-----------------|
| **Skills** | 71 SKILL.md files | ✅ Installed in `skills/gsd/` |
| **Commands** | ~40 slash commands | ✅ Registered via `bridge.ts` |
| **Workflows** | Markdown workflows | ✅ Available via GSDBRIDGE protocol |
| **Hooks** | Node.js Claude Code hooks | ⚠️ Not needed (Logician uses typed events) |
| **Capabilities** | 44 runtime adapters | ⚠️ Not ported (Logician has its own runtime) |
| **STATE.md** | Project memory | ✅ Implemented in `src/state.ts` |
| **Phase Mgmt** | Phase lifecycle | ✅ Implemented in `src/phase.ts` |

## Available Commands

### Project Lifecycle
- `/gsd:new-project [--auto]` — Initialize a new project
- `/gsd:onboard [--auto]` — Onboard an existing codebase
- `/gsd:next` — Smart entry: detect next action
- `/gsd:progress [phase]` — Check project progress

### Phase Workflows
- `/gsd:discuss-phase <phase> [flags]` — Context gathering via questioning
- `/gsd:plan-phase [phase] [flags]` — Create phase plan with verification
- `/gsd:execute-phase <phase>` — Execute phase plans with waves
- `/gsd:verify-work <phase>` — Validate against requirements
- `/gsd:ship [phase]` — Create PR and merge

### Quick Tasks
- `/gsd:quick [task]` — Execute quick task with GSD guarantees
- `/gsd:quick list` — List quick tasks
- `/gsd:quick status <slug>` — Check task status

### Management
- `/gsd:stats` — Project statistics
- `/gsd:config <get|set>` — Configure settings
- `/gsd:settings [list\|get\|set]` — Model profile settings
- `/gsd:phase <add\|remove\|list\|complete>` — Phase CRUD

### Review & Audit
- `/gsd:code-review [phase]` — Source code review
- `/gsd:ui-review [phase]` — Visual UI audit
- `/gsd:audit-fix [target]` — Autonomous audit-to-fix

### Memory
- `/gsd:capture <text>` — Capture ideas and notes
- `/gsd:mempalace-capture <path>` — Store artifact in memory palace
- `/gsd:mempalace-recall [query]` — Recall from memory palace

### Session
- `/gsd:thread <create\|list\|switch>` — Manage context threads
- `/gsd:pause-work [reason]` — Create context handoff
- `/gsd:resume-work` — Restore context from handoff

### Ideation
- `/gsd:explore [topic]` — Socratic ideation
- `/gsd:sketch [desc]` — HTML UI mockups
- `/gsd:spike [topic]` — Experiential exploration

## How It Works

GSD workflows are adapted for Logician via the **GSDBRIDGE protocol**:

1. A GSD slash command is invoked (e.g., `/gsd:plan-phase 3`)
2. The bridge returns a `GSDBRIDGE:workflow:...` response
3. The Logician agent reads the corresponding workflow from `repos/gsd-core/gsd-core/workflows/`
4. The agent executes the workflow using Logician's native tools:
   - `ask_user_question` replaces `AskUserQuestion`
   - `subagent` replaces `Agent` tool
   - `state.ts` provides STATE.md operations
   - `phase.ts` provides phase lifecycle operations

## Workflow Adaptation

The GSD workflow markdown files use bash blocks with `gsd_run` commands.
In Logician, these are handled by:

1. **STATE.md operations** → `state.ts` functions
2. **Phase directory operations** → `phase.ts` functions
3. **Interactive questioning** → `ask_user_question` tool
4. **Subagent spawning** → `subagent` tool
5. **Git operations** → `Bash` tool

## Files

```
skills/gsd/
├── README.md                  # This file
├── bridge.ts                  # Extension registration (commands)
├── *.md                       # 71 GSD skill files (adapted)
└── src/
    ├── state.ts               # STATE.md management
    └── phase.ts               # Phase lifecycle management
```

## Source

Original project: [opengsd/gsd-core](https://github.com/opengsd/gsd-core)
Port: `repos/gsd-core/` (cloned source) → `skills/gsd/` (Logician adaptation)
