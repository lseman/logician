---
title: First Session
description: Walk through your first agent session end to end.
---

# First Session

This tutorial walks through your first agent session — from starting the TUI to getting a code change.

## Step 1: Start the TUI

```bash
cd logician/tui
npm start
```

You should see:

```
┌─────────────────────────────────────────────────┐
│ Logician v0.2.0                    [ask] mode  │
├─────────────────────────────────────────────────┤
│ >                              _     _          │
└─────────────────────────────────────────────────┘
```

## Step 2: Submit a simple task

Type:

```
Show me the contents of README.md
```

Press Enter. The agent will:

1. Think: `💭 Reading README.md...`
2. Call tool: `🔧 read_file README.md`
3. Show result: `✅ README.md: 45 lines`

## Step 3: Make a code change

Type:

```
Add a comment to the main function in src/index.ts explaining what it does
```

If in `ask` mode, you'll see:

```
🔧 edit_file: src/index.ts
   Change 1: Add doc comment to main()
   Apply? [Y/n]
```

Press `Y` to confirm.

## Step 4: Verify

The agent runs verification:

```
🔧 bash: npm run typecheck
✅ Type check passed
🔧 bash: npm test
✅ Tests passed (12/12)
```

## Step 5: Save the session

Your session is automatically persisted. Find it later with:

```
/session list
```

## Next steps

- Learn about [Skills](/guides/skills) to extend capabilities
- Explore [Reasoning Modes](/guides/reasoning) for complex tasks
- Try [Headless Mode](/tutorials/headless) for automation
