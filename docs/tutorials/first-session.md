---
title: First Session
description: Complete a small, safe change and verify it in the TUI.
---

# First Session

This tutorial uses a real repository and keeps the first task deliberately small.

## 1. Start in your workspace

```bash
cd /path/to/your-project
logician
```

Review the trust prompt. Trust enables project-local `.logician.json`, skills, and extensions; decline if you have not inspected them.

## 2. Begin with inspection

Ask:

```text
Explain how this project runs its tests. Do not change files.
```

This confirms the model connection, search tools, and repository context without authorizing edits. Tool cards show what Logician inspected; `Ctrl+O` expands or collapses their details.

## 3. Request a bounded change

Choose a small task with an explicit verification condition:

```text
Add a regression test for the empty-input case, make the smallest implementation change needed, and run the focused test file.
```

Under `ask` permissions, approve or deny each gated operation. You can send corrective steering during the turn with `Ctrl+Enter`.

## 4. Review the outcome

Check the final response for changed files, behavior, and verification evidence. Inspect the worktree yourself when the change matters:

```bash
git diff --check
git diff
```

If verification was incomplete, ask Logician to run the missing check rather than assuming success.

## 5. Find the session later

Sessions save automatically. Use `/session` for the interactive browser, `/sessions` for a list, and `/name first-regression-test` to give the current session a memorable name.

## Next steps

- Learn [permission, sandbox, and trust boundaries](/guides/trust).
- Add reusable instructions with [skills](/guides/skills).
- Use [headless mode](/tutorials/headless) for machine-readable automation.
