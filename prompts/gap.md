---
description: Git add, commit, and push
argument-hint: "<commit message>"
aliases:
  - gap
---
Perform a full git cycle: add all changes, commit, and push.

1. Run `git status` and review what would be staged.
2. Stage all changes: `git add .`
3. Commit with the provided message, or infer a message from the changes if none is provided. Use `git commit -m "<message>"`.
4. Check the current branch. If not `main`, stop and ask what to do.
5. Push the current branch.

Constraints:
- Never force push.
- Run `npm run check` first if any code changed (TypeScript/JavaScript/JSON).
- Report what was committed and pushed.
