---
title: Trust & Safety
description: Permission modes, safe edits, and access control.
---

# Trust & Safety

Logician provides granular control over what the agent can do, from read-only planning to fully autonomous editing.

## Permission modes

| Mode | Description | File edits | Tool execution |
|---|---|---|---|
| `plan` | Read-only, suggests changes | ❌ | ❌ |
| `ask` | Asks before each action | ✅ (confirm) | ✅ (confirm) |
| `acceptEdits` | Edits automatically, asks for tools | ✅ | ⚠️ (confirm) |
| `acceptAll` | Full autonomy | ✅ | ✅ |

## Safe edits

All file edits use strict exact-text matching:

1. **Read first** — the agent reads the file before editing
2. **Exact match** — edits use the exact text from the file
3. **CRLF preservation** — line endings are preserved
4. **BOM handling** — byte order marks are handled correctly
5. **Path normalization** — paths are normalized before operations

```
// Edit uses exact text from file
oldText: "function login(user) { return false; }"
newText: "function login(user) { return validate(user); }"
```

## Tool restrictions

Control which tools the agent can use:

```json
{
  "tools": {
    "allowed": ["read_file", "grep", "find"],
    "denied": ["bash", "edit_file"]
  }
}
```

## File access control

Restrict which directories the agent can access:

```json
{
  "access": {
    "allowedPaths": ["src/", "tests/"],
    "deniedPaths": ["node_modules/", ".git/"]
  }
}
```

## Audit trail

Every action is logged:

```
[10:23:45] read_file: src/auth.ts
[10:23:46] edit_file: src/auth.ts (3 changes)
[10:23:47] bash: npm test (exit 0)
[10:23:52] write_file: docs/changelog.md
```
