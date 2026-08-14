---
title: Trust & Safety
description: Permission modes, safe edits, and access control.
---

# Trust & Safety

Logician provides granular control over what the agent can do, from read-only planning to fully autonomous editing.

## Permission modes

| Mode | Read-only tools | File edits | Other tools |
|---|---|---|---|
| `plan` | Allowed | Denied | Denied |
| `ask` | Allowed | Ask | Ask |
| `acceptEdits` | Allowed | Allowed | Ask |
| `acceptAll` | Allowed | Allowed | Allowed |

`acceptEdits` is the default. Full autonomy must be selected explicitly;
non-interactive approval requests fail closed.

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
  "permissions": {
    "allow": ["read_file", "grep", "find"],
    "deny": ["bash(git push *)", "edit_file(secrets/*)"]
  }
}
```

Rules can match a tool name or a tool plus a glob over its primary argument. Deny rules take precedence, followed by explicit allow rules and then the active mode.

## File access control

Restrict which directories the agent can access:

```json
{
  "allowedPaths": ["/absolute/shared/source"],
  "allowAllPaths": false
}
```

The workspace is always in scope. `allowedPaths` adds absolute roots outside it; `allowAllPaths` disables this boundary and should be used sparingly. Use permission deny rules for tool/argument restrictions inside allowed roots.

## Audit trail

Every action is logged:

```
[10:23:45] read_file: src/auth.ts
[10:23:46] edit_file: src/auth.ts (3 changes)
[10:23:47] bash: npm test (exit 0)
[10:23:52] write_file: docs/changelog.md
```
