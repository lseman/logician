---
title: Internal Resources
description: rag://, pr://, issue://, artifact://, and other internal URL schemes.
---

# Internal Resources

Logician exposes internal resources through special `scheme://target` URLs that
you can pass to the `read` and `write` tools. Schemes are case-insensitive.
Prefix a literal filename with `./` if it contains `://`.

## RAG — `rag://`

Retrieval-Augmented Generation operations on the document vector store:

| URL | Action |
|---|---|
| `read rag://` | Shows usage help |
| `read rag://list` | Lists indexed document IDs |
| `write rag://search` | Search docs — pass JSON `{"query":"...", "k?:number"}` |
| `write rag://ingest` | Ingest a document — pass JSON `{"path":"...", "docId?:string"}` |
| `write rag://delete` | Remove a document — pass JSON `{"docId":"..."}` |

`rag://` also has corresponding core tools (`rag_search`, `rag_ingest`,
`rag_list`, `rag_delete`) available for direct invocation. The URL form is
generally preferred when working from the agent prompt.

## GitHub — `pr://` and `issue://`

Read pull requests and issues through your configured GitHub MCP server:

### `pr://`

| URL | Action |
|---|---|
| `pr://owner/repo` | List open PRs |
| `pr://owner/repo/1428` | PR details |
| `pr://owner/repo/1428/files` | Changed files |
| `pr://owner/repo/1428/comments` | Comments |
| `pr://owner/repo/1428/reviews` | Reviews |
| `pr://owner/repo/1428/commits` | Commits |

### `issue://`

| URL | Action |
|---|---|
| `issue://owner/repo` | List open issues |
| `issue://owner/repo/42` | Issue details |
| `issue://owner/repo/42/comments` | Comments |
| `issue://owner/repo/42/sub-issues` | Sub-issues |

Both require a GitHub MCP server configured; they return a diagnostic message
when unavailable.

## Session artifacts — `artifact://`

Tool output from previous turns is stored as session-scoped artifacts:

| URL | Action |
|---|---|
| `artifact://` | List available artifacts |
| `artifact://<id>` | Read one artifact |
| `artifact://<id>/details.metrics.turns` | Read metadata fields |

## Local workspace — `local://`

Reads files under `.logician/artifacts/`. Supports writes.

## Merge conflicts — `conflict://`

| URL | Action |
|---|---|
| `conflict://<file>` | List all merge conflict blocks in a file |
| `conflict://<file>:<index>` | Read one block |

`write conflict://<file>` with `content` set to `ours`, `theirs`,
`ours+theirs`, or `base` resolves that block (or every block for the bare form).

## Subagents — `agent://`

Lists and reads completed subagent results. Use `agent://` to list, `agent://<id>` for the full result, and dot-notation for field access.

| URL | Action |
|---|---|
| `read agent://` | List completed agent IDs |
| `read agent://<id>` | Full result as JSON |
| `read agent://<id>/content` | Final output text only |
| `read agent://<id>/status` | Completion status |
| `read agent://<id>/details.metrics.turns` | Agent metrics |

## Documentation — `log://`

Reads documentation in the workspace `docs/` directory:

| URL | Action |
|---|---|
| `log://` | List documentation |
| `log://guides/` | List doc categories |
| `log://<path>` | Read a doc file |

## Remote hosts — `ssh://`

Reads files on remote hosts via SSH/scp:

| URL | Action |
|---|---|
| `ssh://` | List configured hosts (see `~/.logician/ssh.json`) |
| `ssh://<host>/path` | Read a remote file |

## Skills — `skill://`

Reads a loaded skill's full instructions. The `<name>` must be an exact skill
name; invalid names are rejected. Extra paths, queries, and fragments are not
supported.

## `memory://`

| URL | Action |
|---|---|
| `memory://list` | List observations |
| `memory://memories` | List durable memories |

## `xd://` device addresses

Tools that support it can also be invoked via `xd://` device addresses. This is
an alternative way to call tools, using the `write` tool to dispatch JSON
payloads.

| Device | Tool |
|---|---|
| `xd://git` | Git operations |
| `xd://sandbox` | Sandbox commands |
| `xd://file_diff` | File diff |
| `xd://browser` | Browser automation |
| `xd://lsp` | Language server protocol |
| `xd://hub` | Subagent message bus |

To list available devices: `read path="xd://"`

To view a device's schema: `read path="xd://<device>"`

To invoke a device: `write path="xd://<device>" content="<json>"`

## `read_skill` tool

The `read_skill` tool loads a skill's full instructions on demand. The system
prompt only advertises a compact skill catalog (name + description); the agent
calls `read_skill` with an exact skill name to pull the full SKILL.md body when
it decides to use a skill.

```
read_skill(name: string): string
```
