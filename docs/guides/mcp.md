---
title: MCP Servers
description: Configure stdio and streamable HTTP MCP servers.
---

# MCP Servers

Model Context Protocol (MCP) servers add external tools to Logician. Server declarations can live in `~/.logician/settings.json`, `~/.logician/mcp.json`, or a trusted project's `.logician.json` or `.mcp.json`.

## Configure servers

Use the top-level `mcpServers` map:

```json
{
  "mcpServers": {
    "filesystem": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
      "enabled": true,
      "timeout": 30
    },
    "remote-docs": {
      "type": "streamable-http",
      "url": "https://example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${DOCS_MCP_TOKEN}"
      },
      "enabled": true,
      "timeout": 30
    }
  }
}
```

Put `DOCS_MCP_TOKEN=...` in `~/.logician/.env`; Logician expands `${NAME}` placeholders in MCP headers and process environments.

## Transports

| Type | Required fields | Use case |
|---|---|---|
| `stdio` | `command`, optional `args` and `env` | Local processes and CLI bridges |
| `streamable-http` | `url`, optional `headers` | Remote MCP endpoints using POST and SSE responses |
| `http` | `url`, optional `headers` | Alias accepted for HTTP MCP servers |

## Loading behavior

MCP discovery normally runs in the background, so a slow server does not block the first user turn. Tools appear as servers finish connecting. Set `mcpEager` when startup must wait for discovery.

Use `/mcp` to inspect or manage declared servers. `logician doctor --json` validates declarations without connecting to them.

## Troubleshooting

1. Run the configured stdio command directly and check its exit status.
2. Confirm remote endpoints accept `POST` and return JSON or `text/event-stream`.
3. Verify referenced environment variables are available in `~/.logician/.env` or the parent process.
4. Increase the server's `timeout` if initialization is legitimately slow.
5. Restart Logician after editing startup configuration.
