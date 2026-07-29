---
title: MCP Servers
description: stdio and streamable HTTP MCP server integration, tool extension.
---

# MCP Servers

Logician integrates with Model Context Protocol (MCP) servers to extend its tool surface. Both stdio and HTTP transports are supported.

## Configuration

Add MCP servers to your config:

```json
{
  "mcp": {
    "servers": {
      "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
        "transport": "stdio"
      },
      "github": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {
          "GITHUB_TOKEN": "your-token"
        },
        "transport": "stdio"
      },
      "custom": {
        "url": "http://localhost:3000/mcp",
        "transport": "http"
      }
    }
  }
}
```

## Available tools

MCP servers expose tools that appear alongside built-in tools:

```
Built-in tools:
  read_file, write_file, edit_file, grep, bash, find, git, ...

MCP tools:
  github_list_issues, github_create_issue, ...
  filesystem_read_dir, filesystem_write_file, ...
```

## Transport types

| Transport | Use case |
|---|---|
| `stdio` | Local servers, CLI tools wrapped as MCP |
| `http` | Remote MCP servers, containerized services |

## Server lifecycle

The agent manages MCP server lifecycle automatically:
- Servers start when the agent initializes
- Tools are discovered and registered
- Connections are maintained across the session
- Servers restart on connection failure

## Debugging

```bash
# List all available MCP tools
npm start -- doctor --mcp

# Test a specific server
npm start -- mcp test filesystem
```
