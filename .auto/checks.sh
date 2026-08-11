#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../tui"

# Type check — only the TUI workspace (our scope).
# The full root typecheck has pre-existing failures in agent-capabilities
# test files that are unrelated to TUI latency optimization.
bun run --filter=@logician/tui typecheck 2>&1 | tail -50

# Lint — TUI packages only
bun run lint 2>&1 | tail -50
