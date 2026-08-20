// ── Node entry point ──────────────────────────────────────────────────────
// Secondary entry point for Node/Bun-specific code (currently just
// NodeExecutionEnv). Kept separate from index.ts so the universal barrel
// stays free of Node builtin imports. Ported from pi coding agent's node.ts.

export { NodeExecutionEnv } from "./env/nodejs.ts";
export * from "./index.ts";
