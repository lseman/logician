// ── Optional-capability registry ─────────────────────────────────────────
// Table of the statically-constructed default tools that can be switched
// on/off, so createDefaultTools() (initial construction) and ToolRouter
// (runtime toggling via setGraphicianEnabled/etc.) read the same list instead
// of each hardcoding its own enabled-tool wiring. Dynamically-discovered
// capabilities (e.g. fffgrep, an MCP-origin tool identified by isFffGrepTool
// below rather than a static export) aren't constructed here, but share this
// file since their enable/disable identity logic is the same kind of thing.

import type { Tool } from "@logician/log-core";
import { graphician } from "./graphician.ts";

export interface OptionalCapability {
	id: string;
	tool: Tool;
	enabledByDefault: boolean;
}

/** Statically-constructed optional capabilities, in default-tool-set order. */
export const OPTIONAL_CAPABILITIES: OptionalCapability[] = [
	{ id: "graphician", tool: graphician, enabledByDefault: true },
];

/** Identify the fff MCP grep tool by origin, since it isn't a static export. */
export function isFffGrepTool(tool: Tool): boolean {
	return (
		tool.origin?.kind === "mcp" &&
		tool.origin.server.toLowerCase() === "fff" &&
		tool.origin.tool.toLowerCase() === "grep"
	);
}
