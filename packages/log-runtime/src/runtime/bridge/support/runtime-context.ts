/**
 * The composition seam between a ToolRouter's current tool set and a
 * ToolRegistry a session actually executes tools through. AgentBridge.getTools()
 * and ToolRouter.buildRegistry() used to duplicate this construction inline
 * (new ToolRegistry(...) + registerMany(...)) at each call site; this module
 * is the one place that does it, so a call site composes instead of
 * reimplementing.
 */

import { ToolRegistry } from "@logician/log-core/runtime";
import type { ToolRouter } from "./tool-router.ts";

/** Registry construction options a caller controls; cwd/tools come from the router. */
export type RegistryConfig = NonNullable<
	ConstructorParameters<typeof ToolRegistry>[0]
>;

/**
 * Build a ToolRegistry over a router's current default tools. Each call
 * produces a fresh registry snapshotting the router's tools at that moment —
 * callers that need to react to tools added later (MCP/skills loading in the
 * background) should re-derive rather than cache the result.
 */
export function buildToolRegistry(
	router: ToolRouter,
	config: RegistryConfig = {},
): ToolRegistry {
	const registry = new ToolRegistry(config);
	registry.registerMany(router.getDefaultTools());
	return registry;
}
