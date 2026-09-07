// ── xd:// Virtual Device Registry ────────────────────────────────────────────
// Mounts tools behind `xd://` URLs for discoverable, rarely-used operations.
//
// Core tools (read_file, write_file, edit_file, etc.) stay top-level.
// Discoverable tools (git, sandbox, browser, lsp, hub, graphician, etc.) are
// mounted as `xd://` devices. `read xd://` lists them; `write xd://<name>`
// dispatches them with JSON arguments.
//
// Resolution devices (xd://resolve, xd://reject) are registered here so that
// both the read and write paths share a single registry.

import type { Tool } from "@logician/log-core";

// ── Device types ─────────────────────────────────────────────────────────────

/** A mounted xd:// device. */
export interface XdDevice {
	/** Device name (used as xd://<name>). */
	name: string;
	/** One-line description shown in `read xd://`. */
	description: string;
	/** Full input schema (JSON Schema) for `write xd://<name>`. */
	schema: Record<string, unknown>;
	/** Execute the device with parsed JSON args; returns the result string. */
	execute: (args: Record<string, unknown>) => Promise<string>;
	/** Whether the device has been mounted. */
	mounted: boolean;
}

/** Device names for the resolution flow (xd://resolve, xd://reject). */
export const RESOLVE_DEVICE_NAME = "resolve";
export const REJECT_DEVICE_NAME = "reject";
export const DISCOVERABLE_XD_PREFIX = "xd://";

// ── Registry ─────────────────────────────────────────────────────────────────

const devices = new Map<string, XdDevice>();

/** Register a tool as an xd:// device. */
export function mountXdDevice(device: XdDevice): void {
	devices.set(device.name, device);
}

/** Remove a device from the registry. */
export function unmountXdDevice(name: string): void {
	devices.delete(name);
}

/** Check whether a device name is registered. */
export function hasXdDevice(name: string): boolean {
	return devices.has(name);
}

/** List all mounted devices as a flat map of name → description. */
export function listXdDevices(): Map<string, string> {
	const result = new Map<string, string>();
	for (const [name, dev] of devices) {
		if (dev.mounted) {
			result.set(name, dev.description);
		}
	}
	return result;
}

/** Read a device's documentation. Returns `null` if not found. */
export function readXdDeviceDocs(name: string): string | null {
	const dev = devices.get(name);
	if (!dev) return null;

	return [
		`# Device: ${dev.name}`,
		``,
		dev.description,
		``,
		`## Input Schema`,
		``,
		"```json",
		JSON.stringify(dev.schema, null, 2),
		"```",
	].join("\n");
}

/**
 * Execute a device call. The caller has already parsed the JSON content
 * into `args`. Returns the device's result string.
 */
export async function dispatchXdDevice(
	name: string,
	args: Record<string, unknown>,
): Promise<string> {
	const dev = devices.get(name);
	if (!dev) {
		return `Unknown xd:// device: ${name}. Run \`read xd://\` to list available devices.`;
	}
	if (!dev.mounted) {
		return `Device ${name} is not available in this session. Check your configuration.`;
	}
	return dev.execute(args);
}

// ── Helpers for resolution devices ───────────────────────────────────────────

/** Check if a device name refers to a resolution device. */
export function isResolutionDeviceName(name: string): boolean {
	return name === RESOLVE_DEVICE_NAME || name === REJECT_DEVICE_NAME;
}

// ── Convenience: mount a Tool as an xd:// device ─────────────────────────────

/**
 * Wrap a `Tool` as an xd:// device. The device's execute function runs the
 * tool's `execute` method with the given context.
 *
 * Note: this helper does not capture the full ToolContext; it passes a minimal
 * one. For tools that need cwd/allowedPaths etc., mount manually instead.
 */
export function toolAsXdDevice(tool: Tool, mounted = true): XdDevice {
	return {
		name: tool.name,
		description: tool.description,
		schema: tool.parameters,
		execute: async (args: Record<string, unknown>) => {
			const result = await tool.execute(args, { cwd: process.cwd() });
			if (typeof result === "string") return result;
			return result.content;
		},
		mounted,
	};
}
