import type { Tool } from "@logician/log-core";
import type {
	InternalResource,
	InternalUrl,
	ProtocolHandler,
} from "../../resources/types.ts";

/** Session-owned catalog. Execution stays in the harness's ToolRegistry. */
export class XdDeviceRegistry implements ProtocolHandler {
	readonly scheme = "xd";
	readonly immutable = true;
	private readonly devices = new Map<string, Tool>();

	mount(tool: Tool): void {
		if (!tool.name || /[/?#\s]/.test(tool.name)) {
			throw new Error(`Invalid device name: ${tool.name}`);
		}
		this.devices.set(tool.name, tool);
	}

	unmount(name: string): void {
		this.devices.delete(name);
	}

	clear(): void {
		this.devices.clear();
	}

	list(): Tool[] {
		return [...this.devices.values()].sort((a, b) =>
			a.name.localeCompare(b.name),
		);
	}

	private get(name: string): Tool {
		const tool = this.devices.get(name);
		if (!tool) {
			throw new Error(
				`Unknown xd:// device: ${name}. Use read with path="xd://" to list available devices.`,
			);
		}
		return tool;
	}

	async resolve(url: InternalUrl): Promise<InternalResource> {
		if (url.target === "" || url.target === "/") {
			const tools = this.list();
			return {
				url: url.href,
				contentType: "text/markdown",
				isDirectory: true,
				content: [
					"# Available xd:// devices",
					"",
					'Read xd://<name> for its schema. Invoke with write path="xd://<name>" and content containing a JSON object.',
					"",
					...(tools.length
						? tools.map(
								tool =>
									`- xd://${tool.name}: ${tool.promptSnippet ?? tool.description}`,
							)
						: ["No devices are mounted in this session."]),
				].join("\n"),
			};
		}
		const tool = this.get(url.target);
		return {
			url: url.href,
			contentType: "text/markdown",
			content: [
				`# Device: xd://${tool.name}`,
				"",
				tool.description,
				"",
				`Invoke with write path="xd://${tool.name}" and content containing a JSON object matching this schema.`,
				"",
				"## Input Schema",
				"```json",
				JSON.stringify(tool.parameters, null, 2),
				"```",
				...(tool.promptGuidelines?.length
					? ["", "## Guidelines", ...tool.promptGuidelines.map(g => `- ${g}`)]
					: []),
			].join("\n"),
		};
	}

	/** Decode transport only; ToolRegistry prepares and authorizes the target. */
	resolveCall(
		name: string,
		content: unknown,
	): { name: string; arguments: Record<string, unknown> } {
		const tool = this.get(name);
		if (typeof content !== "string")
			throw new Error(
				"Device content must be a JSON object encoded as a string.",
			);
		let args: unknown;
		try {
			args = JSON.parse(content);
		} catch {
			throw new Error(`Invalid JSON for xd://${name}. Expected a JSON object.`);
		}
		if (!args || typeof args !== "object" || Array.isArray(args)) {
			throw new Error(
				`Invalid arguments for xd://${name}. Expected a JSON object.`,
			);
		}
		return { name: tool.name, arguments: args as Record<string, unknown> };
	}
}
