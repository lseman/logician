// Sample Pi extension that runs on the Logician adapter
// This demonstrates what a Pi extension looks like and how it translates to Logician

// ── Pi-style extension that runs via the adapter ──────────────────────────────

interface ExtensionAPI {
	on(event: string, handler: (event: any, ctx: any) => Promise<any> | any): void;
	registerTool(tool: {
		name: string;
		label?: string;
		description: string;
		parameters: any;
		execute: (toolCallId: string, params: any) => Promise<any>;
	}): void;
	registerCommand(name: string, options: {
		description?: string;
		handler: (args: string, ctx: any) => Promise<void>;
	}): void;
}

interface ExtensionContext {
	ui: {
		notify(message: string, type?: "info" | "warning" | "error"): void;
		confirm(title: string, message: string): Promise<boolean>;
	};
	cwd: string;
}

export default async function (api: ExtensionAPI) {
	// React to session start
	api.on("session_start", async (event: any, ctx: ExtensionContext) => {
		ctx.ui.notify(`Extension loaded in session ${event.reason}`, "info");
	});

	// Block dangerous bash commands
	api.on("tool_call", async (event: any, ctx: ExtensionContext) => {
		if (event.toolName === "bash") {
			const cmd = event.input?.command as string;
			if (cmd?.includes("rm -rf") || cmd?.includes("sudo rm")) {
				const ok = await ctx.ui.confirm(
					"Dangerous Command",
					`Block "rm -rf" command? ${cmd}`,
				);
				if (!ok) return { block: true, reason: "User denied dangerous command" };
				return { block: true, reason: "Blocked by extension" };
			}
		}
	});

	// Register a custom tool
	api.registerTool({
		name: "hello_logician",
		label: "Hello Logician",
		description: "Say hello from a Pi extension",
		parameters: {
			type: "object",
			properties: {
				name: { type: "string", description: "Name to greet" },
			},
		},
		execute: async (toolCallId, params) => {
			return {
				content: [{ type: "text", text: `Hello, ${params.name || "world"}! This is a Pi extension running on Logician.` }],
			};
		},
	});

	// Register a custom command
	api.registerCommand("pi-status", {
		description: "Show Pi extension status",
		handler: async (_args, ctx) => {
			ctx.ui.notify("Pi extension is active", "info");
		},
	});
}
