// ── memory:// protocol handler ───────────────────────────────────────────────
// Resolves memoriam observations and memories.
// URL forms:
//   memory://              — list sessions and entry counts
//   memory://list          — list recent observations
//   memory://memories      — list memories
//   memory://observe/<id>  — get observation by ID
//   memory://memory/<id>   — get memory by ID

import type {
	InternalResource,
	InternalUrl,
	ProtocolHandler,
	ResolveContext,
} from "./types";

type MemoryGateway = {
	getContext: (
		sessionId: string,
		query: string,
		budget: number,
	) => Promise<string>;
	listObservations: (
		sessionId: string,
		limit: number,
	) => Promise<
		Array<{ id: string; content: string; metadata?: Record<string, unknown> }>
	>;
	listMemories: (
		query?: Record<string, unknown>,
	) => Promise<
		Array<{ id: string; content: string; metadata?: Record<string, unknown> }>
	>;
};

export class MemoryProtocolHandler implements ProtocolHandler {
	readonly scheme = "memory";

	async resolve(
		url: InternalUrl,
		context?: ResolveContext,
	): Promise<InternalResource> {
		const gateway = context?.memory as MemoryGateway | undefined;
		const hostname = url.rawHost || url.hostname;
		const pathname = url.pathname;

		// memory:// — list sessions
		if (hostname === "memory" && pathname === "/") {
			return {
				url: url.href,
				content:
					"# Memory\n\nUse `memory://list` for observations or `memory://memories` for memories.\n",
				contentType: "text/markdown",
				notes: [
					"Memory gateway is not available. Configure Memoriam SDK to enable.",
				],
			};
		}

		if (!gateway) {
			return {
				url: url.href,
				content: "# Memory\n\nMemory gateway is not available.\n",
				contentType: "text/markdown",
				notes: [
					"Configure the Memoriam SDK in .logician.json to enable memory:// access.",
				],
			};
		}

		if (hostname !== "memory") {
			throw new Error(`memory:// URL requires host "memory": ${hostname}`);
		}

		// memory://list — list recent observations
		if (pathname === "/list" || pathname === "/list/") {
			return this.handleList(gateway, url);
		}

		// memory://memories — list memories
		if (pathname === "/memories" || pathname === "/memories/") {
			return this.handleListMemories(gateway, url);
		}

		// memory://observe/<id> — get observation by ID
		if (pathname.startsWith("/observe/")) {
			const id = pathname.slice("/observe/".length);
			if (!id)
				throw new Error("memory://observe/<id> requires an observation ID");
			return this.handleObserve(gateway, id, url);
		}

		// memory://memory/<id> — get memory by ID
		if (pathname.startsWith("/memory/")) {
			const id = pathname.slice("/memory/".length);
			if (!id) throw new Error("memory://memory/<id> requires a memory ID");
			return this.handleMemory(gateway, id, url);
		}

		throw new Error(
			`Unknown memory:// path: ${pathname}. Use memory://list, memory://memories, memory://observe/<id>, or memory://memory/<id>.`,
		);
	}

	private async handleList(
		gateway: MemoryGateway,
		url: InternalUrl,
	): Promise<InternalResource> {
		const observations = await gateway.listObservations("all", 20);
		const lines = observations.map(
			o => `- ${o.id}: ${o.content.slice(0, 120)}`,
		);
		return {
			url: url.href,
			content: `# Recent Observations (${observations.length})\n\n${lines.join("\n") || "No observations."}`,
			contentType: "text/markdown",
		};
	}

	private async handleListMemories(
		gateway: MemoryGateway,
		url: InternalUrl,
	): Promise<InternalResource> {
		const memories = await gateway.listMemories();
		const lines = memories.map(m => `- ${m.id}: ${m.content.slice(0, 120)}`);
		return {
			url: url.href,
			content: `# Memories (${memories.length})\n\n${lines.join("\n") || "No memories."}`,
			contentType: "text/markdown",
		};
	}

	private async handleObserve(
		gateway: MemoryGateway,
		id: string,
		url: InternalUrl,
	): Promise<InternalResource> {
		const observations = await gateway.listObservations("all", 100);
		const obs = observations.find(o => o.id === id);
		if (!obs) {
			throw new Error(`Unknown observation: ${id}`);
		}
		return {
			url: url.href,
			content: obs.content,
			contentType: "text/plain",
			sourcePath: `memory://observe/${id}`,
		};
	}

	private async handleMemory(
		_gateway: MemoryGateway,
		_id: string,
		_url: InternalUrl,
	): Promise<InternalResource> {
		throw new Error(`memory://memory/<id> not yet implemented`);
	}
}
