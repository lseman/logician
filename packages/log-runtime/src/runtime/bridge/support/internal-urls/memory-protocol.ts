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
	SchemeHost,
	UrlCompletion,
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
	readonly immutable = true;
	readonly spec = { backing: "virtual", selectors: "lines" as const };

	promptDoc(host: SchemeHost): string | undefined {
		if (!host.addressable) return undefined;
		return [
			"**memory://** - Access agent memory and observations.",
			"URL forms:",
			"- `memory://list` - recent observations",
			"- `memory://memories` - all stored memories",
			"- `memory://observe/<id>` - get observation by ID",
			"- `memory://memory/<id>` - get memory by ID",
		].join("\n");
	}

	async resolve(
		url: InternalUrl,
		context?: ResolveContext,
	): Promise<InternalResource> {
		const gateway = context?.memory as MemoryGateway | undefined;
		// The shared parser only puts the first path segment into `url.host`;
		// the rest lands in `url.pathname`. Reassemble the full address before
		// dispatching — see log-protocol.ts for the same pattern.
		const full = url.pathname === "/" ? url.host : `${url.host}${url.pathname}`;

		// memory:// — list sessions
		if (full === "") {
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

		// memory://list — list recent observations
		if (full === "list") {
			return this.handleList(gateway, url);
		}

		// memory://memories — list memories
		if (full === "memories") {
			return this.handleListMemories(gateway, url);
		}

		// memory://observe/<id> — get observation by ID
		if (full.startsWith("observe/")) {
			const id = full.slice("observe/".length);
			if (!id)
				throw new Error("memory://observe/<id> requires an observation ID");
			return this.handleObserve(gateway, id, url);
		}

		// memory://memory/<id> — get memory by ID
		if (full.startsWith("memory/")) {
			const id = full.slice("memory/".length);
			if (!id) throw new Error("memory://memory/<id> requires a memory ID");
			return this.handleMemory(gateway, id, url);
		}

		throw new Error(
			`Unknown memory:// path: ${full}. Use memory://list, memory://memories, memory://observe/<id>, or memory://memory/<id>.`,
		);
	}

	/**
	 * Candidates for the token after `memory://`. The static routes are always
	 * offered; when the gateway is configured, `memory/…` and `observe/…`
	 * queries are answered with real IDs (observation window: the same 100
	 * most recent `resolve` scans, so a completed ID is one `observe/<id>` can
	 * actually fetch).
	 */
	async complete(
		query: string,
		context?: ResolveContext,
	): Promise<UrlCompletion[]> {
		const gateway = context?.memory as MemoryGateway | undefined;
		const q = query.toLowerCase();
		const items: UrlCompletion[] = [];
		for (const [value, description] of [
			["list", "recent observations"],
			["memories", "all stored memories"],
		] as const) {
			if (q === "" || value.startsWith(q) || q.startsWith(value)) {
				items.push({ value, description });
			}
		}
		if (!gateway) return items;
		if (q.startsWith("memory/")) {
			const memories = await gateway.listMemories();
			for (const m of memories) {
				items.push({
					value: `memory/${m.id}`,
					description: m.content.slice(0, 80),
				});
			}
		} else if (q.startsWith("observe/")) {
			const observations = await gateway.listObservations("all", 100);
			for (const o of observations) {
				items.push({
					value: `observe/${o.id}`,
					description: o.content.slice(0, 80),
				});
			}
		}
		return items;
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
		// The worker protocol only lists by (session, limit) — no get-by-id or
		// offset. Scan a small window first, then deepen, so recent lookups
		// stay cheap and older observations remain reachable.
		const windows = [100, 2000];
		for (const limit of windows) {
			const observations = await gateway.listObservations("all", limit);
			const obs = observations.find(o => o.id === id);
			if (obs) {
				return {
					url: url.href,
					content: obs.content,
					contentType: "text/plain",
					sourcePath: `memory://observe/${id}`,
				};
			}
		}
		const maxWindow = windows[windows.length - 1];
		throw new Error(
			`Unknown observation: ${id} (not found in the most recent ${maxWindow} observations). Older observations may have been pruned.`,
		);
	}

	private async handleMemory(
		gateway: MemoryGateway,
		id: string,
		url: InternalUrl,
	): Promise<InternalResource> {
		const memories = await gateway.listMemories();
		const mem = memories.find(m => m.id === id);
		if (!mem) {
			throw new Error(`Unknown memory: ${id}`);
		}
		return {
			url: url.href,
			content: mem.content,
			contentType: "text/plain",
			sourcePath: `memory://memory/${id}`,
		};
	}
}
