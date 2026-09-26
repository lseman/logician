// ── Agent Output Registry ─────────────────────────────────────────────────────
// Process-global registry that stores subagent execution results by agent ID.
// Accessed via agent://<id> internal URLs.
//
// Usage:
//   agent://<id>                          — full result as JSON
//   agent://<id>/content                  — the final output text
//   agent://<id>/status                   — completion status
//   agent://<id>/details.metrics.turns    — dot-notation nested access

type AgentOutputEntry = {
	agentId: string;
	agent: string;
	content: string;
	status: string;
	details: Record<string, unknown>;
	timestamp: number;
};

let _registry: AgentOutputRegistry | undefined;

export class AgentOutputRegistry {
	#store = new Map<string, AgentOutputEntry>();

	private constructor() {}

	static instance(): AgentOutputRegistry {
		if (!_registry) {
			_registry = new AgentOutputRegistry();
		}
		return _registry;
	}

	static resetForTests(): void {
		_registry = undefined;
	}

	/** Record a subagent result. Called by spawn_agent tool on completion. */
	store(entry: AgentOutputEntry): void {
		this.#store.set(entry.agentId, entry);
	}

	/** Retrieve a subagent result by ID. */
	get(agentId: string): AgentOutputEntry | undefined {
		return this.#store.get(agentId);
	}

	/** List all stored agent IDs. */
	ids(): string[] {
		return [...this.#store.keys()];
	}

	/** Remove a specific entry. */
	remove(agentId: string): boolean {
		return this.#store.delete(agentId);
	}
}

/**
 * Resolve a dot-notation path into a value from an object.
 * Supports array index access (e.g. findings.0.path) and nested keys.
 * Returns undefined if the path doesn't exist.
 */
export function resolvePath(obj: unknown, path: string): unknown {
	if (!path || obj === null || obj === undefined) return obj;

	// Split on dots, but handle numeric indices as array access
	const segments = path.split(".");
	let current: unknown = obj;

	for (const segment of segments) {
		if (current === null || current === undefined) return undefined;

		// Array index access
		if (Array.isArray(current)) {
			const index = Number(segment);
			if (!Number.isNaN(index) && index >= 0 && index < current.length) {
				current = current[index];
				continue;
			}
			return undefined;
		}

		// Object property access
		if (typeof current === "object" && segment in current) {
			current = (current as Record<string, unknown>)[segment];
			continue;
		}

		return undefined;
	}

	return current;
}
