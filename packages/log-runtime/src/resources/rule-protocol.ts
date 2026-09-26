// ── Rule Protocol Handler ─────────────────────────────────────────────────────
// Resolves rule:// URLs to TTSR rule content. Allows the model to read rules
// on demand, similar to omp's rule:// protocol.
//
// Usage:
//   read rule://secret-exposure
//   read rule://test-skip
//   read rule://list  → lists all available rule names

import type {
	InternalResource,
	InternalUrl,
	ProtocolHandler,
	ResolveContext,
	SchemeHost,
	UrlCompletion,
} from "./types.ts";

interface RuleStore {
	rules: Map<
		string,
		{ name: string; content: string; path: string; description: string }
	>;
}

let _store: RuleStore | null = null;

export function setRuleStore(store: RuleStore): void {
	_store = store;
}

export function getRuleStore(): RuleStore | null {
	return _store;
}

export class RuleProtocolHandler implements ProtocolHandler {
	readonly scheme = "rule";
	readonly immutable = true;

	resolve(
		url: InternalUrl,
		_context?: ResolveContext,
	): Promise<InternalResource> {
		const store = _store;
		if (!store) {
			throw new Error("No rule store configured");
		}

		const target = url.pathname || url.target;

		// rule://list → list all rules
		if (target === "list" || target === "") {
			const rules = Array.from(store.rules.values());
			const lines = rules.map(r => `- ${r.name}: ${r.description} (${r.path})`);
			return Promise.resolve({
				url: `rule://list`,
				content: `## Available TTSR Rules\n\n${lines.join("\n")}\n\nRead individual rules with \`rule://<name>\`.`,
				contentType: "text/markdown",
				size: lines.join("\n").length,
			});
		}

		// rule://<name> → single rule
		const rule = store.rules.get(target);
		if (!rule) {
			const available = Array.from(store.rules.keys());
			throw new Error(
				`Unknown rule: ${target}. Available rules: ${available.join(", ")}`,
			);
		}

		return Promise.resolve({
			url: `rule://${target}`,
			content: `# ${rule.name}\n\n**Source:** ${rule.path}\n\n**Description:** ${rule.description}\n\n---\n\n${rule.content}`,
			contentType: "text/markdown",
			sourcePath: rule.path,
			size: rule.content.length,
		});
	}

	complete(query: string, _context?: ResolveContext): Promise<UrlCompletion[]> {
		const store = _store;
		if (!store) return Promise.resolve([]);

		const rules = Array.from(store.rules.values());
		const matches = query
			? rules.filter(
					r =>
						r.name.toLowerCase().includes(query.toLowerCase()) ||
						r.description.toLowerCase().includes(query.toLowerCase()),
				)
			: rules;

		return Promise.resolve(
			matches.map(r => ({
				value: r.name,
				description: r.description,
			})),
		);
	}

	promptDoc(host: SchemeHost): string | undefined {
		const store = _store;
		if (!store || store.rules.size === 0) return undefined;

		host.addressable = true;
		host.count = store.rules.size;

		const rules = Array.from(store.rules.values());
		const lines = rules.map(r => `- \`rule://${r.name}\`: ${r.description}`);
		return `## rule:// — TTSR Rule Content\n\nAccess Time-Traveling Stream Rules by name. Read rule content to understand project-specific constraints.\n\n${lines.join("\n")}`;
	}
}
