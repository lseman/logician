// ── rule:// protocol handler ─────────────────────────────────────────────────
// Resolves frontmatter rule names to their content.
// URL form: rule://<name>

import type {
	InternalResource,
	InternalUrl,
	ProtocolHandler,
	ResolveContext,
} from "./types";

export class RuleProtocolHandler implements ProtocolHandler {
	readonly scheme = "rule";
	readonly immutable = true;

	async resolve(
		url: InternalUrl,
		context?: ResolveContext,
	): Promise<InternalResource> {
		const rules = context?.rules;
		const ruleName = url.host;

		if (!ruleName) {
			throw new Error(`rule:// URL requires a rule name: rule://<name>`);
		}

		if (url.target !== url.host && url.target !== `${url.host}/`) {
			throw new Error(
				"rule:// supports an exact name only; paths, queries and fragments are not supported.",
			);
		}

		const rule = rules?.find(r => r.name === ruleName);
		if (!rule) {
			const available = rules?.map(r => r.name) ?? [];
			throw new Error(
				`Unknown rule: ${ruleName}\nAvailable: ${available.join(", ") || "none"}`,
			);
		}

		const content = rule.content;
		return {
			url: url.href,
			content,
			contentType: "text/markdown",
			size: Buffer.byteLength(content, "utf-8"),
			sourcePath: rule.path,
			notes: [],
		};
	}

	async complete(
		_query: string,
		context?: ResolveContext,
	): Promise<Array<{ value: string; description?: string }>> {
		return (context?.rules ?? []).map(rule => ({
			value: rule.name,
			description: rule.name,
		}));
	}
}
