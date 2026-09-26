// ── Internal URL router ─────────────────────────────────────────────────────
// One handler per scheme. Sessions own an instance; the static accessor is
// retained for standalone tools and integrations without a session owner.
//
// Inspired by oh-my-pi's protocol specification model: handlers declare their
// capabilities via SchemeSpec so tools consult the router instead of branching
// on scheme names. The router provides introspection (describe, spec) that
// powers system prompt generation and tool approval gates.

import { extractInternalUrlScheme, parseInternalUrl } from "./parse";
import type {
	InternalResource,
	ProtocolHandler,
	ResolveContext,
	SchemeHost,
	SchemeSpec,
	UrlCompletion,
	WriteContext,
} from "./types";

let _instance: InternalUrlRouter | undefined;

export class InternalUrlRouter {
	#handlers = new Map<string, ProtocolHandler>();

	/** Process-global router instance. */
	static instance(): InternalUrlRouter {
		if (!_instance) {
			_instance = new InternalUrlRouter();
		}
		return _instance;
	}

	/** Reset the global instance in tests. */
	static resetForTests(): void {
		_instance = undefined;
	}

	register(handler: ProtocolHandler): void {
		this.#handlers.set(handler.scheme.toLowerCase(), handler);
	}

	unregister(scheme: string): boolean {
		return this.#handlers.delete(scheme.toLowerCase());
	}

	getHandler(scheme: string): ProtocolHandler | undefined {
		return this.#handlers.get(scheme.toLowerCase());
	}

	canResolve(input: string): boolean {
		const handler = this.#handlers.get(extractInternalUrlScheme(input) ?? "");
		return handler !== undefined;
	}

	async resolve(
		input: string,
		context?: ResolveContext,
	): Promise<InternalResource> {
		context?.signal?.throwIfAborted();
		const scheme = extractInternalUrlScheme(input);
		if (!scheme) throw new Error(`Unknown scheme in: ${input}`);
		const handler = this.#handlers.get(scheme);
		if (!handler)
			throw new Error(`Unsupported internal URL scheme: ${scheme}://`);
		const url = parseInternalUrl(input);
		const resource = await handler.resolve(url, context);
		context?.signal?.throwIfAborted();
		return { ...resource, immutable: resource.immutable ?? handler.immutable };
	}

	/** Write to an internal URL through its registered protocol handler. */
	async write(
		input: string,
		content: string,
		context?: WriteContext,
	): Promise<string | void> {
		context?.signal?.throwIfAborted();
		const scheme = extractInternalUrlScheme(input);
		if (!scheme) throw new Error(`Unknown scheme in: ${input}`);
		const handler = this.#handlers.get(scheme);
		if (!handler)
			throw new Error(`Unsupported internal URL scheme: ${scheme}://`);
		if (!handler.write) {
			throw new Error(
				`${scheme}:// is read-only for write; no handler supports mutation for this scheme.`,
			);
		}
		const url = parseInternalUrl(input);
		const result = await handler.write(url, content, context);
		context?.signal?.throwIfAborted();
		return result;
	}

	/** Schemes whose handler supports host/path autocomplete. */
	completionSchemes(): string[] {
		const schemes: string[] = [];
		for (const [scheme, handler] of this.#handlers) {
			if (handler.complete) schemes.push(scheme);
		}
		return schemes;
	}

	/**
	 * Candidate completions for the host/path portion of `scheme://<query>`.
	 * Returns `null` when the scheme is unknown or does not support completion.
	 */
	async complete(
		scheme: string,
		query: string,
		context?: ResolveContext,
	): Promise<UrlCompletion[] | null> {
		const handler = this.#handlers.get(scheme.toLowerCase());
		if (!handler?.complete) return null;
		return handler.complete(query, context);
	}

	// ── Introspection API (inspired by oh-my-pi) ────────────────────────────

	/** Get the scheme spec for a registered scheme. */
	spec(scheme: string): SchemeSpec | undefined {
		const handler = this.#handlers.get(scheme.toLowerCase());
		return handler?.spec;
	}

	/**
	 * Describe all registered schemes with their metadata.
	 * Returns a list of SchemeHost objects for system prompt generation.
	 */
	describe(): SchemeHost[] {
		const hosts: SchemeHost[] = [];
		for (const [scheme] of this.#handlers) {
			const host: SchemeHost = { scheme };
			// Handlers can populate addressable/count via a callback or by examining context.
			// For now, return basic info; handlers that need dynamic counts should
			// set them in their promptDoc implementation.
			hosts.push(host);
		}
		return hosts;
	}

	/**
	 * Build the system prompt fragment for all addressable schemes.
	 * Calls each handler's promptDoc with its SchemeHost, and only includes
	 * schemes where promptDoc returns a non-empty string.
	 */
	buildPromptFragment(): string {
		const parts: string[] = [];
		for (const host of this.describe()) {
			const handler = this.#handlers.get(host.scheme);
			if (!handler || !handler.promptDoc) continue;
			const doc = handler.promptDoc(host);
			if (doc && doc.trim()) parts.push(doc);
		}
		return parts.join("\n\n");
	}

	/** Get all registered scheme names. */
	schemes(): string[] {
		return Array.from(this.#handlers.keys());
	}

	/** Check if a specific scheme is registered. */
	hasScheme(scheme: string): boolean {
		return this.#handlers.has(scheme.toLowerCase());
	}
}
