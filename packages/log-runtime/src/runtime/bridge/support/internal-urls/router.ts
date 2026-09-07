// ── Internal URL router ─────────────────────────────────────────────────────
// Process-global router with one handler per scheme.
// Access via InternalUrlRouter.instance().

import type { InternalResource, InternalUrl, ProtocolHandler, ResolveContext } from "./types";

let _instance: InternalUrlRouter | undefined;

export class InternalUrlRouter {
	#handlers = new Map<string, ProtocolHandler>();

	private constructor() {}

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

	canResolve(input: string): boolean {
		const handler = this.#handlers.get(extractUriScheme(input) ?? "");
		return handler !== undefined;
	}

	async resolve(input: string, context?: ResolveContext): Promise<InternalResource> {
		const scheme = extractUriScheme(input);
		if (!scheme) throw new Error(`Unknown scheme in: ${input}`);
		const handler = this.#handlers.get(scheme);
		if (!handler) throw new Error(`No handler for scheme: ${scheme}`);
		const url = parseInternalUrl(input);
		return handler.resolve(url, context);
	}
}

function extractUriScheme(input: string): string | undefined {
	const m = input.match(/^([a-z][a-z0-9+.-]*):\/\//i);
	return m?.[1]?.toLowerCase();
}

function parseInternalUrl(input: string): InternalUrl {
	const m = input.match(/^([a-z][a-z0-9+.-]*):\/\/([^/?#]*)/i);
	if (!m) throw new Error(`Invalid internal URL: ${input}`);
	const [, scheme, host] = m;
	const rest = input.slice(m[0].length);
	const pathname = rest.startsWith("/") ? rest : "/";
	return {
		scheme,
		host,
		rawHost: host,
		pathname,
		href: `${scheme}://${host}${pathname}`,
		search: "",
		hash: "",
		protocol: `${scheme}:`,
		port: "",
		username: "",
		password: "",
		origin: `${scheme}://${host}`,
		searchParams: new URLSearchParams(),
		hostname: host,
	};
}
