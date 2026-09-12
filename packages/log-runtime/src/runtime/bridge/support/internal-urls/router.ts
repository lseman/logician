// ── Internal URL router ─────────────────────────────────────────────────────
// Process-global router with one handler per scheme.
// Access via InternalUrlRouter.instance().

import { extractInternalUrlScheme, parseInternalUrl } from "./parse";
import type {
	InternalResource,
	ProtocolHandler,
	ResolveContext,
} from "./types";

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
		const handler = this.#handlers.get(extractInternalUrlScheme(input) ?? "");
		return handler !== undefined;
	}

	async resolve(
		input: string,
		context?: ResolveContext,
	): Promise<InternalResource> {
		const scheme = extractInternalUrlScheme(input);
		if (!scheme) throw new Error(`Unknown scheme in: ${input}`);
		const handler = this.#handlers.get(scheme);
		if (!handler)
			throw new Error(`Unsupported internal URL scheme: ${scheme}://`);
		const url = parseInternalUrl(input);
		return handler.resolve(url, context);
	}
}
