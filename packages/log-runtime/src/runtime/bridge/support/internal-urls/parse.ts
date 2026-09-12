// ── Internal URL parser ─────────────────────────────────────────────────────
// Handles colons in the host segment (e.g. skill://plugin:name).
// Uses regex instead of new URL() which interprets colons as port separators.

import type { InternalUrl } from "./types";

const SCHEME_HOST_RE = /^([a-z][a-z0-9+.-]*):\/\/([^/?#]*)/i;
const OPAQUE_URI_RE = /^([a-z][a-z0-9+.-]*):(.+)$/is;
const SELECTOR_CHUNK_SRC = String.raw`(?:raw|conflicts|-?\d+(?:[-+]\d+)?(?:,\d+(?:[-+]\d+)?)*)`;
const SELECTOR_CHAIN_RE = new RegExp(
	`^${SELECTOR_CHUNK_SRC}(?::${SELECTOR_CHUNK_SRC})*$`,
	"i",
);

/** Extract the lowercased scheme from a URI-shaped input. */
export function extractUriScheme(input: string): string | undefined {
	const hierarchical = input.match(SCHEME_HOST_RE);
	if (hierarchical) return hierarchical[1].toLowerCase();
	const opaque = input.match(OPAQUE_URI_RE);
	if (!opaque) return undefined;
	const [, scheme] = opaque;
	if (scheme.length === 1) return undefined;
	if (scheme.includes(".")) return undefined;
	if (SELECTOR_CHAIN_RE.test(opaque[2])) return undefined;
	return scheme.toLowerCase();
}

/** Parse an internal URL into a simple URL-like object. */
export function parseInternalUrl(input: string): InternalUrl {
	const match = input.match(SCHEME_HOST_RE);
	if (!match) throw new Error(`Invalid internal URL: ${input}`);

	const [, scheme, host] = match;
	const rest = input.slice(match[0].length);
	const pathname = rest.startsWith("/") ? rest : "/";
	const href = `${scheme}://${host}${pathname}`;

	return {
		scheme,
		host,
		rawHost: host,
		pathname,
		href,
		toString: () => href,
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
