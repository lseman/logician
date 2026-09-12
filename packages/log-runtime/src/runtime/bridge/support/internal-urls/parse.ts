// Internal resource names may contain colons (skill://plugin:name), so they
// must not be interpreted as browser URL authorities or port numbers.
import type { InternalUrl } from "./types";

const INTERNAL_URL_RE = /^([a-z][a-z0-9+.-]*):\/\/([^/?#]*)/i;

/** Recognize links independently of whether a handler is registered. */
export function extractInternalUrlScheme(input: string): string | undefined {
	const scheme = input.match(INTERNAL_URL_RE)?.[1];
	// Keep Windows drive paths such as C://work/file on the filesystem path.
	return scheme && scheme.length > 1 ? scheme.toLowerCase() : undefined;
}

/** Preserve the entire target; individual protocols interpret its contents. */
export function parseInternalUrl(input: string): InternalUrl {
	const scheme = extractInternalUrlScheme(input);
	const match = input.match(INTERNAL_URL_RE);
	if (!scheme || !match) throw new Error(`Invalid internal URL: ${input}`);
	const host = match[2];
	const target = input.slice(scheme.length + 3);
	const rest = target.slice(host.length);
	return {
		scheme,
		target,
		host,
		pathname: rest.startsWith("/") ? rest : "/",
		href: `${scheme}://${target}`,
	};
}
