/** Pure text-analysis helpers used when creating a Memory — no db or
 * workspace dependency. */

export function extractConcepts(content: string): string[] {
	const concepts = new Set<string>();
	const keywords = [
		"error",
		"bug",
		"fix",
		"crash",
		"panic",
		"timeout",
		"retry",
		"config",
		"setting",
		"env",
		"environment",
		"auth",
		"permission",
		"access",
		"token",
		"database",
		"schema",
		"migration",
		"query",
		"connection",
		"api",
		"endpoint",
		"route",
		"middleware",
		"login",
		"test",
		"unit",
		"integration",
		"mock",
		"stub",
		"build",
		"deploy",
		"pipeline",
		"ci",
		"cd",
		"refactor",
		"optimize",
		"performance",
		"memory",
		"cpu",
		"security",
		"vulnerability",
		"sanitize",
		"escape",
		"cache",
		"index",
		"search",
		"filter",
		"sort",
		"async",
		"promise",
		"callback",
		"event",
		"listener",
		"state",
		"store",
		"redux",
		"context",
		"hook",
		"type",
		"interface",
		"class",
		"module",
		"package",
	];
	const lower = content.toLowerCase();
	for (const kw of keywords) {
		if (lower.includes(kw)) concepts.add(kw);
	}
	// Hashtags
	const hashtags = content.match(/#(\w+)/g);
	if (hashtags) {
		for (const h of hashtags) concepts.add(h.slice(1));
	}
	return [...concepts].slice(0, 10);
}

export function extractFiles(content: string): string[] {
	const files = new Set<string>();
	// Match file paths: src/foo.ts, ./lib/bar.js, ../test/baz.py
	const patterns = [
		/(?:src|lib|pkg|test|app|src|vendor|node_modules|dist|build)\//g,
		/\/[\w.-]+\.(ts|js|tsx|jsx|py|rs|go|rb|java|c|h|cpp|json|yaml|yml|toml|md|css|scss|html|sh|bash)/g,
	];
	for (const pattern of patterns) {
		let match;
		const str = content;
		while ((match = pattern.exec(str)) !== null) {
			const path = content.slice(
				Math.max(0, match.index - 50),
				match.index + match[0].length,
			);
			if (path.includes("/")) files.add(path.slice(0, 300));
		}
	}
	return [...files].slice(0, 10);
}

export function assignStrength(content: string): number {
	const lower = content.toLowerCase();
	if (/^fix|^bug|error|panic|crash|exception/i.test(lower)) return 8;
	if (/^decid|^architect|^design|^pattern/i.test(lower)) return 7;
	if (/^todo|^next|^future|suggestion/i.test(lower)) return 4;
	return 5;
}
