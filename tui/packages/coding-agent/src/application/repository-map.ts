import { execFileSync } from "node:child_process";
import { lstatSync, readFileSync } from "node:fs";
import path from "node:path";

interface RepositoryMapEntry {
	file: string;
	imports: string[];
	symbols: string[];
	mtimeMs: number;
	size: number;
}

export interface RepositoryMapOptions {
	maxTokens?: number;
	maxFileBytes?: number;
}

const SOURCE_EXTENSIONS = new Set([
	".c",
	".cc",
	".cpp",
	".cs",
	".go",
	".java",
	".js",
	".jsx",
	".kt",
	".php",
	".py",
	".rb",
	".rs",
	".swift",
	".ts",
	".tsx",
]);

const SYMBOL_PATTERNS = [
	/^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:class|function|interface|type|enum|namespace)\s+([A-Za-z_$][\w$]*)/gm,
	/^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?=[:=])/gm,
	/^\s*(?:def|class)\s+([A-Za-z_][\w]*)/gm,
	/^\s*(?:pub\s+)?(?:async\s+)?(?:fn|struct|enum|trait|type|mod)\s+([A-Za-z_][\w]*)/gm,
];

const IMPORT_PATTERN =
	/(?:from\s+|import\s*(?:\([^)]*\)\s*from\s*)?|require\s*\()\s*["']([^"']+)["']/g;

/** Change-refreshed, query-ranked repository context with a hard token budget. */
export class RepositoryMap {
	private readonly maxTokens: number;
	private readonly maxFileBytes: number;
	private readonly cache = new Map<string, RepositoryMapEntry>();

	constructor(
		private readonly cwd: string,
		options: RepositoryMapOptions = {},
	) {
		this.maxTokens = Math.max(128, options.maxTokens ?? 2_000);
		this.maxFileBytes = Math.max(1_024, options.maxFileBytes ?? 256_000);
	}

	render(query = ""): string {
		const files = this.listFiles();
		const live = new Set(files);
		for (const cached of this.cache.keys()) {
			if (!live.has(cached)) this.cache.delete(cached);
		}
		const entries = files
			.map(file => this.entryFor(file))
			.filter((entry): entry is RepositoryMapEntry => entry !== undefined)
			.sort(
				(a, b) =>
					this.score(b, query) - this.score(a, query) ||
					a.file.localeCompare(b.file),
			);

		const maxChars = this.maxTokens * 4;
		const header = "<repository-map>\n";
		const footer = "</repository-map>";
		let output = header;
		for (const entry of entries) {
			const details = [
				entry.symbols.length ? `symbols: ${entry.symbols.join(", ")}` : "",
				entry.imports.length ? `imports: ${entry.imports.join(", ")}` : "",
			].filter(Boolean);
			const line = `${entry.file}${details.length ? ` — ${details.join("; ")}` : ""}\n`;
			if (output.length + line.length + footer.length > maxChars) continue;
			output += line;
		}
		return output === header ? "" : `${output}${footer}`;
	}

	private listFiles(): string[] {
		try {
			return execFileSync("git", ["ls-files", "-co", "--exclude-standard"], {
				cwd: this.cwd,
				encoding: "utf8",
				stdio: ["ignore", "pipe", "ignore"],
			})
				.split("\n")
				.filter(file =>
					SOURCE_EXTENSIONS.has(path.extname(file).toLowerCase()),
				);
		} catch {
			return [];
		}
	}

	private entryFor(file: string): RepositoryMapEntry | undefined {
		try {
			const absolute = path.join(this.cwd, file);
			const stat = lstatSync(absolute);
			if (
				stat.isSymbolicLink() ||
				!stat.isFile() ||
				stat.size > this.maxFileBytes
			)
				return undefined;
			const cached = this.cache.get(file);
			if (
				cached &&
				cached.mtimeMs === stat.mtimeMs &&
				cached.size === stat.size
			)
				return cached;
			const source = readFileSync(absolute, "utf8");
			const symbols = new Set<string>();
			for (const pattern of SYMBOL_PATTERNS) {
				pattern.lastIndex = 0;
				for (const match of source.matchAll(pattern)) {
					if (match[1]) symbols.add(match[1]);
				}
			}
			IMPORT_PATTERN.lastIndex = 0;
			const imports = new Set<string>();
			for (const match of source.matchAll(IMPORT_PATTERN)) {
				if (match[1]) imports.add(match[1]);
			}
			const entry = {
				file,
				imports: [...imports].slice(0, 12),
				symbols: [...symbols].slice(0, 24),
				mtimeMs: stat.mtimeMs,
				size: stat.size,
			};
			this.cache.set(file, entry);
			return entry;
		} catch {
			return undefined;
		}
	}

	private score(entry: RepositoryMapEntry, query: string): number {
		const terms = query.toLowerCase().match(/[a-z_$][\w$-]{2,}/g) ?? [];
		const haystack =
			`${entry.file} ${entry.symbols.join(" ")} ${entry.imports.join(" ")}`.toLowerCase();
		return terms.reduce(
			(score, term) => score + (haystack.includes(term) ? 1 : 0),
			0,
		);
	}
}
