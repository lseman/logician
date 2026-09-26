// ── Syntax highlighter ────────────────────────────────────────────────────────
// Wraps emphasize (wooorm) → ANSI output. Supports 190+ languages, auto-detect,
// chalk-style sheet. Falls back to plain grey when no grammar matches.

import {
	type AutoOptions,
	common,
	createEmphasize,
	type Sheet,
} from "emphasize";

// ANSI sheet: maps highlight.js token classes to 256-color ANSI sequences.
// Designed for dark terminals — high contrast, readable at any size.
const DARK_SHEET: Record<string, (t: string) => string> = {
	// Keywords (import, export, const, function, return, etc.)
	keyword: t => `\x1b[38;5;141m${t}\x1b[39m`, // cyan
	// Strings (quotes, template literals)
	string: t => `\x1b[38;5;114m${t}\x1b[39m`, // green
	// Comments
	comment: t => `\x1b[38;5;245m${t}\x1b[39m`, // grey
	// Numbers
	number: t => `\x1b[38;5;179m${t}\x1b[39m`, // gold
	// Functions
	function: t => `\x1b[38;5;111m${t}\x1b[39m`, // bright green
	// Class names
	"class name": t => `\x1b[38;5;81m${t}\x1b[39m`, // cyan
	// Built-in types
	built_in: t => `\x1b[38;5;220m${t}\x1b[39m`, // yellow
	// Literals (true, false, null, undefined)
	literal: t => `\x1b[38;5;203m${t}\x1b[39m`, // red
	// Punctuation
	punctuation: t => `\x1b[38;5;244m${t}\x1b[39m`, // dark grey
	// Operators
	operator: t => `\x1b[38;5;147m${t}\x1b[39m`, // periwinkle
	// Regular expressions
	regex: t => `\x1b[38;5;208m${t}\x1b[39m`, // orange
	// Template expressions
	"template expression": t => `\x1b[38;5;147m${t}\x1b[39m`, // periwinkle
	// Attributes (HTML/XML)
	attribute: t => `\x1b[38;5;179m${t}\x1b[39m`, // gold
	// Doctypes
	doctype: t => `\x1b[38;5;245m${t}\x1b[39m`, // grey
	// XML tags
	"xml tag": t => `\x1b[38;5;141m${t}\x1b[39m`, // cyan
	// Properties
	property: t => `\x1b[38;5;111m${t}\x1b[39m`, // bright green
	// Meta
	meta: t => `\x1b[38;5;245m${t}\x1b[39m`, // grey
	// Shebang
	shebang: t => `\x1b[38;5;245m${t}\x1b[39m`, // grey
	// Subst (template substitutions)
	subst: t => `\x1b[38;5;208m${t}\x1b[39m`, // orange
	// Symbol
	symbol: t => `\x1b[38;5;179m${t}\x1b[39m`, // gold
	// Type
	type: t => `\x1b[38;5;81m${t}\x1b[39m`, // cyan
	// Params
	params: t => `\x1b[38;5;220m${t}\x1b[39m`, // yellow
	// Title
	title: t => `\x1b[38;5;111m${t}\x1b[39m`, // bright green
	// Section
	section: t => `\x1b[38;5;81m${t}\x1b[39m`, // cyan
	// Link
	link: t => `\x1b[38;5;111m${t}\x1b[39m`, // bright green
	// Code
	code: t => `\x1b[38;5;114m${t}\x1b[39m`, // green
	// Additions
	addition: t => `\x1b[38;5;114m${t}\x1b[39m`, // green
	// Deletions
	deletion: t => `\x1b[38;5;203m${t}\x1b[39m`, // red
};

// Common grammars subset (37 languages) — covers 99% of use cases.
const GRAMMARS = common;

let instance: ReturnType<typeof createEmphasize>;

try {
	instance = createEmphasize(GRAMMARS);
} catch (_e: unknown) {
	// Fallback: no highlighting if emphasize fails to initialize.
	instance = {
		highlightAuto: () => ({ value: "", language: undefined, relevance: 0 }),
		highlight: () => ({ value: "", language: undefined, relevance: 0 }),
		listLanguages: () => [],
		register: () => {},
		registerAlias: () => {},
		registered: () => false,
	} as unknown as ReturnType<typeof createEmphasize>;
}

export interface HighlightResult {
	/** Highlighted text with ANSI sequences. */
	value: string;
	/** Detected language name (undefined if auto-detect failed). */
	language?: string;
}

// A TUI turn re-renders its whole transcript on every streamed chunk (see
// TranscriptDisplay's per-turn cache in apps/tui), which re-runs this
// highlighter over every code fence in that turn every time — including
// fences that finished streaming turns ago and whose content hasn't changed
// since. Highlighting is a pure function of (code, language), so a small
// bounded cache turns "already-closed code block" back into an O(1) hit
// instead of a full re-parse on every subsequent chunk of the same turn.
const HIGHLIGHT_CACHE_MAX = 200;
const highlightCache = new Map<string, HighlightResult>();

function cacheGet(key: string): HighlightResult | undefined {
	const hit = highlightCache.get(key);
	if (hit === undefined) return undefined;
	// Refresh insertion order so recently-used entries survive eviction.
	highlightCache.delete(key);
	highlightCache.set(key, hit);
	return hit;
}

function cacheSet(key: string, value: HighlightResult): void {
	if (highlightCache.size >= HIGHLIGHT_CACHE_MAX) {
		const oldest = highlightCache.keys().next().value;
		if (oldest !== undefined) highlightCache.delete(oldest);
	}
	highlightCache.set(key, value);
}

/**
 * Highlight code with auto-detection. Falls back to plain text.
 */
export function highlightAuto(code: string): HighlightResult {
	const key = `auto ${code}`;
	const cached = cacheGet(key);
	if (cached) return cached;
	const opts: AutoOptions = { sheet: DARK_SHEET as Sheet };
	const result = instance.highlightAuto(code, opts);
	const out: HighlightResult = {
		value: result.value ?? code,
		language: result.language,
	};
	cacheSet(key, out);
	return out;
}

/**
 * Highlight code for a specific language. Falls back to plain text.
 */
export function highlight(code: string, language: string): HighlightResult {
	const key = `${language} ${code}`;
	const cached = cacheGet(key);
	if (cached) return cached;
	const result = instance.highlight(language, code, DARK_SHEET as Sheet);
	const out: HighlightResult = {
		value: result.value ?? code,
		language: result.language,
	};
	cacheSet(key, out);
	return out;
}

/**
 * List all supported language names.
 */
