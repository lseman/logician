// ── TTSR Built-In Rules ───────────────────────────────────────────────────────
// Default rules that detect harmful patterns mid-stream and inject course
// corrections. Each rule is a TtsrRule definition with conditions, scope,
// and interrupt behavior.
//
// NOTE: conditions are string patterns compiled to RegExp by TtsrManager.

import type { TtsrRule } from "../types/ttsr.ts";

// ── Secret Exposure ───────────────────────────────────────────────────────────

export const RULE_SECRET_EXPOSURE: TtsrRule = {
	name: "secret-exposure",
	path: "@logician/log-core/built-in-rules",
	description:
		"Detects hardcoded secrets, API keys, tokens, or passwords in source files.",
	content:
		"Do not write secrets, API keys, tokens, or passwords into source code. Use environment variables or a secret manager instead. If you need credentials, reference them via process.env or the project's secret store.",
	conditions: [
		// Generic secret patterns in assignments or object literals
		"(?:password|passwd|pwd)\\s*[:=]\\s*[\"'][^\"']{8,}[\"']",
		// API keys and tokens
		"(?:api[_-]?key|apikey|secret[_-]?key|access[_-]?token|auth[_-]?token)\\s*[:=]\\s*[\"'][A-Za-z0-9+/=]{16,}[\"']",
		// Hardcoded bearer tokens
		"bearer\\s+[A-Za-z0-9\\-._~+/]+=*",
		// AWS/GCP/Azure credential patterns
		"(?:AKIA|GCP|AIza)[A-Za-z0-9\\-_]{10,}",
	],
	scope: ["text", "tool"],
	interruptMode: "always",
	builtin: true,
};

// ── Test Skipping ─────────────────────────────────────────────────────────────

export const RULE_TEST_SKIP: TtsrRule = {
	name: "test-skip",
	path: "@logician/log-core/built-in-rules",
	description:
		"Detects when the agent skips, disables, or ignores tests during implementation.",
	content:
		"Do not skip or disable tests. If a test is failing, fix the root cause rather than adding skip/xit/describe.skip. Tests are a safety net — removing them makes future regressions more likely.",
	conditions: [
		// Jest/Mocha skip patterns
		"\\bx(?:it|describe|test)\\b",
		"\\.skip\\s*\\(",
		// xit standalone
		"\\bxit\\b",
		// ts-ignore comments used to silence failing tests
		"\\/\\/\\s*@ts-ignore.*test",
	],
	scope: ["text", "tool"],
	interruptMode: "prose-only",
	builtin: true,
};

// ── Unsafe Shell Commands ─────────────────────────────────────────────────────

export const RULE_UNSAFE_SHELL: TtsrRule = {
	name: "unsafe-shell",
	path: "@logician/log-core/built-in-rules",
	description:
		"Detects dangerous shell commands that could cause data loss or system damage.",
	content:
		"Avoid destructive shell commands. Never use 'rm -rf /' or 'sudo rm -rf'. Use targeted file operations with explicit paths. If you need to clean a directory, list its contents first and confirm before deleting.",
	conditions: [
		// Catastrophic rm patterns
		"rm\\s+-rf\\s+\\/\\s*$",
		"rm\\s+-rf\\s+[.][.]\\/[ ]*$",
		// Dangerous sudo + rm combos
		"sudo\\s+rm\\s+-rf\\b",
		// dd with wrong of= (potential data wipe)
		"dd\\s+.*of=\\/dev\\/(sd|nvme|vd)",
	],
	scope: ["tool:bash"],
	interruptMode: "always",
	builtin: true,
};

// ── Overly Broad File Operations ──────────────────────────────────────────────

export const RULE_BROAD_FILE_WRITE: TtsrRule = {
	name: "broad-file-write",
	path: "@logician/log-core/built-in-rules",
	description:
		"Detects file writes that target system directories or the project root indiscriminately.",
	content:
		"Be specific about where you write files. Avoid writing to /etc, /usr, /root, or other system directories. Within the project, prefer writing to known subdirectories (src/, lib/, app/) rather than the project root.",
	conditions: [
		// Writes to system directories
		"write.*(?:\\/etc\\/|\\/usr\\/local\\/|\\/root\\/)",
		// Glob writes that could match anything
		"write.*\\*{3,}",
	],
	scope: ["tool"],
	globs: ["**"],
	interruptMode: "prose-only",
	builtin: true,
};

// ── Missing Error Handling ────────────────────────────────────────────────────

export const RULE_MISSING_ERROR_HANDLING: TtsrRule = {
	name: "missing-error-handling",
	path: "@logician/log-core/built-in-rules",
	description:
		"Detects async operations without error handling — bare awaits or .then() without catch.",
	content:
		"Always handle errors in async operations. Use try/catch blocks around await expressions, or chain .catch() after .then(). Unhandled promise rejections cause silent failures and are hard to debug.",
	conditions: [
		// Bare await fetch (with optional preceding identifier)
		"await\\s+(\\w+\\.)?fetch\\(",
		// .then() without .catch()
		"\\.then\\s*\\([^)]*\\)\\s*(?!.*\\.catch)",
	],
	scope: ["text", "thinking"],
	interruptMode: "prose-only",
	builtin: true,
};

// ── Security Anti-Patterns ────────────────────────────────────────────────────

export const RULE_SECURITY_ANTIPATTERN: TtsrRule = {
	name: "security-antipattern",
	path: "@logician/log-core/built-in-rules",
	description:
		"Detects dangerous JavaScript patterns like eval(), innerHTML, or SQL injection vectors.",
	content:
		"Avoid dangerous security anti-patterns. Do not use eval(), Function() constructor, or innerHTML with user-controlled data. Use parameterized queries for database operations and sanitize all user input.",
	conditions: [
		// eval() usage
		"\\beval\\s*\\(",
		// new Function()
		"new\\s+Function\\s*\\(",
		// innerHTML assignment
		"\\.innerHTML\\s*=",
		// SQL string concatenation (common injection vector)
		"(?:SELECT|INSERT|UPDATE|DELETE)\\s+.*\\+\\s*\\w+",
	],
	scope: ["text", "tool"],
	interruptMode: "prose-only",
	builtin: true,
};

// ── Documentation Gap ─────────────────────────────────────────────────────────

export const RULE_DOC_GAP: TtsrRule = {
	name: "doc-gap",
	path: "@logician/log-core/built-in-rules",
	description:
		"Detects when the agent claims to document something but doesn't actually produce documentation.",
	content:
		"If you mention documenting code, updating README, or writing comments, actually do it. Don't say 'I'll add documentation' and then move on without producing any docs. Documentation is part of the work, not an afterthought.",
	conditions: [
		// TODO/FIXME with document keyword
		"\\/\\/\\s*(TODO|FIXME|HACK|XXX)\\b.*(?:document|doc)",
		// Promise to add documentation
		"(?:will\\s+)?(?:add|update|write|create).*(?:doc|readme|comment|documentation)",
	],
	scope: ["text", "thinking"],
	interruptMode: "never",
	builtin: true,
};

// ── Rule Registry ─────────────────────────────────────────────────────────────

/** All built-in TTSR rules. */
export const BUILTIN_RULES: TtsrRule[] = [
	RULE_SECRET_EXPOSURE,
	RULE_TEST_SKIP,
	RULE_UNSAFE_SHELL,
	RULE_BROAD_FILE_WRITE,
	RULE_MISSING_ERROR_HANDLING,
	RULE_SECURITY_ANTIPATTERN,
	RULE_DOC_GAP,
];

/** Get built-in rules by name for quick lookup. */
export function getBuiltInRuleByName(name: string): TtsrRule | undefined {
	return BUILTIN_RULES.find(r => r.name === name);
}

/** Check if a rule name is a built-in rule. */
export function isBuiltInRule(name: string): boolean {
	return BUILTIN_RULES.some(r => r.name === name);
}
