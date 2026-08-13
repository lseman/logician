import { fileURLToPath } from "node:url";
import tsPlugin from "@typescript-eslint/eslint-plugin";
import tsParser from "@typescript-eslint/parser";

// This config lives in apps/tui/ (not repo root), alongside its ESLint
// and typescript-eslint dependencies. `basePath` re-anchors file globs at the
// repo root so this config still lints every package, not just tui.
const repoRoot = fileURLToPath(new URL("../..", import.meta.url));

export default [
	{
		basePath: repoRoot,
	},
	{
		ignores: ["dist/**", "node_modules/**", "*.js"],
	},
	{
		files: ["packages/**/*.ts"],
		languageOptions: {
			parser: tsParser,
			parserOptions: {
				ecmaVersion: 2023,
				sourceType: "module",
				project: ["packages/*/tsconfig.json"],
				tsconfigRootDir: repoRoot,
			},
		},
		plugins: {
			"@typescript-eslint": tsPlugin,
		},
		rules: {
			// ── Errors ──────────────────────────────────────────────────
			"@typescript-eslint/no-unused-vars": [
				"error",
				{
					argsIgnorePattern: "^_",
					varsIgnorePattern: "^_",
					caughtErrorsIgnorePattern: "^_",
				},
			],
			"@typescript-eslint/no-explicit-any": "warn",
			"@typescript-eslint/no-non-null-assertion": "warn",
			"@typescript-eslint/no-floating-promises": "error",
			"@typescript-eslint/no-misused-promises": "error",

			// ── Styling ─────────────────────────────────────────────────
			semi: ["error", "always"],
			quotes: ["error", "double"],
			// Single-line if/else without braces is consistent across the codebase.
			// "curly": ["error", "all"],
			eqeqeq: ["error", "always"],
			"no-console": "warn",
			"no-duplicate-imports": "error",
			"prefer-const": "error",

			// ── Complexity ──────────────────────────────────────────────
			"@typescript-eslint/max-params": ["warn", { max: 7 }],
			"max-lines-per-function": [
				"warn",
				{ max: 300, skipBlankLines: true, skipComments: true },
			],
		},
	},
	// Test files: test runner manages async lifecycle, no-floating-promises doesn't apply
	{
		files: ["packages/**/*.test.ts"],
		rules: {
			"@typescript-eslint/no-floating-promises": "off",
		},
	},
];
