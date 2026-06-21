import tsPlugin from "@typescript-eslint/eslint-plugin";
import tsParser from "@typescript-eslint/parser";

export default [
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
				tsconfigRootDir: import.meta.dirname,
			},
		},
		plugins: {
			"@typescript-eslint": tsPlugin,
		},
		rules: {
			// ── Errors ──────────────────────────────────────────────────
			"@typescript-eslint/no-unused-vars": [
				"error",
				{ argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
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
];
