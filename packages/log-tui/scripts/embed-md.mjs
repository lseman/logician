#!/usr/bin/env node
// Embeds a markdown file as a TypeScript template-literal constant, so the
// compiled binary doesn't need to read the .md file at runtime.
//
// Usage: node embed-md.mjs <source.md> <dest.ts> <CONST_NAME>

import { readFileSync, writeFileSync } from "node:fs";

const [, , src, dest, constName] = process.argv;
if (!src || !dest || !constName) {
	console.error("Usage: node embed-md.mjs <source.md> <dest.ts> <CONST_NAME>");
	process.exit(1);
}

const raw = readFileSync(src, "utf8");
const escaped = raw
	.replaceAll("\\", "\\\\")
	.replaceAll("`", "\\`")
	.replaceAll("${", "\\${");

writeFileSync(
	dest,
	`// Auto-generated from ${src} at build time\nexport const ${constName} = \`${escaped}\`;\n`,
);
