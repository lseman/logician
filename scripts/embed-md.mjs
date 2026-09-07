// Build-time script: embeds a .md file as a TypeScript string export.
// Usage: node scripts/embed-md.mjs <source.md> <output.ts> [varName]

import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

const mdPath = process.argv[2];
const outPath = process.argv[3];
const varName = process.argv[4] || "EMBEDDED_CONTENT";

if (!mdPath || !outPath) {
  console.error("Usage: node scripts/embed-md.mjs <source.md> <output.ts> [varName]");
  process.exit(1);
}

const content = readFileSync(mdPath, "utf-8");
const escaped = content
  .replace(/\\/g, "\\\\")
  .replace(/`/g, "\\`")
  .replace(/\$\{/g, "\\${");

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, `// Auto-generated from ${mdPath} at build time\nexport const ${varName} = \`${escaped}\`;\n`);
console.log(`Embedded ${content.length} chars from ${mdPath} -> ${outPath}`);
