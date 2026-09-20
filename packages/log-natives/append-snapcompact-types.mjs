// Appends the snapcompact type declarations to the napi-generated
// index.d.ts. The snapcompact crate self-registers its N-API exports at load
// time (constructor-based, see crates/pi-natives/src/lib.rs), so napi codegen
// never sees them; their declarations are kept in snapcompact.d.ts and
// appended here after every build. Idempotent.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dtsPath = join(here, "index.d.ts");
const snippet = readFileSync(join(here, "snapcompact.d.ts"), "utf8");

const dts = readFileSync(dtsPath, "utf8");
if (!dts.includes("renderSnapcompactPng")) {
	writeFileSync(
		dtsPath,
		`${dts.replace(/\s*$/, "")}\n\n${snippet.replace(/\s*$/, "")}\n`,
	);
	console.log("appended snapcompact declarations to index.d.ts");
}
