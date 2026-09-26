import { createRequire } from "node:module";
import { loadNative } from "../shared/native-addon.ts";

const require = createRequire(import.meta.url);
const exportsByCapability = {
	ast: "astMatch",
	editing: "EditSession",
	search: "grep",
	glob: "glob",
	fuzzyFind: "fuzzyFind",
	diff: "diffLines",
	tokens: "countTokens",
	snapcompact: "renderSnapcompactPng",
} as const;

export async function inspectNative() {
	try {
		const native = await loadNative();
		const capabilities = Object.fromEntries(
			Object.entries(exportsByCapability).map(([name, symbol]) => [
				name,
				typeof native[symbol] === "function",
			]),
		);
		const paths = Object.keys(require.cache).filter(
			path => path.endsWith(".node") && path.includes("log-natives"),
		);
		return {
			loaded: true,
			healthy: Object.values(capabilities).every(Boolean),
			paths,
			capabilities,
			error: null,
		};
	} catch (error) {
		return {
			loaded: false,
			healthy: false,
			paths: [],
			capabilities: {},
			error: error instanceof Error ? error.message : String(error),
		};
	}
}
