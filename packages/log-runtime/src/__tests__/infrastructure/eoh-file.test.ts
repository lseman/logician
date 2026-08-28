import { expect, test } from "bun:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadEohFile } from "../../runtime/eoh/file.ts";

test("EOH loader accepts a multiline typed heuristic signature", async () => {
	const directory = mkdtempSync(join(tmpdir(), "logician-eoh-file-"));
	const file = join(directory, "heuristic.py");
	writeFileSync(
		file,
		[
			"Optimize this scoring heuristic.",
			"# EOH-BEGIN",
			"def heuristic(",
			"    value: float,",
			") -> float:",
			"    return value * 2",
			"# EOH-END",
			"def evaluate(heuristic):",
			"    return heuristic(2.0)",
		].join("\n"),
		"utf8",
	);

	const target = await loadEohFile(file, directory);

	expect(target.functionSignature).toContain("def heuristic(");
	expect(target.functionSignature).toContain("value: float");
	expect(target.heuristicCode).toContain("return value * 2");
});
