import { execFile } from "node:child_process";
import { chmodSync, renameSync, writeFileSync } from "node:fs";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

export const EOH_BEGIN = "# EOH-BEGIN";
export const EOH_END = "# EOH-END";

export interface EohFileTarget {
	path: string;
	source: string;
	prefix: string;
	heuristicCode: string;
	suffix: string;
	functionSignature: string;
	description: string;
	mode: number;
}

function markerRange(source: string): {
	begin: number;
	contentStart: number;
	end: number;
	contentEnd: number;
} {
	const begin = source.indexOf(EOH_BEGIN);
	const end = source.indexOf(EOH_END);
	if (begin < 0 || end < 0 || end <= begin) {
		throw new Error(
			`The file must contain one ${EOH_BEGIN} / ${EOH_END} region`,
		);
	}
	if (
		source.indexOf(EOH_BEGIN, begin + EOH_BEGIN.length) >= 0 ||
		source.indexOf(EOH_END, end + EOH_END.length) >= 0
	) {
		throw new Error("The file must contain exactly one EoH region");
	}
	const beginLineEnd = source.indexOf("\n", begin);
	if (beginLineEnd < 0 || beginLineEnd >= end) {
		throw new Error(`${EOH_BEGIN} must be followed by heuristic code`);
	}
	const contentStart = beginLineEnd + 1;
	const contentEnd = source.lastIndexOf("\n", end);
	if (contentEnd < contentStart) {
		throw new Error("The EoH heuristic region is empty");
	}
	return { begin, contentStart, end, contentEnd };
}

export async function loadEohFile(
	rawPath: string,
	cwd: string,
): Promise<EohFileTarget> {
	const targetPath = path.resolve(cwd, rawPath);
	if (path.extname(targetPath).toLowerCase() !== ".py") {
		throw new Error("EoH currently requires a Python (.py) file");
	}
	const [source, metadata] = await Promise.all([
		readFile(targetPath, "utf8"),
		stat(targetPath),
	]);
	const range = markerRange(source);
	const heuristicCode = source
		.slice(range.contentStart, range.contentEnd)
		.trim();
	const functionMatch =
		/^\s*def\s+([A-Za-z_]\w*)\s*(\([^]*?\))\s*(?:->\s*[^:]+)?\s*:/m.exec(
			heuristicCode,
		);
	if (!functionMatch) {
		throw new Error("The EoH region must define a Python function");
	}
	if (functionMatch[1] !== "heuristic") {
		throw new Error("The EoH region must define heuristic(...)");
	}
	if (!/\bdef\s+evaluate\s*\(/.test(source.slice(range.end + EOH_END.length))) {
		throw new Error(
			"The file must define evaluate(heuristic) after the EoH region",
		);
	}
	const description =
		/^([\s\S]*?)(?=\n\s*(?:from|import|def|class|# EOH-BEGIN))/m
			.exec(source)?.[1]
			.trim() ||
		`Improve the heuristic in ${path.basename(targetPath)}. Higher evaluation scores are better.`;
	return {
		path: targetPath,
		source,
		prefix: source.slice(0, range.contentStart),
		heuristicCode,
		suffix: source.slice(range.contentEnd),
		functionSignature: `def ${functionMatch[1]}${functionMatch[2]}:`,
		description,
		mode: metadata.mode,
	};
}

export function renderEohCandidate(
	target: EohFileTarget,
	heuristicCode: string,
): string {
	return `${target.prefix}${heuristicCode.trim()}\n${target.suffix.replace(/^\n*/, "\n")}`;
}

const EVALUATE_SCRIPT = [
	"import json, math, runpy, sys",
	"ns = runpy.run_path(sys.argv[1])",
	"heuristic = ns.get('heuristic')",
	"evaluate = ns.get('evaluate')",
	"if not callable(heuristic):",
	"    raise TypeError('EOH region must define callable heuristic')",
	"if not callable(evaluate):",
	"    raise TypeError('file must define callable evaluate(heuristic)')",
	"score = float(evaluate(heuristic))",
	"if not math.isfinite(score):",
	"    raise ValueError('evaluate() returned a non-finite score')",
	"print('__LOGICIAN_EOH_SCORE__' + json.dumps(score))",
].join("\n");

export async function evaluateEohCandidate(
	target: EohFileTarget,
	heuristicCode: string,
	timeoutMs: number,
): Promise<number> {
	const tempDir = await mkdtemp(path.join(os.tmpdir(), "logician-eoh-"));
	const candidatePath = path.join(tempDir, path.basename(target.path));
	try {
		await writeFile(candidatePath, renderEohCandidate(target, heuristicCode), {
			encoding: "utf8",
			mode: target.mode,
		});
		const python = process.env.EOH_PYTHON?.trim() || "python3";
		const { stdout } = await execFileAsync(
			python,
			["-c", EVALUATE_SCRIPT, candidatePath],
			{
				cwd: path.dirname(target.path),
				timeout: timeoutMs,
				maxBuffer: 4 * 1024 * 1024,
			},
		);
		const scoreLine = stdout
			.split(/\r?\n/)
			.findLast(line => line.startsWith("__LOGICIAN_EOH_SCORE__"));
		if (!scoreLine)
			throw new Error("evaluate() did not produce a fitness score");
		const score = Number(scoreLine.slice("__LOGICIAN_EOH_SCORE__".length));
		if (!Number.isFinite(score)) {
			throw new Error("evaluate() returned a non-finite score");
		}
		return score;
	} finally {
		await rm(tempDir, { recursive: true, force: true });
	}
}

export function applyEohCandidate(
	target: EohFileTarget,
	heuristicCode: string,
): void {
	const output = renderEohCandidate(target, heuristicCode);
	const temporaryPath = `${target.path}.eoh-${process.pid}.tmp`;
	writeFileSync(temporaryPath, output, {
		encoding: "utf8",
		mode: target.mode,
	});
	chmodSync(temporaryPath, target.mode);
	renameSync(temporaryPath, target.path);
	target.source = output;
	target.heuristicCode = heuristicCode.trim();
}
