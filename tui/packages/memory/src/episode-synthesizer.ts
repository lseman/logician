import type {
	CompressedObservation,
	ObservationType,
	RawObservation,
} from "./types.js";

export interface TurnToolEvidence {
	id: string;
	name: string;
	args: Record<string, unknown>;
	result: string;
	isError: boolean;
}

export interface TurnEvidence {
	sessionId: string;
	timestamp: string;
	workspace: string;
	userIntent: string;
	assistantOutcome: string;
	tools: TurnToolEvidence[];
}

const VERIFICATION_TOOL =
	/(?:test|check|lint|typecheck|build|compile|verify|pytest|vitest|jest|cargo|go test)/i;
const WRITE_TOOL = /(?:edit|write|patch|append|overwrite|notebookedit)/i;
const READ_TOOL = /(?:read|search|grep|find|open|fetch)/i;

function cleanLine(value: string): string {
	return (
		value
			.split(/\r?\n/)
			.map(line =>
				line
					.trim()
					.replace(/^#{1,6}\s*/, "")
					.replace(/^[-*]\s+/, ""),
			)
			.find(line => line.length >= 12 && !/^```/.test(line))
			?.replace(/[*_`]/g, "")
			.slice(0, 180) || ""
	);
}

function collectPaths(value: unknown, output: Set<string>, depth = 0): void {
	if (depth > 6 || value == null) return;
	if (typeof value === "string") {
		for (const match of value.matchAll(
			/(?:^|[\s"'`])((?:\.?\.?\/|\/)?[\w.@+-]+(?:\/[\w.@+-]+)+\.[\w+-]+)/g,
		)) {
			output.add(match[1]?.replace(/[,:;)]+$/, ""));
		}
		return;
	}
	if (Array.isArray(value)) {
		value.forEach(item => collectPaths(item, output, depth + 1));
		return;
	}
	if (typeof value === "object") {
		for (const [key, item] of Object.entries(
			value as Record<string, unknown>,
		)) {
			if (
				/(?:path|file|filename|target)/i.test(key) &&
				typeof item === "string" &&
				item.trim()
			) {
				output.add(item.trim());
			}
			collectPaths(item, output, depth + 1);
		}
	}
}

function inferEpisodeType(
	evidence: TurnEvidence,
	changed: boolean,
	failed: boolean,
): ObservationType {
	const text =
		`${evidence.userIntent} ${evidence.assistantOutcome}`.toLowerCase();
	if (failed && !changed) return "error";
	if (/\b(?:decid|chose|choice|trade-?off|rationale|architecture)\b/.test(text))
		return "decision";
	if (
		/\b(?:bug|fix|broken|regression|crash|incorrect|failure)\b/.test(text) &&
		changed
	)
		return "bugfix";
	if (changed) return "implementation";
	if (
		/\b(?:discover|found|identified|confirmed|traced|root cause|learned)\b/.test(
			text,
		)
	)
		return "discovery";
	return "conversation";
}

/** Convert a complete turn into one grounded, future-useful semantic episode. */
export function synthesizeTurnEpisode(evidence: TurnEvidence): {
	raw: RawObservation;
	compressed: CompressedObservation;
} | null {
	const intent = evidence.userIntent.trim();
	const outcome = evidence.assistantOutcome.trim();
	if (!intent || (!outcome && evidence.tools.length === 0)) return null;

	const paths = new Set<string>();
	evidence.tools.forEach(tool => collectPaths(tool.args, paths));
	const changedTools = evidence.tools.filter(tool =>
		WRITE_TOOL.test(tool.name),
	);
	const failedTools = evidence.tools.filter(tool => tool.isError);
	const verification = evidence.tools.filter(tool =>
		VERIFICATION_TOOL.test(`${tool.name} ${JSON.stringify(tool.args)}`),
	);
	const readTools = evidence.tools.filter(tool => READ_TOOL.test(tool.name));
	const type = inferEpisodeType(
		evidence,
		changedTools.length > 0,
		failedTools.length > 0,
	);
	const outcomeLine = cleanLine(outcome);
	const intentLine = cleanLine(intent) || intent.slice(0, 160);
	const title =
		outcomeLine &&
		!/^(?:done|completed|implemented|fixed|updated)\.?$/i.test(outcomeLine)
			? outcomeLine
			: `${type === "bugfix" ? "Fixed" : type === "implementation" ? "Implemented" : "Learned"}: ${intentLine}`;

	const facts: string[] = [`User intent: ${intentLine}`];
	if (outcomeLine) facts.push(`Outcome: ${outcomeLine}`);
	if (paths.size)
		facts.push(
			`${changedTools.length ? "Relevant files" : "Files inspected"}: ${[...paths].slice(0, 12).join(", ")}`,
		);
	if (verification.length) {
		const passed = verification.filter(
			tool =>
				!tool.isError &&
				!/\b[1-9]\d*\s+(?:fail(?:ed|ure)?|errors?)\b|\b(?:tests?|checks?)\s+failed\b/i.test(
					tool.result,
				),
		);
		facts.push(
			`Verification: ${passed.length}/${verification.length} checks completed without reported failure (${verification.map(tool => tool.name).join(", ")})`,
		);
	}
	if (failedTools.length)
		facts.push(
			`Failed evidence: ${failedTools.map(tool => `${tool.name}: ${cleanLine(tool.result) || "failed"}`).join("; ")}`,
		);

	const concepts = new Set<string>();
	if (changedTools.length) concepts.add("what-changed");
	if (type === "bugfix" || failedTools.length) concepts.add("problem-solution");
	if (type === "decision") concepts.add("trade-off");
	if (readTools.length) concepts.add("how-it-works");
	if (verification.length) concepts.add("verified");

	const id = `episode:${crypto.randomUUID()}`;
	const narrative = [
		`Request: ${intent.slice(0, 1200)}`,
		outcome ? `Result: ${outcome.slice(0, 2400)}` : "",
	]
		.filter(Boolean)
		.join("\n\n");
	const importance =
		type === "decision" || type === "bugfix"
			? 9
			: changedTools.length
				? 8
				: failedTools.length
					? 8
					: 6;
	const compressed: CompressedObservation = {
		id,
		sessionId: evidence.sessionId,
		timestamp: evidence.timestamp,
		type,
		title,
		subtitle: `${evidence.tools.length} tool events; ${changedTools.length} mutations; ${verification.length} verification checks`,
		facts,
		narrative,
		concepts: [...concepts],
		files: [...paths].slice(0, 20),
		importance,
		consolidated: false,
		workspace: evidence.workspace,
	};
	return {
		raw: {
			id,
			sessionId: evidence.sessionId,
			timestamp: evidence.timestamp,
			hookType: "stop",
			userPrompt: intent,
			workspace: evidence.workspace,
			raw: { kind: "semantic_episode", ...evidence },
		},
		compressed,
	};
}
