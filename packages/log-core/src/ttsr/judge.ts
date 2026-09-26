// ── TTSR judged rules ─────────────────────────────────────────────────────────
// Judged rules carry a yes/no `question` instead of (or gated by) a stream
// pattern. After an assistant output completes, every eligible question is
// asked about it in ONE judge request, so the shared output is sent once.
// A "yes" becomes a non-interrupting warning — the output already took effect.
//
// The judge is any text-completion function; the runtime binds it to the
// session's model. Verdicts fail closed: a reply that doesn't parse flags
// nothing, so a flaky judge can only miss violations, never invent them.

import type { JudgedCandidate, TtsrOutput, TtsrRule } from "../types/ttsr.ts";

/** Text-completion function the judge runs on (system + user prompt → reply). */
export type TtsrJudge = (request: {
	system: string;
	user: string;
	signal?: AbortSignal | undefined;
}) => Promise<string>;

/**
 * Characters of output content sent per judgment. Keeps a judge request well
 * inside small context windows; the head of an output is where rule-relevant
 * decisions (approach, file contents) land.
 */
export const JUDGED_CONTENT_MAX_CHARS = 48_000;

const JUDGE_SYSTEM = [
	"You are a strict rule judge for a coding agent.",
	"You are shown one output the agent produced and numbered yes/no questions about it.",
	'Answer each question "yes" only when the output clearly satisfies it; otherwise "no".',
	'Reply with ONLY a JSON object mapping each question id to "yes" or "no", e.g. {"q0":"no","q1":"yes"}.',
].join("\n");

/** Longest prefix of `text` within `maxChars`, never ending on half a surrogate pair. */
function prefix(text: string, maxChars: number): string {
	if (text.length <= maxChars) return text;
	const last = text.charCodeAt(maxChars - 1);
	const end = last >= 0xd800 && last <= 0xdbff ? maxChars - 1 : maxChars;
	return `${text.slice(0, end)}\n[… output truncated for judging]`;
}

/** Build the judge prompt for one output and its candidate questions. */
export function buildJudgeRequest(
	output: TtsrOutput,
	candidates: readonly JudgedCandidate[],
): { system: string; user: string } {
	const questions = candidates
		.map((candidate, index) => `q${index}: ${candidate.question}`)
		.join("\n");
	const user = [
		`Output (${output.subject}):`,
		"<output>",
		prefix(output.content, JUDGED_CONTENT_MAX_CHARS),
		"</output>",
		"",
		"Questions:",
		questions,
	].join("\n");
	return { system: JUDGE_SYSTEM, user };
}

/**
 * Parse the judge's reply into the indices answered "yes". Accepts the JSON
 * object anywhere in the reply (models wrap it in prose or code fences);
 * anything unparseable yields no verdicts.
 */
export function parseJudgeVerdicts(reply: string, count: number): number[] {
	const start = reply.indexOf("{");
	const end = reply.lastIndexOf("}");
	if (start === -1 || end <= start) return [];
	let parsed: unknown;
	try {
		parsed = JSON.parse(reply.slice(start, end + 1));
	} catch {
		return [];
	}
	if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return [];
	const answers = parsed as Record<string, unknown>;
	const flagged: number[] = [];
	for (let index = 0; index < count; index++) {
		const answer = answers[`q${index}`];
		const yes =
			answer === true ||
			(typeof answer === "string" && answer.trim().toLowerCase() === "yes");
		if (yes) flagged.push(index);
	}
	return flagged;
}

/**
 * Ask every candidate's question about `output` in one judge request and
 * return the rules judged violated. Judge failures propagate to the caller.
 */
export async function judgeRules(
	judge: TtsrJudge,
	output: TtsrOutput,
	candidates: readonly JudgedCandidate[],
	signal?: AbortSignal,
): Promise<TtsrRule[]> {
	if (candidates.length === 0) return [];
	const request = buildJudgeRequest(output, candidates);
	const reply = await judge({ ...request, signal });
	return parseJudgeVerdicts(reply, candidates.length).flatMap(index => {
		const candidate = candidates[index];
		return candidate ? [candidate.rule] : [];
	});
}
