import {
	synthesizeTurnEpisode,
	type TurnEvidence,
} from "./episode-synthesizer.js";
import type {
	CompressedObservation,
	ObservationType,
	RawObservation,
} from "./types.js";

export interface SemanticExtractionRequest {
	systemPrompt: string;
	userPrompt: string;
}

export type SemanticExtractor = (
	request: SemanticExtractionRequest,
) => Promise<string | unknown>;

export interface SemanticExtractionResult {
	raw: RawObservation;
	compressed: CompressedObservation;
	source: "model" | "deterministic";
	rejectionReason?: string;
}

const KINDS = new Set<ObservationType>([
	"implementation",
	"bugfix",
	"decision",
	"discovery",
	"error",
	"conversation",
]);
const STATUSES = new Set(["tentative", "verified", "invalidated"]);
const GENERIC =
	/^(?:updated|fixed|implemented|completed|worked on|made changes|ran tests|investigated)(?:\s+(?:files?|code|issue|it|this))?[.!]?$/i;

function parseJson(value: unknown): unknown {
	if (typeof value !== "string") return value;
	const trimmed = value
		.trim()
		.replace(/^```(?:json)?\s*/i, "")
		.replace(/\s*```$/, "");
	try {
		return JSON.parse(trimmed);
	} catch {
		return null;
	}
}

function strings(value: unknown, max: number): string[] | null {
	if (!Array.isArray(value) || value.some(item => typeof item !== "string"))
		return null;
	return [...new Set(value.map(item => item.trim()).filter(Boolean))].slice(
		0,
		max,
	);
}

function extractionPrompt(evidence: TurnEvidence): SemanticExtractionRequest {
	const events = evidence.tools.map(tool => ({
		evidence_id: tool.id,
		tool: tool.name,
		input: tool.args,
		outcome: tool.result.slice(0, 6000),
		failed: tool.isError,
	}));
	return {
		systemPrompt: `You are a technical memory extractor. Return exactly one JSON object and no prose.
Record durable knowledge, not a narration of tool usage. Every claim must be self-contained, specific, and grounded in supplied evidence IDs.
Never invent files, symbols, commands, outcomes, or verification. A claim may be "verified" only when a supplied check completed successfully.
Skip trivia by returning {"skip":true}.
Schema: {"kind":"implementation|bugfix|decision|discovery|error|conversation","title":string,"summary":string,"claims":[{"text":string,"confidence":number_0_to_1,"status":"tentative|verified|invalidated","evidenceEventIds":string[]}],"rationale":string_optional,"outcome":string_optional,"filesRead":string[],"filesModified":string[],"concepts":string[]}`,
		userPrompt: JSON.stringify({
			user_intent: evidence.userIntent,
			assistant_outcome: evidence.assistantOutcome,
			workspace: evidence.workspace,
			evidence_events: [
				{
					evidence_id: "user-intent",
					kind: "user_intent",
					content: evidence.userIntent,
				},
				{
					evidence_id: "assistant-outcome",
					kind: "assistant_outcome",
					content: evidence.assistantOutcome,
				},
				...events,
			],
		}),
	};
}

function validateModelEpisode(
	value: unknown,
	evidence: TurnEvidence,
): { episode?: CompressedObservation; reason?: string; skipped?: boolean } {
	const parsed = parseJson(value);
	if (!parsed || typeof parsed !== "object" || Array.isArray(parsed))
		return { reason: "invalid JSON object" };
	const obj = parsed as Record<string, unknown>;
	if (obj.skip === true) return { skipped: true };
	if (typeof obj.kind !== "string" || !KINDS.has(obj.kind as ObservationType))
		return { reason: "invalid kind" };
	if (
		typeof obj.title !== "string" ||
		obj.title.trim().length < 12 ||
		GENERIC.test(obj.title.trim())
	)
		return { reason: "vague title" };
	if (
		typeof obj.summary !== "string" ||
		obj.summary.trim().length < 24 ||
		GENERIC.test(obj.summary.trim())
	)
		return { reason: "vague summary" };
	if (
		!Array.isArray(obj.claims) ||
		obj.claims.length === 0 ||
		obj.claims.length > 8
	)
		return { reason: "invalid claims" };

	const allowedEvidence = new Set([
		"user-intent",
		"assistant-outcome",
		...evidence.tools.map(tool => tool.id),
	]);
	const successfulVerification = new Set(
		evidence.tools
			.filter(
				tool =>
					!tool.isError &&
					/(?:test|check|lint|typecheck|build|compile|verify|pytest|vitest|jest|cargo|go test)/i.test(
						`${tool.name} ${JSON.stringify(tool.args)}`,
					) &&
					!/\b[1-9]\d*\s+(?:fail(?:ed|ure)?|errors?)\b|\b(?:tests?|checks?)\s+failed\b/i.test(
						tool.result,
					),
			)
			.map(tool => tool.id),
	);
	const groundedPaths = new Set<string>();
	for (const tool of evidence.tools) {
		for (const [key, item] of Object.entries(tool.args)) {
			if (
				/(?:path|file|filename|target)/i.test(key) &&
				typeof item === "string"
			)
				groundedPaths.add(item);
		}
	}

	const facts: string[] = [];
	for (const rawClaim of obj.claims) {
		if (!rawClaim || typeof rawClaim !== "object" || Array.isArray(rawClaim))
			return { reason: "invalid claim object" };
		const claim = rawClaim as Record<string, unknown>;
		const ids = strings(claim.evidenceEventIds, 12);
		if (
			typeof claim.text !== "string" ||
			claim.text.trim().length < 16 ||
			GENERIC.test(claim.text.trim())
		)
			return { reason: "vague claim" };
		if (
			typeof claim.confidence !== "number" ||
			claim.confidence < 0 ||
			claim.confidence > 1
		)
			return { reason: "invalid confidence" };
		if (typeof claim.status !== "string" || !STATUSES.has(claim.status))
			return { reason: "invalid claim status" };
		if (!ids?.length || ids.some(id => !allowedEvidence.has(id)))
			return { reason: "ungrounded evidence ID" };
		if (
			claim.status === "verified" &&
			!ids.some(id => successfulVerification.has(id))
		)
			return { reason: "unsupported verified claim" };
		facts.push(
			`[${claim.status}; confidence=${claim.confidence.toFixed(2)}; evidence=${ids.join(",")}] ${claim.text.trim()}`,
		);
	}

	const filesRead = strings(obj.filesRead ?? [], 20);
	const filesModified = strings(obj.filesModified ?? [], 20);
	const concepts = strings(obj.concepts ?? [], 10);
	if (!filesRead || !filesModified || !concepts)
		return { reason: "invalid string arrays" };
	if ([...filesRead, ...filesModified].some(file => !groundedPaths.has(file)))
		return { reason: "ungrounded file" };

	const id = `episode:${crypto.randomUUID()}`;
	const rationale =
		typeof obj.rationale === "string" ? obj.rationale.trim() : "";
	const outcome = typeof obj.outcome === "string" ? obj.outcome.trim() : "";
	return {
		episode: {
			id,
			sessionId: evidence.sessionId,
			timestamp: evidence.timestamp,
			type: obj.kind as ObservationType,
			title: obj.title.trim().slice(0, 200),
			subtitle: obj.summary.trim().slice(0, 300),
			facts,
			narrative: [
				obj.summary.trim(),
				rationale && `Rationale: ${rationale}`,
				outcome && `Outcome: ${outcome}`,
			]
				.filter(Boolean)
				.join("\n\n")
				.slice(0, 4000),
			concepts,
			files: [...new Set([...filesModified, ...filesRead])],
			importance:
				obj.kind === "decision" || obj.kind === "bugfix"
					? 9
					: obj.kind === "implementation"
						? 8
						: 7,
			consolidated: false,
			workspace: evidence.workspace,
		},
	};
}

export async function extractSemanticEpisode(
	evidence: TurnEvidence,
	extractor?: SemanticExtractor,
): Promise<SemanticExtractionResult | null> {
	const fallback = synthesizeTurnEpisode(evidence);
	if (!extractor)
		return fallback ? { ...fallback, source: "deterministic" } : null;
	try {
		const output = await extractor(extractionPrompt(evidence));
		const validated = validateModelEpisode(output, evidence);
		if (validated.skipped) return null;
		if (!validated.episode)
			return fallback
				? {
						...fallback,
						source: "deterministic",
						rejectionReason: validated.reason,
					}
				: null;
		return {
			raw: {
				id: validated.episode.id,
				sessionId: evidence.sessionId,
				timestamp: evidence.timestamp,
				hookType: "stop",
				userPrompt: evidence.userIntent,
				workspace: evidence.workspace,
				raw: { kind: "semantic_episode", extraction_source: "model", evidence },
			},
			compressed: validated.episode,
			source: "model",
		};
	} catch (error) {
		return fallback
			? {
					...fallback,
					source: "deterministic",
					rejectionReason:
						error instanceof Error ? error.message : String(error),
				}
			: null;
	}
}
