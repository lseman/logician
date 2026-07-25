import { runDropper } from "./dropper.ts";
import { runObserver } from "./observer.ts";
import { runReflector } from "./reflector.ts";
import type { Observation, Reflection } from "../types.ts";

export interface ConsolidationConfig {
	model: string;
	apiKey: string;
	baseUrl?: string;
	headers?: Record<string, string>;
	hasUI?: boolean;
	observationsPoolTargetTokens?: number;
	thinkingLevel?: string;
}

export interface SourceEntry {
	id: string;
	role: string;
	content: string;
	tokenCount?: number;
}

export interface LaunchParams {
	observeDue: boolean;
	reflectDue: boolean;
	observations: Observation[];
	reflections: Reflection[];
	sourceEntries?: SourceEntry[];
}

export type ConsolidationStage = "observer" | "reflector" | "dropper";

export interface ConsolidationResult {
	ran: boolean;
	stage?: ConsolidationStage;
	observationsRecorded: number;
	reflectionsRecorded: number;
	droppedCount: number;
	observations: Observation[];
	reflections: Reflection[];
	droppedObservationIds: string[];
}

export interface ConsolidationStatus {
	inFlight: boolean;
	stage?: ConsolidationStage;
	lastRunAt?: string;
	lastError?: string;
}

const RELEVANCE_DROP_RANK = {
	low: 0,
	medium: 1,
	high: 2,
	critical: 3,
} as const;

export class ConsolidationPipeline {
	private readonly config: ConsolidationConfig;
	private inFlight = false;
	private abortController?: AbortController;
	private status: ConsolidationStatus = { inFlight: false };

	constructor(config: ConsolidationConfig) {
		this.config = config;
	}

	getStatus(): ConsolidationStatus {
		return { ...this.status };
	}

	cancel(): void {
		this.abortController?.abort();
	}

	async maybeLaunch(
		params: LaunchParams,
	): Promise<ConsolidationResult | undefined> {
		if (this.inFlight) return undefined;
		const observerDue = params.observeDue;
		const reflectorDue = params.reflectDue;
		if (!observerDue && !reflectorDue) return undefined;

		this.inFlight = true;
		this.abortController = new AbortController();
		const result: ConsolidationResult = {
			ran: true,
			observationsRecorded: 0,
			reflectionsRecorded: 0,
			droppedCount: 0,
			observations: [],
			reflections: [],
			droppedObservationIds: [],
		};

		try {
			// Observation has priority so reflection never works from a stale pool.
			if (observerDue) {
				result.stage = "observer";
				this.status = { inFlight: true, stage: "observer" };
				const observations = await this.runObserverStage(params);
				if (observations) {
					result.observations = observations;
					result.observationsRecorded = observations.length;
				}
				return result;
			}

			result.stage = "reflector";
			this.status = { inFlight: true, stage: "reflector" };
			const reflections = await this.runReflectorStage(params);
			if (!reflections?.length) return result;
			result.reflections = reflections;
			result.reflectionsRecorded = reflections.length;

			const currentReflections = mergeReflections(
				params.reflections,
				reflections,
			);
			const maxDrops = maxDropCountForPool(
				params.observations,
				this.config.observationsPoolTargetTokens ?? 10_000,
			);
			if (maxDrops === 0) return result;

			result.stage = "dropper";
			this.status = { inFlight: true, stage: "dropper" };
			const proposals = await runDropper({
				model: this.config.model,
				apiKey: this.config.apiKey,
				baseUrl: this.config.baseUrl,
				headers: this.config.headers,
				observations: params.observations,
				reflections: currentReflections,
				targetTokens: this.config.observationsPoolTargetTokens ?? 10_000,
				thinkingLevel: this.config.thinkingLevel,
				signal: this.abortController.signal,
			});
			result.droppedObservationIds = selectDropCandidates(
				proposals ?? [],
				params.observations,
				currentReflections,
				maxDrops,
			);
			result.droppedCount = result.droppedObservationIds.length;
			return result;
		} catch (error) {
			if (this.abortController.signal.aborted) return result;
			const message = error instanceof Error ? error.message : String(error);
			this.status = {
				inFlight: false,
				stage: result.stage,
				lastRunAt: new Date().toISOString(),
				lastError: message,
			};
			throw error;
		} finally {
			this.inFlight = false;
			if (!this.status.lastError) {
				this.status = {
					inFlight: false,
					stage: result.stage,
					lastRunAt: new Date().toISOString(),
				};
			}
			this.abortController = undefined;
		}
	}

	private async runObserverStage(
		params: LaunchParams,
	): Promise<Observation[] | undefined> {
		const chunk = (params.sourceEntries ?? [])
			.map(
				(entry) =>
					`[Source entry id: ${entry.id}]\n[${entry.role}] ${entry.content}`,
			)
			.join("\n\n");
		if (!chunk) return undefined;
		return runObserver({
			model: this.config.model,
			apiKey: this.config.apiKey,
			baseUrl: this.config.baseUrl,
			headers: this.config.headers,
			priorObservations: params.observations
				.slice(-20)
				.map((item) => `[${item.relevance}] ${item.content}`),
			priorReflections: params.reflections
				.slice(-10)
				.map((item) => item.content),
			chunk,
			allowedSourceEntryIds: (params.sourceEntries ?? []).map(
				(entry) => entry.id,
			),
			thinkingLevel: this.config.thinkingLevel,
			signal: this.abortController?.signal,
		});
	}

	private async runReflectorStage(
		params: LaunchParams,
	): Promise<Reflection[] | undefined> {
		const coverage = reflectionCoverage(
			params.observations,
			params.reflections,
		);
		return runReflector({
			model: this.config.model,
			apiKey: this.config.apiKey,
			baseUrl: this.config.baseUrl,
			headers: this.config.headers,
			observations: params.observations.map((item) => ({
				id: item.id,
				content: item.content,
				timestamp: item.timestamp,
				relevance: item.relevance,
				coverage: coverage.get(item.id) ?? "none",
			})),
			reflections: params.reflections,
			thinkingLevel: this.config.thinkingLevel,
			signal: this.abortController?.signal,
		});
	}
}

export function maxDropCountForPool(
	observations: readonly Observation[],
	targetTokens: number,
): number {
	const total = observations.reduce((sum, item) => sum + item.tokenCount, 0);
	if (observations.length === 0 || targetTokens < 0 || total <= targetTokens)
		return 0;
	const average = total / observations.length;
	return Math.min(
		observations.length,
		Math.max(1, Math.ceil((total - targetTokens) / average)),
	);
}

export function selectDropCandidates(
	proposedIds: readonly string[],
	observations: readonly Observation[],
	reflections: readonly Reflection[],
	maxDrops: number,
): string[] {
	if (maxDrops <= 0) return [];
	const byId = new Map(observations.map((item) => [item.id, item]));
	const coverage = reflectionCoverage(observations, reflections);
	const unique = Array.from(new Set(proposedIds))
		.map((id, proposalIndex) => ({
			id,
			proposalIndex,
			observation: byId.get(id),
		}))
		.filter(
			(
				item,
			): item is {
				id: string;
				proposalIndex: number;
				observation: Observation;
			} => item.observation !== undefined,
		);
	return unique
		.sort((a, b) => {
			const coverageDelta =
				coverageRank(coverage.get(a.id)) - coverageRank(coverage.get(b.id));
			const relevanceDelta =
				RELEVANCE_DROP_RANK[a.observation.relevance] -
				RELEVANCE_DROP_RANK[b.observation.relevance];
			const ageDelta =
				timestampRank(a.observation.timestamp) -
				timestampRank(b.observation.timestamp);
			return (
				coverageDelta ||
				relevanceDelta ||
				ageDelta ||
				a.proposalIndex - b.proposalIndex
			);
		})
		.slice(0, maxDrops)
		.map((item) => item.id);
}

function reflectionCoverage(
	observations: readonly Observation[],
	reflections: readonly Reflection[],
): Map<string, "partial" | "strong"> {
	const valid = new Set(observations.map((item) => item.id));
	const counts = new Map<string, number>();
	for (const reflection of reflections) {
		for (const id of new Set(reflection.supportingObservationIds)) {
			if (valid.has(id)) counts.set(id, (counts.get(id) ?? 0) + 1);
		}
	}
	return new Map(
		Array.from(counts, ([id, count]) => [
			id,
			count >= 2 ? "strong" : "partial",
		]),
	);
}

function mergeReflections(
	existing: readonly Reflection[],
	additional: readonly Reflection[],
): Reflection[] {
	const merged = new Map(existing.map((item) => [item.id, item]));
	for (const item of additional) merged.set(item.id, item);
	return Array.from(merged.values());
}

function coverageRank(value: "partial" | "strong" | undefined): number {
	if (value === "strong") return 2;
	if (value === "partial") return 1;
	return 0;
}

function timestampRank(value: string): number {
	const parsed = Date.parse(value);
	return Number.isFinite(parsed) ? parsed : Number.POSITIVE_INFINITY;
}
