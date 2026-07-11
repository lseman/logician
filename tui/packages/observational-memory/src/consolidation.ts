// ── Consolidation pipeline ───────────────────────────────────────────────
// Observer → Reflector → Dropper pipeline, triggered on turn_end.
// Runs as a background task (non-blocking).

import type { Observation, Reflection } from "./types.ts";
import { runObserver } from "./observer.ts";
import { runReflector } from "./reflector.ts";
import { runDropper } from "./dropper.ts";

export interface ConsolidationConfig {
	/** Model name or identifier */
	model: string;
	/** API key for LLM calls */
	apiKey: string;
	/** OpenAI-compatible API base URL. */
	baseUrl?: string;
	/** Optional request headers */
	headers?: Record<string, string>;
	/** Whether a UI is available for notifications */
	hasUI?: boolean;
	/** Observation pool target (tokens below which no drops occur) */
	observationsPoolTargetTokens?: number;
	/** Max turns per agent stage */
	maxTurns?: number;
	/** Thinking level for LLM calls */
	thinkingLevel?: string;
}

export interface LaunchParams {
	/** Current raw token count in session */
	currentTokens: number;
	/** Token threshold to trigger observer */
	observeThreshold: number;
	/** Token threshold to trigger reflector */
	reflectThreshold: number;
	/** Current active observations */
	observations: Observation[];
	/** Current reflections */
	reflections: Reflection[];
	/** Real session entries that have not yet been observed. */
	sourceEntries?: Array<{ id: string; role: string; content: string }>;
}

export interface ConsolidationResult {
	/** Whether any stage ran */
	ran: boolean;
	/** Observations recorded (0 if observer didn't run or returned empty) */
	observationsRecorded: number;
	/** Reflections recorded (0 if reflector didn't run or returned empty) */
	reflectionsRecorded: number;
	/** Observation IDs dropped (0 if dropper didn't run or returned empty) */
	droppedCount: number;
	observations: Observation[];
	reflections: Reflection[];
	droppedObservationIds: string[];
}

export class ConsolidationPipeline {
	private config: ConsolidationConfig;
	private inFlight: boolean = false;
	private observeThreshold: number;
	private reflectThreshold: number;

	constructor(
		config: ConsolidationConfig,
		thresholds?: { observeAfterTokens: number; reflectAfterTokens: number },
	) {
		this.config = config;
		this.observeThreshold =
			thresholds?.observeAfterTokens ??
			config.observationsPoolTargetTokens ??
			10_000;
		this.reflectThreshold =
			thresholds?.reflectAfterTokens ??
			config.observationsPoolTargetTokens ??
			20_000;
	}

	/**
	 * Check if consolidation should launch and run it asynchronously.
	 * Non-blocking: returns immediately.
	 */
	async maybeLaunch(
		params: LaunchParams,
	): Promise<ConsolidationResult | undefined> {
		if (this.inFlight) return undefined;

		const {
			currentTokens,
			observeThreshold,
			reflectThreshold,
			observations: _observations,
			reflections: _reflections,
		} = params;

		// Check if any stage is due
		const observerDue = currentTokens >= observeThreshold;
		const reflectorDue = currentTokens >= reflectThreshold;

		if (!observerDue && !reflectorDue) return undefined;

		this.inFlight = true;
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
			// Stage 1: Observer
			const observerResult = observerDue
				? await this.runObserverStage(params)
				: undefined;
			if (observerResult) {
				result.observationsRecorded = observerResult.length;
				result.observations = observerResult;
			}

			// Stage 2: Reflector
			const reflectorResult = reflectorDue
				? await this.runReflectorStage(params, observerResult)
				: undefined;
			if (reflectorResult) {
				result.reflectionsRecorded = reflectorResult.length;
				result.reflections = reflectorResult;
			}

			// Stage 3: Dropper (only after reflector)
			if (reflectorResult && reflectorResult.length > 0) {
				const dropperResult = await this.runDropperStage(params);
				if (dropperResult) {
					result.droppedCount = dropperResult.length;
					result.droppedObservationIds = dropperResult;
				}
			}
		} catch (error) {
			console.error(
				"[observational-memory] Consolidation pipeline error:",
				error,
			);
		} finally {
			this.inFlight = false;
		}

		return result;
	}

	private async runObserverStage(
		params: LaunchParams,
	): Promise<Observation[] | undefined> {
		const {
			currentTokens,
			observations: priorObsRaw,
			reflections: priorRefsRaw,
		} = params;
		if (currentTokens < this.observeThreshold) return undefined;

		// Prepare prior observation/reflection summaries
		const priorObs = priorObsRaw
			.map((o: Observation) => `[${o.relevance}] ${o.content}`)
			.slice(-20);
		const priorRefs =
			priorRefsRaw?.map((r: Reflection) => r.content).slice(-10) ?? [];

		// Build source-addressed chunk (simplified — in production, this would use actual session entries)
		const chunk = this.buildSourceChunk(params);

		const result = await runObserver({
			model: this.config.model,
			apiKey: this.config.apiKey,
			baseUrl: this.config.baseUrl,
			headers: this.config.headers,
			priorObservations: priorObs,
			priorReflections: priorRefs,
			chunk,
			allowedSourceEntryIds: (params.sourceEntries ?? []).map((entry) => entry.id),
			maxTurns: this.config.maxTurns,
			thinkingLevel: this.config.thinkingLevel,
		});

		if (!result || result.length === 0) return undefined;
		return result;
	}

	private async runReflectorStage(
		params: LaunchParams,
		_newObservations: Observation[] | undefined,
	): Promise<Reflection[] | undefined> {
		const {
			currentTokens,
			observations: priorObservations,
			reflections: refList,
		} = params;
		if (currentTokens < this.reflectThreshold) return undefined;

		// Compute coverage tiers
		const coveredObs = new Set<string>();
		for (const ref of refList) {
			for (const obsId of ref.supportingObservationIds) {
				coveredObs.add(obsId);
			}
		}

		const refObs = [...priorObservations, ...(_newObservations ?? [])];
		const obsWithCoverage = refObs.map((o: Observation) => ({
			content: o.content,
			coverage: coveredObs.has(o.id) ? ("partial" as const) : ("none" as const),
		}));

		const result = await runReflector({
			model: this.config.model,
			apiKey: this.config.apiKey,
			baseUrl: this.config.baseUrl,
			headers: this.config.headers,
			observations: obsWithCoverage,
			reflections: refList,
			maxTurns: this.config.maxTurns,
			thinkingLevel: this.config.thinkingLevel,
		});

		if (!result || result.length === 0) return undefined;
		return result;
	}

	private async runDropperStage(
		params: LaunchParams,
	): Promise<string[] | undefined> {
		const targetTokens = this.config.observationsPoolTargetTokens ?? 10_000;
		const activeTokens = params.observations.reduce(
			(sum, o) => sum + o.tokenCount,
			0,
		);

		if (activeTokens <= targetTokens) return undefined;

		const result = await runDropper({
			model: this.config.model,
			apiKey: this.config.apiKey,
			baseUrl: this.config.baseUrl,
			headers: this.config.headers,
			observations: params.observations,
			reflections: params.reflections,
			targetTokens,
			maxTurns: this.config.maxTurns,
			thinkingLevel: this.config.thinkingLevel,
		});

		return result;
	}

	private buildSourceChunk(params: LaunchParams): string {
		if (params.sourceEntries?.length) {
			return params.sourceEntries
				.map((entry) =>
					`[Source entry id: ${entry.id}]\n[${entry.role}] ${entry.content}`,
				)
				.join("\n\n");
		}
		const lines: string[] = [];
		for (const obs of params.observations.slice(-50)) {
			lines.push(`[Source entry id: ${obs.id}]`);
			lines.push(obs.content);
			lines.push("");
		}
		return lines.join("\n") || "No source entries available.";
	}
}
