import type { RAGChunk, RetrievalDiagnostics, SearchHit } from "./types.ts";

/** Weighted reciprocal-rank fusion across independent retrieval routes. */
export function fuseRankedHits(
	routes: Array<{ name: string; hits: SearchHit[]; weight?: number }>,
	options?: { rrfK?: number },
): SearchHit[] {
	const rrfK = options?.rrfK ?? 60;
	const fused = new Map<string, SearchHit>();
	for (const route of routes) {
		const weight = route.weight ?? 1;
		for (let rank = 0; rank < route.hits.length; rank++) {
			const hit = route.hits[rank];
			const prior = fused.get(hit.chunk.id);
			const contribution = weight / (rrfK + rank + 1);
			if (prior) {
				prior.score += contribution;
				prior.retrievalRoutes = [
					...new Set([...(prior.retrievalRoutes ?? []), route.name]),
				];
			} else {
				fused.set(hit.chunk.id, {
					...hit,
					score: contribution,
					retrievalRoutes: [route.name],
				});
			}
		}
	}
	return [...fused.values()].sort((a, b) => b.score - a.score);
}

function tokenSet(chunk: RAGChunk): Set<string> {
	return new Set(chunk.text.toLowerCase().match(/[\p{L}\p{N}_]+/gu) ?? []);
}

function jaccard(a: Set<string>, b: Set<string>): number {
	let intersection = 0;
	for (const value of a) if (b.has(value)) intersection++;
	const union = a.size + b.size - intersection;
	return union ? intersection / union : 0;
}

/** Maximal-marginal-relevance selection using lexical redundancy and source diversity. */
export function selectDiverseHits(
	hits: SearchHit[],
	topK: number,
	options?: { relevanceWeight?: number; sameDocumentPenalty?: number },
): SearchHit[] {
	if (hits.length <= 1 || topK <= 0) return hits.slice(0, Math.max(0, topK));
	const relevanceWeight = options?.relevanceWeight ?? 0.78;
	const sameDocumentPenalty = options?.sameDocumentPenalty ?? 0.08;
	const maxScore = Math.max(...hits.map(hit => hit.score), Number.EPSILON);
	const sets = new Map(hits.map(hit => [hit.chunk.id, tokenSet(hit.chunk)]));
	const remaining = [...hits];
	const selected: SearchHit[] = [];

	while (remaining.length && selected.length < topK) {
		let bestIndex = 0;
		let bestUtility = Number.NEGATIVE_INFINITY;
		for (let i = 0; i < remaining.length; i++) {
			const candidate = remaining[i];
			let redundancy = 0;
			let repeatsDocument = false;
			for (const chosen of selected) {
				redundancy = Math.max(
					redundancy,
					jaccard(sets.get(candidate.chunk.id)!, sets.get(chosen.chunk.id)!),
				);
				repeatsDocument ||= Boolean(
					candidate.chunk.documentId &&
						candidate.chunk.documentId === chosen.chunk.documentId,
				);
			}
			const utility =
				relevanceWeight * (candidate.score / maxScore) -
				(1 - relevanceWeight) * redundancy -
				(repeatsDocument ? sameDocumentPenalty : 0);
			if (utility > bestUtility) {
				bestUtility = utility;
				bestIndex = i;
			}
		}
		selected.push(remaining.splice(bestIndex, 1)[0]);
	}
	return selected;
}

export function diagnoseRetrieval(
	queryVariants: string[],
	candidates: SearchHit[],
	selected: SearchHit[],
): RetrievalDiagnostics {
	const agreement = selected.length
		? selected.filter(hit => (hit.retrievalRoutes?.length ?? 0) > 1).length /
			selected.length
		: 0;
	const top = selected[0];
	const signals = [
		Math.min(1, candidates.length / 10),
		agreement,
		top
			? Math.min(
					1,
					(top.retrievalRoutes?.length ?? 1) /
						Math.max(1, queryVariants.length),
				)
			: 0,
	];
	const confidence =
		signals.reduce((sum, value) => sum + value, 0) / signals.length;
	const reasons: string[] = [];
	if (!selected.length) reasons.push("no_candidates");
	if (candidates.length < 3) reasons.push("low_candidate_coverage");
	if (queryVariants.length > 1 && agreement === 0)
		reasons.push("no_route_agreement");
	return {
		queryVariants,
		candidateCount: candidates.length,
		selectedCount: selected.length,
		routeAgreement: agreement,
		confidence,
		insufficientEvidence: !selected.length || confidence < 0.25,
		reasons,
	};
}
