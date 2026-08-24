export interface ContextSelectionCandidate {
	id: string;
	score: number;
	tokens: number;
	recency: number;
	sourceKey: string;
	similarityText: string;
}

export interface ContextSelectionOptions {
	budget: number;
	maxItems?: number;
	relevanceWeight?: number;
	preferredItemsPerSource?: number;
}

const TOKEN_PATTERN = /[\p{L}\p{N}_-]{2,}/gu;
const STOP_WORDS = new Set([
	"a",
	"an",
	"and",
	"are",
	"as",
	"at",
	"be",
	"by",
	"for",
	"from",
	"in",
	"is",
	"it",
	"of",
	"on",
	"or",
	"the",
	"this",
	"to",
	"with",
]);

function terms(value: string): Set<string> {
	return new Set(
		(value.normalize("NFKC").toLowerCase().match(TOKEN_PATTERN) || []).filter(
			token => !STOP_WORDS.has(token),
		),
	);
}

function jaccard(left: Set<string>, right: Set<string>): number {
	if (!left.size || !right.size) return 0;
	let intersection = 0;
	for (const term of left) if (right.has(term)) intersection++;
	return intersection / (left.size + right.size - intersection);
}

function preferredOver<T extends ContextSelectionCandidate>(
	candidate: T,
	current: T | undefined,
): boolean {
	if (!current) return true;
	if (candidate.score !== current.score) return candidate.score > current.score;
	const efficiency = candidate.score / candidate.tokens;
	const currentEfficiency = current.score / current.tokens;
	if (efficiency !== currentEfficiency) return efficiency > currentEfficiency;
	if (candidate.recency !== current.recency)
		return candidate.recency > current.recency;
	return candidate.id < current.id;
}

/**
 * Selects prompt context with maximal marginal relevance (MMR).
 *
 * Relevance chooses the first item; subsequent rounds trade a small amount of
 * relevance for novel evidence. Source diversity is a soft preference so a
 * single well-supported source can still fill otherwise-unused budget.
 */
export function selectContextCandidates<T extends ContextSelectionCandidate>(
	candidates: readonly T[],
	options: ContextSelectionOptions,
): T[] {
	const budget = Math.max(0, options.budget);
	const maxItems = Math.max(0, options.maxItems ?? 40);
	const relevanceWeight = Math.min(
		1,
		Math.max(0, options.relevanceWeight ?? 0.72),
	);
	const preferredItemsPerSource = Math.max(
		1,
		options.preferredItemsPerSource ?? 2,
	);
	const remaining = candidates.filter(
		candidate => candidate.tokens > 0 && candidate.tokens <= budget,
	);
	if (!remaining.length || maxItems === 0) return [];

	const maximumScore = Math.max(...remaining.map(candidate => candidate.score));
	const minimumScore = Math.min(...remaining.map(candidate => candidate.score));
	const normalizedScore = (candidate: T): number => {
		if (minimumScore >= 0 && maximumScore > 0)
			return candidate.score / maximumScore;
		if (maximumScore === minimumScore) return 1;
		return (candidate.score - minimumScore) / (maximumScore - minimumScore);
	};
	const candidateTerms = new Map(
		remaining.map(candidate => [candidate, terms(candidate.similarityText)]),
	);
	const sourceCounts = new Map<string, number>();
	const selected: T[] = [];
	let usedTokens = 0;

	while (remaining.length && selected.length < maxItems) {
		let bestIndex = -1;
		let bestUtility = Number.NEGATIVE_INFINITY;

		for (let index = 0; index < remaining.length; index++) {
			const candidate = remaining[index];
			if (usedTokens + candidate.tokens > budget) continue;
			const termsForCandidate =
				candidateTerms.get(candidate) ?? new Set<string>();
			let similarity = 0;
			for (const chosen of selected) {
				similarity = Math.max(
					similarity,
					jaccard(
						termsForCandidate,
						candidateTerms.get(chosen) ?? new Set<string>(),
					),
				);
			}
			const sourceCount = sourceCounts.get(candidate.sourceKey) || 0;
			const sourcePenalty =
				sourceCount * 0.025 +
				Math.max(0, sourceCount - preferredItemsPerSource + 1) * 0.1;
			const utility =
				relevanceWeight * normalizedScore(candidate) -
				(1 - relevanceWeight) * similarity -
				sourcePenalty;
			if (
				utility > bestUtility ||
				(utility === bestUtility &&
					preferredOver(candidate, remaining[bestIndex]))
			) {
				bestIndex = index;
				bestUtility = utility;
			}
		}

		if (bestIndex < 0) break;
		const [best] = remaining.splice(bestIndex, 1);
		selected.push(best);
		usedTokens += best.tokens;
		sourceCounts.set(
			best.sourceKey,
			(sourceCounts.get(best.sourceKey) || 0) + 1,
		);
	}

	return selected;
}
