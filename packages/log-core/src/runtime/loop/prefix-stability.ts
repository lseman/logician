/**
 * Prompt-prefix stability tracking.
 *
 * Hosted providers report prompt-cache hits in usage; local backends
 * (llama.cpp, local gateways) report nothing, so the only client-side signal
 * for how much of the previous turn's prefix is still warm is a per-message
 * content digest compared across provider requests. This tracker maintains
 * exactly that: a digest of the last sent payload (chat messages + tool
 * definitions) and the divergence report for the current one.
 *
 * The digest is content-based (same FNV-1a fingerprint legroom's prefix memo
 * uses), not identity-based: the loop rebuilds chat messages on every call,
 * so object identity cannot be used.
 *
 * Divergence semantics (append-only log invariants):
 * - append-only growth → stable, prefix intact;
 * - in-place rewrite of message i → `divergedAt: i`;
 * - payload shrink (compaction) → `rewritten: true` — expected and separately
 *   emitted as a compaction event, not a silent divergence;
 * - tool-spec change → `divergedAt: 0, rewritten: true` — the spec is part
 *   of the provider's cache key, so a changed spec invalidates everything.
 */

export interface PrefixDivergence {
	/** True when the current payload fully shares the previously recorded prefix. */
	stable: boolean;
	/**
	 * Index of the first message whose bytes changed relative to the previous
	 * payload; equal to `previousLength` on a pure append; 0 when the tool
	 * spec changed or the payload was rewritten.
	 */
	divergedAt: number;
	/** Message count of the previously recorded payload (0 on the first record). */
	previousLength: number;
	/** True when the payload shrank (compaction) or the tool spec changed. */
	rewritten: boolean;
}

/** FNV-1a 32-bit — cheap content fingerprint, matching legroom's prefix memo. */
function fnv1a(input: string): number {
	let hash = 0x811c9dc5;
	for (let i = 0; i < input.length; i++) {
		hash ^= input.charCodeAt(i);
		hash = Math.imul(hash, 0x01000193);
	}
	return hash >>> 0;
}

function fingerprint(
	messages: readonly Record<string, unknown>[],
): Uint32Array {
	const hashes = new Uint32Array(messages.length);
	for (let i = 0; i < messages.length; i++) {
		hashes[i] = fnv1a(JSON.stringify(messages[i]));
	}
	return hashes;
}

/**
 * One run's prefix-stability state. Created per agent run (alongside
 * `createProviderTurnState`) and fed once per provider request, before the
 * request goes out, with the exact chat messages + tool definitions that
 * will be sent.
 */
export class PrefixStabilityTracker {
	#messageDigests: Uint32Array<ArrayBufferLike> = new Uint32Array(0);
	#toolDigest = 0;
	#recorded = false;

	record(
		chatMessages: readonly Record<string, unknown>[],
		toolDefinitions: readonly Record<string, unknown>[],
	): PrefixDivergence {
		const toolDigest = fnv1a(JSON.stringify(toolDefinitions));
		const previousLength = this.#messageDigests.length;
		const digests = fingerprint(chatMessages);
		let report: PrefixDivergence;
		if (!this.#recorded) {
			report = {
				stable: true,
				divergedAt: 0,
				previousLength: 0,
				rewritten: false,
			};
		} else if (toolDigest !== this.#toolDigest) {
			report = {
				stable: false,
				divergedAt: 0,
				previousLength,
				rewritten: true,
			};
		} else if (chatMessages.length < previousLength) {
			report = {
				stable: false,
				divergedAt: 0,
				previousLength,
				rewritten: true,
			};
		} else {
			let divergedAt = -1;
			const bound = Math.min(previousLength, digests.length);
			for (let i = 0; i < bound; i++) {
				if (digests[i] !== this.#messageDigests[i]) {
					divergedAt = i;
					break;
				}
			}
			report = {
				stable: divergedAt === -1,
				divergedAt: divergedAt === -1 ? previousLength : divergedAt,
				previousLength,
				rewritten: false,
			};
		}

		this.#messageDigests = digests;
		this.#toolDigest = toolDigest;
		this.#recorded = true;
		return report;
	}

	/** Drop all state (run reset / model switch). */
	reset(): void {
		this.#messageDigests = new Uint32Array(0);
		this.#toolDigest = 0;
		this.#recorded = false;
	}
}
