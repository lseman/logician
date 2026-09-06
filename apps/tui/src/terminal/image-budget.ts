// ── ImageBudget ───────────────────────────────────────────────────────────────
// Bounds how many inline images render as live terminal graphics at once.
// Older images fall back to text placeholders when the cap is exceeded.

const EMPTY_IDS: readonly number[] = [];
const EMPTY_TRANSMITS: readonly string[] = [];

export const DEFAULT_MAX_INLINE_IMAGES = 8;

interface PlacementState {
	widthPx: number;
	heightPx: number;
	epoch: number;
	lastAttachTopFrameRow: number | undefined;
	cellsArchived: boolean;
}

/**
 * Bounds how many inline images render as live terminal graphics at once.
 * Displays at most `cap` images; older ones (in display order) fall back to text.
 * Handles placement geometry, transmit tracking, and epoch management.
 */
export class ImageBudget {
	private _cap: number;
	private _requestRender: () => void;
	private _nextId = Math.floor(Math.random() * 0xffffff) + 1;
	private _keyToId = new Map<string, number>();
	private _idToKey = new Map<number, string>();
	private _passIds: number[] = [];
	private _passSuppression = new Map<number, boolean>();
	private _onTerminal = 0;
	private _planned = 0;
	private _applyingReset = false;
	private _lastTotal = 0;
	private _purgeIds: number[] = [];
	private _transmitted = new Set<number>();
	private _pendingTransmits = new Map<number, string>();
	private _stablePass = false;
	private _suppressedIds = new Set<number>();
	private _placementState = new Map<number, PlacementState>();

	constructor(
		cap: number = DEFAULT_MAX_INLINE_IMAGES,
		requestRender: () => void = () => {},
	) {
		this._cap = normalizeCap(cap);
		this._requestRender = requestRender;
	}

	get cap(): number {
		return this._cap;
	}

	get enabled(): boolean {
		return this._cap > 0;
	}

	setRequestRender(requestRender: () => void): void {
		this._requestRender = requestRender;
	}

	setCap(cap: number): void {
		const next = normalizeCap(cap);
		if (next === this._cap) return;
		this._cap = next;
		this._reconcile(this._lastTotal);
	}

	/**
	 * Stable graphics id for a logical image. A non-empty `key` maps to the
	 * same id across re-creations; a missing key gets a fresh id every call.
	 */
	acquireId(key?: string): number {
		if (key) {
			const existing = this._keyToId.get(key);
			if (existing !== undefined) return existing;
		}
		const id = this._nextId;
		this._nextId = (this._nextId + 1) & 0xffffff || 1;
		if (key) {
			this._keyToId.set(key, id);
			this._idToKey.set(id, key);
		}
		return id;
	}

	/**
	 * Begin a render pass. Call `stable: true` for partial/throwaway passes
	 * that do not walk the whole tree in display order.
	 */
	beginPass(stable = false): void {
		this._passIds.length = 0;
		this._passSuppression.clear();
		this._stablePass = stable;
		this._applyingReset =
			!stable && this._cap > 0 && this._planned > this._onTerminal;
	}

	/**
	 * Record an image in display order and report whether it must render its
	 * text fallback this frame. Called by every Image during render — including
	 * on a cache hit, so the image keeps its display-order slot.
	 */
	observe(imageId: number): boolean {
		const existing = this._passSuppression.get(imageId);
		if (existing !== undefined) return existing;
		if (this._stablePass) {
			const suppressed = this._cap > 0 && this._suppressedIds.has(imageId);
			this._passSuppression.set(imageId, suppressed);
			return suppressed;
		}
		const index = this._passIds.length;
		this._passIds.push(imageId);
		const suppressed = this._cap > 0 && index < this._planned;
		this._passSuppression.set(imageId, suppressed);
		return suppressed;
	}

	/**
	 * End a render pass. Returns true when the pass discovered a stricter budget
	 * and must be repeated before its terminal frame is emitted.
	 */
	endPass(): boolean {
		const total = this._passIds.length;
		this._lastTotal = total;
		if (this._applyingReset) {
			for (let i = this._onTerminal; i < this._planned && i < total; i++) {
				const id = this._passIds[i];
				if (!this._pendingTransmits.delete(id)) this._purgeIds.push(id);
				this._transmitted.delete(id);
				this.#forgetKeyForId(id);
			}
			this._onTerminal = this._planned;
			this._applyingReset = false;
		}
		const retry = this._reconcile(total);
		this._suppressedIds = new Set(this._passIds.slice(0, this._onTerminal));
		return retry;
	}

	/** Image ids to delete from the terminal this frame; clears the pending set. */
	takePurgeIds(): readonly number[] {
		if (this._purgeIds.length === 0) return EMPTY_IDS;
		const ids = this._purgeIds;
		this._purgeIds = [];
		return ids;
	}

	/** All image ids believed to be loaded in the terminal store; clears tracking. */
	takeAllTransmittedIds(): readonly number[] {
		if (this._transmitted.size === 0) return EMPTY_IDS;
		const ids = [...this._transmitted];
		this._transmitted.clear();
		this._purgeIds = [];
		this._pendingTransmits.clear();
		this._keyToId.clear();
		this._idToKey.clear();
		return ids;
	}

	/** Whether `imageId`'s data still needs to be transmitted to the terminal. */
	shouldTransmit(imageId: number): boolean {
		return !this._transmitted.has(imageId);
	}

	/** Record a direct-placement image's source pixel geometry. */
	registerPlacementGeometry(
		imageId: number,
		widthPx: number,
		heightPx: number,
	): void {
		this._placementState.set(imageId, {
			widthPx,
			heightPx,
			epoch: 1,
			lastAttachTopFrameRow: undefined,
			cellsArchived: false,
		});
	}

	/**
	 * Resolve the placement id and geometry for a direct-placement emit.
	 * Advances the epoch when the placement's cells have entered scrollback.
	 */
	resolvePlacementEmit(
		imageId: number,
		_attachTopFrameRow: number,
		_committedTo: number,
	): { placementId: number; widthPx: number; heightPx: number } | null {
		const state = this._placementState.get(imageId);
		if (!state) return null;

		// Check if this placement has been archived (committed to scrollback)

		if (state.cellsArchived) {
			state.epoch += 1;
			state.cellsArchived = false;
		}

		return {
			placementId: state.epoch,
			widthPx: state.widthPx,
			heightPx: state.heightPx,
		};
	}

	/** Restart every placement epoch after a destructive history clear. */
	resetPlacementEpochs(): ReadonlyArray<{
		imageId: number;
		lastEpoch: number;
	}> {
		const stale: Array<{ imageId: number; lastEpoch: number }> = [];
		for (const [imageId, state] of this._placementState) {
			stale.push({ imageId, lastEpoch: state.epoch });
			state.epoch = 1;
			state.cellsArchived = false;
		}
		return stale;
	}

	/** Queue a one-time transmit for `imageId`. No-op if already transmitted. */
	enqueueTransmit(imageId: number, sequence: string): void {
		if (this._transmitted.has(imageId)) return;
		this._transmitted.add(imageId);
		this._pendingTransmits.set(imageId, sequence);
	}

	/** Whether a frame has image data queued but not yet written to the terminal. */
	hasPendingTransmits(): boolean {
		return this._pendingTransmits.size > 0;
	}

	/**
	 * True when the budget has nothing in flight: no live images observed,
	 * no queued transmits, no pending purges, and no stricter threshold left.
	 */
	get quiescent(): boolean {
		return (
			this._lastTotal === 0 &&
			this._pendingTransmits.size === 0 &&
			this._purgeIds.length === 0 &&
			this._planned === this._onTerminal
		);
	}

	/** Transmit sequences to write before this frame's placements; clears the queue. */
	takeTransmits(): readonly string[] {
		if (this._pendingTransmits.size === 0) return EMPTY_TRANSMITS;
		const sequences = [...this._pendingTransmits.values()];
		this._pendingTransmits.clear();
		return sequences;
	}

	/** Drop transmit tracking so every still-live image re-enqueues on next render. */
	forgetTransmitted(): void {
		if (this._transmitted.size === 0 && this._pendingTransmits.size === 0)
			return;
		this._transmitted.clear();
		this._pendingTransmits.clear();
	}

	#forgetKeyForId(id: number): void {
		const key = this._idToKey.get(id);
		if (key === undefined) return;
		this._idToKey.delete(id);
		if (this._keyToId.get(key) === id) this._keyToId.delete(key);
	}

	_reconcile(total: number): boolean {
		const desired = this._cap > 0 ? Math.max(0, total - this._cap) : 0;
		if (desired === this._planned) {
			if (this._planned < this._onTerminal) this._onTerminal = this._planned;
			return false;
		}
		const retry = desired > this._onTerminal;
		this._planned = desired;
		if (desired <= this._onTerminal) this._onTerminal = desired;
		this._requestRender();
		return retry;
	}
}

function normalizeCap(cap: number): number {
	if (!Number.isFinite(cap)) return 0;
	return Math.max(0, Math.trunc(cap));
}
