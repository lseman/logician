// ── Image component ───────────────────────────────────────────────────────────
// Renders inline images using the detected terminal image protocol, or falls
// back to a styled text placeholder when the protocol is unavailable or
// the image has been suppressed by the budget.

import type { Component } from "../terminal/primitives.ts";
import { theme } from "../terminal/theme.ts";
import {
	getTERMINAL,
	imageFallback,
	renderImage,
	type ImageDimensions,
	type ImageRenderOptions,
} from "../terminal/image-protocol.ts";
import type { ImageBudget } from "../terminal/image-budget.ts";

const SAVE_CURSOR = "\x1b7";
const RESTORE_CURSOR = "\x1b8";
// Direct placements reserve height with leading zero-width rows. Keep them
// non-plain so transcript blank-edge trimming does not collapse image-only blocks.
const RESERVED_IMAGE_ROW = "\x1b[0m";

export interface ImageOptions extends ImageRenderOptions {
	/** Shared budget that caps how many inline images render as live graphics. */
	budget?: ImageBudget;
	/** Stable identity for the underlying image. Lets the budget hand back the
	 * same graphics id across component re-creations so a repaint replaces the
	 * placement instead of stacking a duplicate. */
	imageKey?: string;
}

/**
 * Renders a single inline image. The component owns its base64 data and
 * delegates to the terminal's image protocol for rendering. When the budget
 * suppresses an image (too many concurrent images), it falls back to a
 * text placeholder.
 */
export class ImageComponent implements Component {
	#base64Data: string;
	#mimeType: string;
	#dimensions: ImageDimensions;
	#options: ImageOptions;
	#budget?: ImageBudget;
	#imageId?: number;
	#cachedLines?: string[];
	#cachedWidth?: number;
	#cachedSuppressed = false;
	#cachedImageProtocol: string | null = null;
	#cachedCellWidthPx = 0;
	#cachedCellHeightPx = 0;
	#renderedGraphicRows = 0;

	constructor(
		base64Data: string,
		mimeType: string,
		options: ImageOptions = {},
		dimensions?: ImageDimensions,
	) {
		this.#base64Data = base64Data;
		this.#mimeType = mimeType;
		this.#options = options;
		this.#dimensions = dimensions || { widthPx: 800, heightPx: 600 };
		this.#budget = options.budget;
		this.#imageId = options.budget ? options.budget.acquireId(options.imageKey) : undefined;
	}

	invalidate(): void {
		this.#cachedLines = undefined;
		this.#cachedWidth = undefined;
	}

	render(width: number): string[] {
		const terminal = getTERMINAL();
		const imageProtocol = terminal.imageProtocol;
		const imageProtocolStr = imageProtocol === null ? "none" : String(imageProtocol);
		const hasProtocol = imageProtocol != null;
		const cellDimensions = { widthPx: 8, heightPx: 16 }; // Will be replaced when proper cell dims are available

		// observe() must run on every pass so the image keeps its display-order slot.
		const suppressed = hasProtocol && this.#budget !== undefined
			? this.#budget.observe(this.#imageId ?? 0)
			: false;

		if (
			this.#cachedLines &&
			this.#cachedWidth === width &&
			this.#cachedSuppressed === suppressed &&
			this.#cachedImageProtocol === imageProtocolStr &&
			this.#cachedCellWidthPx === cellDimensions.widthPx &&
			this.#cachedCellHeightPx === cellDimensions.heightPx
		) {
			return this.#cachedLines;
		}

		const cap = this.#options.maxWidthCells;
		const maxWidth = cap != null && cap > 0 ? Math.min(width - 2, cap) : width - 2;

		let lines: string[];

		if (hasProtocol && !suppressed) {
			const needsTransmit = this.#imageId != null && (this.#budget?.shouldTransmit(this.#imageId) ?? false);
			const result = renderImage(this.#base64Data, this.#dimensions, {
				maxWidthCells: maxWidth,
				maxHeightCells: this.#options.maxHeightCells,
				imageId: this.#imageId,
				includeTransmit: needsTransmit,
				preserveAspectRatio: true,
			});

			if (result?.transmit && this.#imageId != null && this.#budget !== undefined) {
				this.#budget.enqueueTransmit(this.#imageId, result.transmit);
			}

			if (result?.lines) {
				// Unicode placeholders: the image is already a block of real text cells.
				lines = result.lines;
			} else if (result?.sequence) {
				// Direct placement: return `rows` lines so TUI accounts for image height.
				// First (rows-1) lines are reserved (non-plain so blank-edge trimming
				// doesn't collapse them); the last line has the placement sequence.
				if (this.#imageId != null && this.#budget !== undefined) {
					this.#budget.registerPlacementGeometry(
						this.#imageId,
						this.#dimensions.widthPx,
						this.#dimensions.heightPx,
					);
				}
				lines = [];
				for (let i = 0; i < result.rows - 1; i++) {
					lines.push(RESERVED_IMAGE_ROW);
				}
				const cursorRows = result.rows - 1;
				const moveUp = cursorRows > 0 ? `\x1b[${cursorRows}A` : "";
				lines.push(cursorRows > 0 ? SAVE_CURSOR + moveUp + result.sequence + RESTORE_CURSOR : result.sequence);
			} else {
				lines = this.#fallbackLines();
			}
			this.#renderedGraphicRows = Math.max(this.#renderedGraphicRows, lines.length);
		} else {
			lines = this.#fallbackLines();
		}

		this.#cachedLines = lines;
		this.#cachedWidth = width;
		this.#cachedSuppressed = suppressed;
		this.#cachedImageProtocol = imageProtocolStr;
		this.#cachedCellWidthPx = cellDimensions.widthPx;
		this.#cachedCellHeightPx = cellDimensions.heightPx;

		return lines;
	}

	/**
	 * Text fallback, height-preserving once a graphic has rendered: a demoted
	 * image must keep occupying the rows its placement used, because those rows
	 * may already be committed to native scrollback.
	 */
	#fallbackLines(): string[] {
		const fallback = theme.fg("muted", "");
		const text = imageFallback(this.#mimeType, this.#dimensions);
		const reset = "\x1b[0m";
		const styled = `${fallback}${text}${reset}`;
		if (this.#renderedGraphicRows <= 1) return [styled];
		const lines: string[] = [];
		for (let i = 0; i < this.#renderedGraphicRows - 1; i++) {
			lines.push(RESERVED_IMAGE_ROW);
		}
		lines.push(styled);
		return lines;
	}
}
