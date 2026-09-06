// ── Terminal image capability detection ───────────────────────────────────────
// Detects which inline image protocol the terminal supports (Kitty, iTerm2,
// Sixel) and exposes a singleton `TERMINAL` for the image rendering pipeline.

import { encodeSixel } from "./image-encoding.ts";
import {
	getKittyGraphics,
	KITTY_PLACEHOLDER_MAX_CELLS,
	renderKittyPlaceholderLines,
	setKittyGraphics,
} from "./kitty-graphics.ts";

// ── Image protocol enumeration ────────────────────────────────────────────────

export enum ImageProtocol {
	Kitty = "\x1b_G",
	Iterm2 = "\x1b]1337;File=",
	Sixel = "\x1bPq",
}

/**
 * Terminal capability details used for rendering.
 * Mirrors pi's TerminalInfo but scoped to image support.
 */
export class TerminalInfo {
	constructor(
		public readonly id: string,
		public readonly imageProtocol: ImageProtocol | null,
		public readonly trueColor: boolean,
	) {}

	isImageLine(line: string): boolean {
		if (!this.imageProtocol) return false;
		if (this.imageProtocol === ImageProtocol.Sixel) {
			return hasSixelDcsStart(line);
		}
		return hasNeedleBefore(line, this.imageProtocol, 512)
			|| hasNeedleBefore(line, "\u{10eeee}", 512);
	}
}

function hasNeedleBefore(line: string, needle: string, limit: number): boolean {
	const index = line.indexOf(needle);
	return index !== -1 && index + needle.length <= limit;
}

function hasSixelDcsStart(line: string): boolean {
	const limit = Math.min(line.length, 128);
	let from = 0;
	for (;;) {
		const start = line.indexOf("\x1bP", from);
		if (start === -1 || start + 3 > limit) return false;
		let i = start + 2;
		while (i < limit) {
			const code = line.charCodeAt(i);
			if ((code >= 0x30 && code <= 0x39) || code === 0x3b) {
				i++;
				continue;
			}
			break;
		}
		if (i < limit && line.charCodeAt(i) === 0x71) return true;
		from = start + 2;
	}
}

// ── Forced protocol overrides ─────────────────────────────────────────────────

function getForcedImageProtocol(): ImageProtocol | null | undefined {
	const raw = process.env.FORCE_IMAGE_PROTOCOL?.trim().toLowerCase();
	if (!raw) return undefined;
	if (raw === "kitty") return ImageProtocol.Kitty;
	if (raw === "iterm2" || raw === "iterm") return ImageProtocol.Iterm2;
	if (raw === "sixel") return ImageProtocol.Sixel;
	if (raw === "off" || raw === "none" || raw === "0" || raw === "false")
		return null;
	return null;
}

// ── Cell dimensions (used for image fitting) ──────────────────────────────────

interface CellDimensions {
	widthPx: number;
	heightPx: number;
}

let cellDimensions: CellDimensions = { widthPx: 8, heightPx: 16 };

export function getCellDimensions(): CellDimensions {
	return cellDimensions;
}

export function setCellDimensions(dims: CellDimensions): void {
	cellDimensions = dims;
}

// ── Image fitting ─────────────────────────────────────────────────────────────

export interface ImageRenderOptions {
	maxWidthCells?: number;
	maxHeightCells?: number;
	preserveAspectRatio?: boolean;
}

export interface ImageDimensions {
	widthPx: number;
	heightPx: number;
}

function calculateImageFit(
	imageDimensions: ImageDimensions,
	options: ImageRenderOptions,
	cellDims: CellDimensions,
): { columns: number; rows: number } {
	let columns = imageDimensions.widthPx / cellDims.widthPx;
	let rows = imageDimensions.heightPx / cellDims.heightPx;

	if (options.maxWidthCells && options.maxWidthCells > 0) {
		columns = Math.min(columns, options.maxWidthCells);
	}
	if (options.maxHeightCells && options.maxHeightCells > 0) {
		rows = Math.min(rows, options.maxHeightCells);
	}

	if (options.preserveAspectRatio !== false) {
		const imageAspect = imageDimensions.widthPx / imageDimensions.heightPx;
		const cellAspect = cellDims.widthPx / cellDims.heightPx;
		if (columns / rows > imageAspect * cellAspect) {
			rows = columns / (imageAspect * cellAspect);
		} else {
			columns = rows * imageAspect * cellAspect;
		}
	}

	return {
		columns: Math.max(1, Math.floor(columns)),
		rows: Math.max(1, Math.floor(rows)),
	};
}

// ── Terminal singleton ────────────────────────────────────────────────────────

let TERMINAL: TerminalInfo = new TerminalInfo("base", null, false);

export function getTERMINAL(): TerminalInfo {
	return TERMINAL;
}

export function setTERMINAL(info: TerminalInfo): void {
	TERMINAL = info;
}

// ── Detection logic ───────────────────────────────────────────────────────────

export function detectImageProtocol(env: NodeJS.ProcessEnv = Bun.env): TerminalInfo {
	const forced = getForcedImageProtocol();
	if (forced !== undefined) {
		return new TerminalInfo("forced", forced, true);
	}

	const termId = env.TERM ?? "";

	// Detect terminal type
	let id = "base";
	if (env.KITTY_WINDOW_ID) id = "kitty";
	else if (env.WEZTERM_PANE) id = "wezterm";
	else if (env.TERM_PROGRAM === "iTerm.app") id = "iterm2";
	else if (env.TERM_PROGRAM === "Ghostty") id = "ghostty";
	else if (env.TERM_PROGRAM === "vscode") id = "vscode";
	else if (env.TERM_PROGRAM === "WezTerm") id = "wezterm";
	else if (env.TERM_PROGRAM === "Warp") id = "warp";

	// Detect image protocol support
	let imageProtocol: ImageProtocol | null = null;

	if (id === "kitty" || id === "ghostty") {
		imageProtocol = ImageProtocol.Kitty;
		const hasPlaceholders = detectKittyPlaceholders(id, env);
		setKittyGraphics({ unicodePlaceholders: hasPlaceholders });
	} else if (id === "iterm2") {
		imageProtocol = ImageProtocol.Iterm2;
	} else if (termId.includes("sixel") || env.TERM === "sixel") {
		imageProtocol = ImageProtocol.Sixel;
	}

	// Check for true color
	const trueColor =
		"COLORTERM" in env
			? env.COLORTERM === "truecolor" || env.COLORTERM === "24bit"
			: termId.includes("256") || termId.includes("true");

	return new TerminalInfo(id, imageProtocol, trueColor);
}

function detectKittyPlaceholders(terminalId: string, env: NodeJS.ProcessEnv): boolean {
	if (env.PI_NO_KITTY_PLACEHOLDERS === "1") return false;
	if (env.PI_KITTY_PLACEHOLDERS === "1") return true;
	if (env.PI_KITTY_PLACEHOLDERS === "0") return false;
	if (terminalId === "kitty" || terminalId === "ghostty") return true;
	if (terminalId === "wezterm" && env.PI_FORCE_IMAGE_PROTOCOL === "kitty") return true;
	if (env.TMUX && env.PI_FORCE_IMAGE_PROTOCOL === "kitty") return true;
	return false;
}

// ── Image render dispatch ─────────────────────────────────────────────────────

export function renderImage(
	base64Data: string,
	imageDimensions: ImageDimensions,
	options: ImageRenderOptions & { imageId?: number; placementId?: number; includeTransmit?: boolean } = {},
): { sequence?: string; lines?: string[]; rows: number; transmit?: string } | null {
	if (!TERMINAL.imageProtocol) return null;

	const cellDims = getCellDimensions();
	const fit = calculateImageFit(imageDimensions, options, cellDims);

	if (TERMINAL.imageProtocol === ImageProtocol.Kitty) {
		const graphics = getKittyGraphics();

		let transmit: string | undefined;
		if (options.includeTransmit && options.imageId != null) {
			transmit = `\x1b_Ga=t,f=100,q=2,i=${options.imageId},${base64Data}\x1b\\`;
		}

		// Unicode placeholders render as real text cells.
		if (graphics.unicodePlaceholders && fit.columns <= KITTY_PLACEHOLDER_MAX_CELLS && fit.rows <= KITTY_PLACEHOLDER_MAX_CELLS) {
			const lines = renderKittyPlaceholderLines({
				imageId: options.imageId ?? 0,
				placementId: options.placementId ?? options.imageId,
				columns: fit.columns,
				rows: fit.rows,
			});
			return { lines, rows: fit.rows, transmit };
		}

		// Direct placement.
		const imageId = options.imageId ?? 0;
		const placementId = options.placementId ?? imageId;
		const params: string[] = ["a=p", "q=2", "C=1", `i=${imageId}`];
		if (placementId) params.push(`p=${placementId}`);
		params.push(`c=${fit.columns}`, `r=${fit.rows}`);
		const sequence = `\x1b_G${params.join(",")}\x1b\\`;
		return { sequence, rows: fit.rows, transmit };
	}

	if (TERMINAL.imageProtocol === ImageProtocol.Sixel) {
		const rawHeightPx = Math.max(1, fit.rows * cellDims.heightPx);
		const targetHeightPx = Math.max(6, Math.floor(rawHeightPx / 6) * 6);
		const heightScale = targetHeightPx / rawHeightPx;
		const targetWidthPx = Math.max(1, Math.round(fit.columns * cellDims.widthPx * heightScale));
		const rows = Math.max(1, Math.ceil(targetHeightPx / cellDims.heightPx));
		try {
			const decoded = new Uint8Array(Buffer.from(base64Data, "base64"));
			const sequence = encodeSixel(decoded, targetWidthPx, targetHeightPx);
			return { sequence, rows };
		} catch {
			return null;
		}
	}

	if (TERMINAL.imageProtocol === ImageProtocol.Iterm2) {
		let params = "filename=;create=";
		if (options.preserveAspectRatio !== false) params += ":aspectRatio:";
		params += `;width=${fit.columns}`;
		params += ";height=auto";
		const sequence = `\x1b]1337;${params}\x07${base64Data}\x07`;
		return { sequence, rows: fit.rows };
	}

	return null;
}

export function imageFallback(
	mimeType: string,
	dimensions?: ImageDimensions,
	filename?: string,
): string {
	const parts: string[] = [];
	if (filename) parts.push(filename);
	parts.push(`[${mimeType}]`);
	if (dimensions) parts.push(`${dimensions.widthPx}x${dimensions.heightPx}`);
	return `[Image: ${parts.join(" ")}]`;
}
