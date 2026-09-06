// ── Image encoding utilities ──────────────────────────────────────────────────
// Kitty graphics and iTerm2 image protocol encoding functions.

import { wrapTmuxPassthroughIfNeeded } from "./tmux";

/** Chunk a Kitty APC payload to stay under terminal write limits. */
function chunkKittyApc(params: string, base64Data: string): string {
	const CHUNK_SIZE = 4096;
	const parts: string[] = [];
	let pos = 0;
	while (pos < base64Data.length) {
		const chunk = base64Data.slice(pos, pos + CHUNK_SIZE);
		parts.push(`\x1b_G${params},f=100,q=2,a=t,${chunk}\x1b\\`);
		pos += CHUNK_SIZE;
	}
	return parts.join("");
}

/**
 * Transmit image data only (`a=t`), keyed by `imageId`, without displaying it.
 * Sent once per image; the data then persists in the terminal's store, so
 * subsequent frames display it with a tiny placement sequence instead of
 * re-sending the base64.
 */
export function encodeKittyTransmit(base64Data: string, imageId: number): string {
	return chunkKittyApc(`i=${imageId}`, base64Data);
}

/**
 * Display a previously transmitted image (`a=p`) at the cursor. `C=1` keeps
 * the terminal cursor anchored at the placement origin so the renderer's
 * explicit cursor movement remains the only row accounting. A stable
 * `placementId` means re-emitting the sequence replaces the existing placement
 * rather than stacking a duplicate.
 */
export function encodeKittyPlacement(options: {
	imageId: number;
	placementId?: number;
	columns: number;
	rows: number;
}): string {
	const params: string[] = ["a=p", "q=2", "C=1", `i=${options.imageId}`];
	if (options.placementId) params.push(`p=${options.placementId}`);
	params.push(`c=${options.columns}`, `r=${options.rows}`);
	return wrapTmuxPassthroughIfNeeded(`\x1b_G${params.join(",")}\x1b\\`);
}

/**
 * Exact shape of the direct-placement line: optional `ESC 7` + `CUU(rows-1)`
 * prefix, the placement APC, optional `ESC 8` suffix.
 */
const KITTY_DIRECT_PLACEMENT_LINE =
	/^(?:\x1b7(?:\x1b\[(\d+)A)?)?\x1b_Ga=p,q=2,C=1,i=(\d+)(?:,p=(\d+))?(?:,c=(\d+))?(?:,r=(\d+))?\x1b\\(?:\x1b8)?$/;

export interface ParsedKittyPlacementLine {
	imageId: number;
	placementId: number | undefined;
	columns: number;
	rows: number;
}

/** Parse a frame line that consists solely of a Kitty direct placement. */
export function parseKittyDirectPlacementLine(
	line: string,
): ParsedKittyPlacementLine | null {
	const m = KITTY_DIRECT_PLACEMENT_LINE.exec(line);
	if (!m) return null;
	const columns = m[4] !== undefined ? Number(m[4]) : 0;
	const rows = m[5] !== undefined ? Number(m[5]) : 0;
	if (columns <= 0 || rows <= 0) return null;
	return {
		imageId: Number(m[2]),
		placementId: m[3] !== undefined ? Number(m[3]) : undefined,
		columns,
		rows,
	};
}

/**
 * Rebuild an Image direct-placement line for the viewport row it is written at.
 * Clamps the source rectangle to the visible bottom slice when the block
 * straddles the viewport top.
 */
export function encodeKittyPlacementLine(options: {
	imageId: number;
	placementId: number;
	columns: number;
	rows: number;
	screenRow: number;
	imageHeightPx: number;
}): string {
	const clippable = options.imageHeightPx > 0;
	const hiddenRows = clippable ? Math.max(0, options.rows - 1 - options.screenRow) : 0;
	const visibleRows = options.rows - hiddenRows;
	const params: string[] = ["a=p", "q=2", "C=1", `i=${options.imageId}`, `p=${options.placementId}`];
	params.push(`c=${options.columns}`, `r=${visibleRows}`);
	if (hiddenRows > 0) {
		const srcY = Math.floor((options.imageHeightPx * hiddenRows) / options.rows);
		params.push(`y=${srcY}`, `h=${Math.max(1, options.imageHeightPx - srcY)}`);
	}
	const apc = `\x1b_G${params.join(",")}\x1b\\`;
	const cuu = visibleRows - 1;
	return cuu > 0 ? `\x1b7\x1b[${cuu}A${apc}\x1b8` : apc;
}

/**
 * Kitty graphics delete command for a single image id. Uses `d=I` (capital)
 * which removes the image and every placement — on screen and in scrollback —
 * and frees the backing data.
 */
export function encodeKittyDeleteImage(imageId: number): string {
	return wrapTmuxPassthroughIfNeeded(`\x1b_Ga=d,d=I,i=${imageId},q=2\x1b\\`);
}

/** Delete every Kitty image and placement in the terminal. */
export function encodeKittyDeleteAllImages(): string {
	return wrapTmuxPassthroughIfNeeded("\x1b_Ga=d,d=A,q=2\x1b\\");
}

/**
 * Delete a single placement of an image (`d=i`, lowercase): removes its cells
 * and registry entry but keeps the transmitted data.
 */
export function encodeKittyDeletePlacement(imageId: number, placementId: number): string {
	return wrapTmuxPassthroughIfNeeded(
		`\x1b_Ga=d,d=i,i=${imageId},p=${placementId},q=2\x1b\\`,
	);
}

/**
 * Transmit-and-display (`a=T`) — the self-contained form used when no stable
 * id is available.
 */
export function encodeKitty(
	base64Data: string,
	options: { columns?: number; rows?: number } = {},
): string {
	const params: string[] = ["a=T", "f=100", "q=2", "C=1"];
	if (options.columns) params.push(`c=${options.columns}`);
	if (options.rows) params.push(`r=${options.rows}`);
	const parts: string[] = [];
	let pos = 0;
	const CHUNK_SIZE = 4096;
	while (pos < base64Data.length) {
		const chunk = base64Data.slice(pos, pos + CHUNK_SIZE);
		const isLast = pos + CHUNK_SIZE >= base64Data.length;
		const action = isLast ? "T" : "t";
		parts.push(
			`\x1b_G${params.join(",")},a=${action},${chunk}\x1b\\`,
		);
		pos += CHUNK_SIZE;
	}
	return parts.join("");
}

/**
 * Self-contained transmit-and-display for iTerm2 image protocol.
 */
export function encodeITerm2(
	base64Data: string,
	options: { width?: number; height?: string; name?: string; preserveAspectRatio?: boolean } = {},
): string {
	let params = "filename=";
	if (options.name) params += options.name;
	params += ";create=";
	if (options.preserveAspectRatio ?? true) params += "=:aspectRatio:";
	params += ";width=" + (options.width ?? "");
	params += ";height=" + (options.height ?? "auto");
	return `\x1b]1337;${params}\x07${base64Data}\x07`;
}

/**
 * Encode Sixel image data. Uses the `@oh-my-pi/pi-natives` encodeSixel if
 * available, otherwise falls back to a basic implementation.
 */
export function encodeSixel(data: Uint8Array, width: number, height: number): string {
	// Basic Sixel encoding: resize + encode
	// A proper implementation would use pi-natives' encodeSixel for accuracy.
	// This is a simplified fallback.
	const header = `\x1bPq${width}v0${height}r0`;
	const pixels = Array.from(data)
		.map(b => b.toString(2).padStart(8, "0"))
		.join("");
	let sixelData = "";
	for (let y = 0; y < height; y++) {
		let row = "";
		for (let x = 0; x < width; x++) {
			let bits = 0;
			for (let p = 0; p < 7; p++) {
				const bitOffset = y * width * 7 + x * 7 + p;
				if (bitOffset < pixels.length && pixels[bitOffset] === "1") {
					bits |= 1 << p;
				}
			}
			row += String.fromCharCode(33 + bits);
		}
		sixelData += row + "!";
	}
	return header + sixelData + "\x1b\\";
}
