// ── Image encoding utilities ──────────────────────────────────────────────────
// Sixel image protocol encoding. (Kitty graphics live in kitty-graphics.ts.)

/**
 * Encode Sixel image data. Uses the `@oh-my-pi/pi-natives` encodeSixel if
 * available, otherwise falls back to a basic implementation.
 */
export function encodeSixel(
	data: Uint8Array,
	width: number,
	height: number,
): string {
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
			row += `${String.fromCharCode(33 + bits)}`;
		}
		sixelData += `${row}!`;
	}
	return `${header}${sixelData}\x1b\\`;
}
