// ── Image dimension parsing ───────────────────────────────────────────────────
// Extract pixel dimensions from base64-encoded PNG, JPEG, and WebP images.

export interface ImageDimensions {
	widthPx: number;
	heightPx: number;
}

/** Parse PNG IHDR chunk for width and height. */
export function getPngDimensions(base64Data: string): ImageDimensions | null {
	try {
		const buffer = Buffer.from(base64Data, "base64");
		// PNG signature: 8 bytes, then IHDR chunk starts at offset 8
		// IHDR: 4 bytes length + 4 bytes type ("IHDR") + 4 bytes width + 4 bytes height
		if (buffer.length < 24) return null;
		const width = buffer.readUInt32BE(16);
		const height = buffer.readUInt32BE(20);
		if (width === 0 || height === 0) return null;
		return { widthPx: width, heightPx: height };
	} catch {
		return null;
	}
}

/** Parse JPEG SOF0/SOF2 markers for dimensions. */
export function getJpegDimensions(base64Data: string): ImageDimensions | null {
	try {
		const buffer = Buffer.from(base64Data, "base64");
		// JPEG starts with SOI (0xFFD8), then APPn markers, then SOF
		let offset = 2;
		while (offset < buffer.length - 1) {
			if (buffer[offset] !== 0xff) break;
			offset++;
			const marker = buffer[offset];
			offset++;
			// SOF markers: C0-SOF0, C1-SOF1, C2-SOF2 (baseline, progressive, DHR)
			if ((marker >= 0xc0 && marker <= 0xc3) || marker === 0xc9 || marker === 0xca) {
				if (offset + 7 > buffer.length) break;
				// Segment length is 2 bytes at current offset
				const segLen = buffer.readUInt16BE(offset);
				if (segLen < 7) break;
				const height = buffer.readUInt16BE(offset + 3);
				const width = buffer.readUInt16BE(offset + 5);
				if (width === 0 || height === 0) return null;
				return { widthPx: width, heightPx: height };
			}
			// Skip segments: length is 2 bytes after the marker
			const segLen = buffer.readUInt16BE(offset);
			if (segLen < 2 || offset + segLen > buffer.length) break;
			offset += segLen;
		}
		return null;
	} catch {
		return null;
	}
}

/** Parse WebP VP8/VP8L/VP8X headers for dimensions. */
export function getWebpDimensions(base64Data: string): ImageDimensions | null {
	try {
		const buffer = Buffer.from(base64Data, "base64");
		// WebP signature: "RIFF" + size + "WEBP" (12 bytes)
		if (buffer.length < 16) return null;
		const riff = buffer.toString("ascii", 0, 4);
		const webp = buffer.toString("ascii", 8, 12);
		if (riff !== "RIFF" || webp !== "WEBP") return null;

		const chunkType = buffer.toString("ascii", 12, 16);
		if (chunkType === "VP8 ") {
			// Lossy VP8: dimensions at offset 26 (little-endian 16-bit)
			if (buffer.length < 30) return null;
			const width = buffer.readUInt16LE(26);
			const height = buffer.readUInt16LE(28);
			// VP8 stores width/height as 16-bit + lower 2 bits in bit 14-15 of next word
			if (width === 0 || height === 0) return null;
			return { widthPx: width, heightPx: height };
		}
		if (chunkType === "VP8L") {
			// Lossless VP8L: 4-byte signature then 5-byte header
			if (buffer.length < 25) return null;
			const b0 = buffer[20];
			const b1 = buffer[21];
			const b2 = buffer[22];
			const b3 = buffer[23];
			const b4 = buffer[24];
			const width = ((b1 & 0x3f) << 8) | b0;
			const height = ((b3 & 0x3f) << 8) | b2;
			if (width === 0 || height === 0) return null;
			return { widthPx: width + 1, heightPx: height + 1 };
		}
		if (chunkType === "VP8X") {
			// Extended WebP with VP8X header
			if (buffer.length < 30) return null;
			const flags = buffer[26];
			const hasAlpha = (flags >> 3) & 1;
			const hasMeta = (flags >> 4) & 1;
			// Canvas size: 3 bytes each for width and height (24-bit)
			const width =
				(buffer[19] << 16) | (buffer[18] << 8) | buffer[17];
			const height =
				(buffer[22] << 16) | (buffer[21] << 8) | buffer[20];
			if (width === 0 || height === 0) return null;
			return { widthPx: width + 1, heightPx: height + 1 };
		}
		return null;
	} catch {
		return null;
	}
}

/** Parse image dimensions from base64 data based on MIME type. */
export function getImageDimensions(
	base64Data: string,
	mimeType: string,
): ImageDimensions | null {
	if (mimeType === "image/png") return getPngDimensions(base64Data);
	if (mimeType === "image/jpeg" || mimeType === "image/jpg")
		return getJpegDimensions(base64Data);
	if (mimeType === "image/webp") return getWebpDimensions(base64Data);
	return null;
}
