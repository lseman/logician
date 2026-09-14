// ── Markdown inline image parser ──────────────────────────────────────────────
// Extracts inline images from markdown content. Handles data URIs and file
// references. Returns parsed image nodes for rendering.

export interface ParsedImage {
	/** Markdown alt text. */
	alt: string;
	/** MIME type (e.g. "image/png"). */
	mimeType: string;
	/** Base64-encoded image data, or file path for external images. */
	data: string;
	/** Whether data is a data URI (true) or a file path (false). */
	isDataUri: boolean;
}

// Regex for data URIs: data:image/{type};base64,{base64}
const DATA_URI_PATTERN = /^data:image\/(\w+);base64,(.+)$/;

/**
 * Parse inline images from a single line of markdown. Returns the first
 * image found (if any) and the line with the image reference removed.
 */
export function parseInlineImageFromLine(line: string): {
	image: ParsedImage | null;
	text: string;
} {
	const imageRegex = /!\[([^\]]*)\]\((data:[^)]+)\)/;
	const match = imageRegex.exec(line);
	if (!match) return { image: null, text: line };

	const src = match[2];
	const alt = match[1];
	const cleanSrc = src.startsWith('"') ? src.slice(1, -1) : src;

	const dataUriMatch = cleanSrc.match(DATA_URI_PATTERN);
	if (dataUriMatch) {
		const mimeType = `image/${dataUriMatch[1]}`;
		const base64Data = dataUriMatch[2];
		const image: ParsedImage = {
			alt,
			mimeType,
			data: base64Data,
			isDataUri: true,
		};
		// Replace the image reference with a placeholder for line rendering
		const text = line.replace(match[0], "");
		return { image, text };
	}

	return { image: null, text: line };
}
