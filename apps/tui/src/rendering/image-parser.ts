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

// Regex for markdown images: ![alt](src) or ![alt](src "title")
const MARKDOWN_IMAGE_PATTERN = /!\[([^\]]*)\]\((data:[^)]+|"([^"]*)")\)/g;
// Regex for data URIs: data:image/{type};base64,{base64}
const DATA_URI_PATTERN = /^data:image\/(\w+);base64,(.+)$/;

/**
 * Parse inline images from markdown text. Returns an array of image nodes
 * that replace their original positions in the text, plus the text with
 * image placeholders removed.
 */
export function parseInlineImages(text: string): { images: ParsedImage[]; text: string } {
	const images: ParsedImage[] = [];
	const segments: Array<string | ParsedImage> = [];

	// First pass: find all images
	let lastIndex = 0;
	let match: RegExpExecArray | null;
	// Reset regex state
	MARKDOWN_IMAGE_PATTERN.lastIndex = 0;

	while ((match = MARKDOWN_IMAGE_PATTERN.exec(text)) !== null) {
		// Add text before this image
		if (match.index > lastIndex) {
			segments.push(text.slice(lastIndex, match.index));
		}

		// Parse the image source
		const src = match[2]; // Full src including quotes
		const alt = match[1];

		// Remove surrounding quotes if present
		const cleanSrc = src.startsWith('"') ? src.slice(1, -1) : src;

		const dataUriMatch = cleanSrc.match(DATA_URI_PATTERN);
		if (dataUriMatch) {
			const mimeType = `image/${dataUriMatch[1]}`;
			const base64Data = dataUriMatch[2];
			segments.push({
				alt,
				mimeType,
				data: base64Data,
				isDataUri: true,
			});
		} else {
			// External image reference — not supported in TUI
			segments.push(alt);
		}

		lastIndex = match.index + match[0].length;
	}

	// Add remaining text
	if (lastIndex < text.length) {
		segments.push(text.slice(lastIndex));
	}

	// Separate images from text
	for (const seg of segments) {
		if (typeof seg === "object" && "mimeType" in seg) {
			images.push(seg);
		}
	}

	const textOnly = segments.map(s => typeof s === "string" ? s : "").join("");

	return { images, text: textOnly };
}

/**
 * Parse inline images from a single line of markdown. Returns the first
 * image found (if any) and the line with the image reference removed.
 */
export function parseInlineImageFromLine(line: string): { image: ParsedImage | null; text: string } {
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
