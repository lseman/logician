import { RESET, visibleWidth } from "../../terminal/core.ts";

export function wrapText(text: string, maxLineLength: number): string[] {
	const width = Math.max(1, Math.floor(maxLineLength));
	const lines: string[] = [];
	for (const rawLine of text.split("\n")) {
		if (visibleWidth(rawLine) <= width) {
			lines.push(rawLine);
			continue;
		}
		const words = rawLine.split(/\s+/);
		let current = "";
		for (const word of words) {
			const chunks = hardWrapVisible(word, width);
			for (let index = 0; index < chunks.length; index++) {
				const chunk = chunks[index];
				if (index > 0) {
					if (current) lines.push(current);
					current = chunk;
					continue;
				}
				if (!current) {
					current = chunk;
				} else if (
					visibleWidth(current) + 1 + visibleWidth(chunk) <= width
				) {
					current += ` ${chunk}`;
				} else {
					lines.push(current);
					current = chunk;
				}
				if (visibleWidth(current) === width && index < chunks.length - 1) {
					lines.push(current);
					current = "";
				}
			}
		}
		if (current) lines.push(current);
	}
	return lines;
}

function hardWrapVisible(text: string, width: number): string[] {
	if (visibleWidth(text) <= width) return [text];
	const chunks: string[] = [];
	let chunk = "";
	let chunkWidth = 0;
	let activeCodes = "";
	let index = 0;
	while (index < text.length) {
		const character = text[index];
		if (character === "\x1b") {
			const start = index;
			index++;
			const next = text[index];
			if (next === "[") {
				index++;
				while (index < text.length) {
					const finalCode = text.charCodeAt(index);
					index++;
					if (finalCode >= 0x40 && finalCode <= 0x7e) break;
				}
			} else if (next === "]" || next === "_") {
				index++;
				while (index < text.length) {
					if (text[index] === "\x07") {
						index++;
						break;
					}
					if (text[index] === "\x1b" && text[index + 1] === "\\") {
						index += 2;
						break;
					}
					index++;
				}
			}
			const sequence = text.slice(start, index);
			chunk += sequence;
			if (/^\x1b\[0m$/.test(sequence)) {
				activeCodes = "";
			} else if (sequence.startsWith("\x1b[")) {
				activeCodes += sequence;
			}
			continue;
		}
		const characterWidth = visibleWidth(character);
		if (chunk && chunkWidth + characterWidth > width) {
			chunks.push(chunk + (activeCodes ? RESET : ""));
			chunk = activeCodes;
			chunkWidth = 0;
		}
		chunk += character;
		chunkWidth += characterWidth;
		index++;
	}
	if (chunk && activeCodes) {
		chunks.push(chunk + RESET);
	} else if (chunk) {
		chunks.push(chunk);
	}
	return chunks;
}
