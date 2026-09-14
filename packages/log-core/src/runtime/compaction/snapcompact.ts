// ── Snapcompact: local, deterministic context compaction ───────────────────────
//
// Instead of calling an LLM to summarize discarded history, snapcompact
// serializes the conversation into a structured text format, normalizes it
// (Unicode folding, whitespace collapsing, data URL elision), and renders
// the result as compact text frames.
//
// This replaces the LLM-based `generateCompactionSummary` path with a
// zero-latency, zero-cost local compaction.

import { SNAPCOMPACT_PRESERVE_KEY } from "../../system/types/types-messages.ts";

// ============================================================================
// Types
// ============================================================================

/** Message type compatible with logicin's CompactableMessage. */
export interface CompactableMessage {
	role: string;
	content?: unknown[] | string | null;
	usage?: Record<string, number>;
	entryId?: string;
	toolCallId?: string;
}

/** One rendered snapcompact frame: a base64 PNG plus its geometry. */
export interface Frame {
	/** Base64-encoded PNG payload (empty when canvas unavailable). */
	data: string;
	/** Characters per row in the frame grid. */
	cols: number;
	/** Text rows in the frame grid. */
	rows: number;
	/** Characters actually printed onto this frame. */
	chars: number;
	/** Shape metadata (font name). */
	font?: string;
}

/** Provider-aware frame sizing configuration. */
export interface FrameConfig {
	/** Vision model provider for frame-size tuning ("gpt4o", "claude", "gemini", etc.). */
	provider?: string;
	/** Override columns; auto-resolved via PROVIDER_COLS when provider is set. */
	cols?: number;
	/** Override rows. */
	rows?: number;
	/** Force PNG rendering: true to enable, false to disable. */
	render?: boolean;
	/** Maximum frames to carry in rebuilt requests. */
	maxFrames?: number;
}
/** Archive of rendered frames persisted between compactions. */
export interface Archive {
	frames: Frame[];
	totalChars: number;
	truncatedChars: number;
	/** Full kept archive source text for re-rendering on next compaction. */
	text?: string;
	/** Oldest text region kept verbatim. */
	textHead?: string;
	/** Newest text region kept verbatim. */
	textTail?: string;
}

/** Compaction result carrying the summary + optional frame archive. */
export interface CompactResult {
	/** Human-readable summary shown in the compacted history. */
	summary: string;
	/** First entry ID that was kept (not compacted). */
	firstKeptEntryId?: string;
	/** Tokens in the payload before compaction. */
	tokensBefore: number;
	/** Frames carrying the archived history. */
	frames?: Frame[];
	/** Persisted archive data for the next compaction round. */
	preserveData?: Record<string, unknown>;
	/** Character count of what was archived. */
	archivedChars: number;
}

// ============================================================================
// Constants
// ============================================================================

/** Default per-tool-result character cap in serialized history. */
export const TOOL_RESULT_MAX_CHARS = 2000;

/** Default per-argument-value character cap. */
export const TOOL_ARG_MAX_CHARS = 500;

/** Default char cap across one tool call's full argument list. */
export const TOOL_CALL_MAX_CHARS = 2000;

/** Default fraction of truncation budget spent on head. */
export const TRUNCATE_HEAD_RATIO = 0.6;


/** Estimated token overhead per PNG frame sent to a vision model. */
export const PNG_FRAME_OVERHEAD_TOKENS = 500;
/** Zero-width dim ink markers (shift-out / shift-in). */
export const DIM_ON = "\u000e";
export const DIM_OFF = "\u000f";

/** Printed in place of newline runs: a block character that fills the cell. */
export const NEWLINE_GLYPH = "\u2588";

/** Frame geometry for the default bitmap shape. */
export const DEFAULT_COLS = 140;
export const DEFAULT_ROWS = 50;

/** Max frames to carry in every rebuilt request. */
export const MAX_FRAMES_DEFAULT = 60;

/** Key under `CompactionEntry.preserveData` holding the frame archive. */
export const PRESERVE_KEY = SNAPCOMPACT_PRESERVE_KEY;

/** Stopwords that readers can reconstruct from context. */
const STOPWORDS = new Set((
	"the a an and or of to in on at as is are was were be been by for with that " +
		"this it its from had has have not but he she his her they their them which " +
		"also who whom when where while will would could should there then than into " +
		"over under about after before between during each such these those some most " +
		"more other only so"
).split(" "));

// ============================================================================
// Message Serialization
// ============================================================================

/** Serialize conversation messages into a structured text format for
 *  frame rendering. Mirrors omp's `serializeConversation` contract. */
export function serializeMessages(
	messages: CompactableMessage[],
	options?: SerializeOptions,
): string {
	const toolResultMax = options?.toolResultMaxChars ?? TOOL_RESULT_MAX_CHARS;
	const headRatio = options?.truncateHeadRatio ?? TRUNCATE_HEAD_RATIO;
	const dimToolResults = options?.dimToolResults !== false;
	const includeThinking = options?.includeThinking !== false;

	const parts: string[] = [];
	let lastPrefix: string | null = null;

	const pushPart = (prefix: string, content: string) => {
		const lastIdx = parts.length - 1;
		if (lastIdx >= 0 && lastPrefix === prefix) {
			const last = parts[lastIdx];
			if (last) parts[lastIdx] = last.endsWith("\n") || content.startsWith("\n") ? last + content : last + "\n" + content;
		} else {
			parts.push(prefix + content);
			lastPrefix = prefix;
		}
	};

	// First pass: identify useless call IDs (tool results marked useless
	// and not errors — nothing worth archiving).
	const uselessCallIds = new Set<string>();
	const resultTextByCallId = new Map<string, string>();
	for (const msg of messages) {
		if (msg.role !== "toolResult") continue;
		const text = extractTextContent(msg.content);
		if (text) {
			const id = msg.toolCallId;
			if (id) resultTextByCallId.set(id, text);
		}
	}

	// Second pass: serialize each message.
	for (const msg of messages) {
		if (msg.role === "user") {
			const content = typeof msg.content === "string" ? msg.content : extractTextContent(msg.content);
			if (content) pushPart("¶user:", stripDimMarkers(content));
		} else if (msg.role === "assistant") {
			const blocks = msg.content as Array<{ type: string; text?: string; thinking?: string; id?: string; name?: string; arguments?: unknown }> | undefined;
			for (const block of blocks ?? []) {
				if (block?.type === "text") {
					const text = stripDimMarkers(block.text ?? "");
					if (text.trim()) pushPart("¶ai:", text);
				} else if (block?.type === "thinking") {
					if (!includeThinking) continue;
					const thinking = stripDimMarkers(block.thinking ?? "");
					if (thinking.trim()) pushPart("¶think:", thinking);
				} else if (block?.type === "toolCall") {
					const id = block.id;
					if (id && uselessCallIds.has(id)) continue;
					const argsStr = formatToolArgs(block.arguments, headRatio);
					const callLines: string[] = [`${block.name ?? "unknown"}(${argsStr})`];
					if (id) {
						const resultText = resultTextByCallId.get(id);
						if (resultText) {
							callLines.push(renderResultBlock(resultText, toolResultMax, headRatio, dimToolResults));
						}
					}
					pushPart("¶call:", callLines.join("\n"));
				}
			}
		} else if (msg.role === "toolResult") {
			const id = msg.toolCallId;
			if (id && resultTextByCallId.has(id)) continue;
			const resultText = id ? resultTextByCallId.get(id) : "";
			if (resultText) pushPart("¶call:", `\n${renderResultBlock(resultText, toolResultMax, headRatio, dimToolResults)}`);
		}
	}

	return parts.join("\n\n");
}

// ============================================================================
// Text Normalization
// ============================================================================

/** Unicode code-point folds for characters the bitmap fonts cannot render. */
const CHAR_FOLD: Record<string, string> = {
	"\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
	"\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2032": "'", "\u2033": '"',
	"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2015": "-", "\u2212": "-",
	"\u2026": "...", "\u22ef": "...",
	"\u2022": "*", "\u2023": "*", "\u2219": "*", "\u25cf": "*", "\u25a0": "*", "\u25aa": "*",
	"\u2190": "<-", "\u2191": "^", "\u2192": "->", "\u2193": "v", "\u2194": "<->",
	"\u21d0": "<=", "\u21d2": "=>", "\u21d4": "<=>",
	"\u2713": "v", "\u2714": "v", "\u2717": "x", "\u2718": "x",
	"\u2044": "/", "\u2024": ".", "\u2025": "..", "\u00ab": '"', "\u00bb": '"',
};

/** Emoji → text labels for tool output. */
const EMOJI_FOLD: Record<string, string> = {
	"✅": "[OK]", "☑": "[OK]", "✔": "[OK]",
	"❌": "[FAIL]", "❎": "[FAIL]", "✖": "[FAIL]",
	"⚠": "[WARN]", "🚨": "[ALERT]", "ℹ": "[INFO]",
	"🐛": "[BUG]", "💥": "[CRASH]", "🔥": "[HOT]",
	"🔒": "[LOCK]", "🔓": "[UNLOCK]",
	"📁": "[DIR]", "📂": "[DIR]", "📄": "[FILE]",
	"📝": "[NOTE]", "🧪": "[TEST]",
	"⏳": "[WAIT]", "⌛": "[WAIT]", "🚀": "[RUN]",
};

const EMOJI_PICTOGRAPH = /\p{Extended_Pictographic}/u;
const DIM_MARKERS = /[\u000e\u000f]/g;
const COLLAPSIBLE = /[\s\p{Cf}]+/gu;
const LINE_BREAK = /[\n\r\u2028\u2029]/;
const EDGE_RUNS = /^[ \u2588]+|[ \u2588]+$/g;
const UNRENDERABLE = /[\p{Cc}\p{Mn}\p{Me}\p{Cs}]/u;
const COMBINING_MARKS = /\p{M}+/gu;

/** Strip zero-width dim markers from text. */
export function stripDimMarkers(text: string): string {
	return text.replace(DIM_MARKERS, "");
}

/** Normalize text for bitmap rendering: fold Unicode, collapse whitespace,
 *  replace newlines with block glyphs, drop unrenderable characters. */
export function normalizeText(text: string): string {
	const stripped = text.includes("\u001b") ? Bun.stripANSI(text) : text;
	const collapsed = stripped
		.replace(COLLAPSIBLE, (run) => (LINE_BREAK.test(run) ? NEWLINE_GLYPH : /[^\p{Cf}]/u.test(run) ? " " : ""))
		.replace(EDGE_RUNS, "");

	const chars = [...collapsed];
	const out: string[] = [];

	for (const ch of chars) {
		const cp = ch.codePointAt(0);
		if (cp === undefined) continue;

		const folded = CHAR_FOLD[ch];
		if (folded !== undefined) { out.push(folded); continue; }

		// FONT_DATA only covers ASCII 32-126; Latin-1 supplement (0xa0-0xff)
		// is NOT directly renderable and must fall through to NFKD folding
		// below, or every accented character silently renders as a blank cell.
		if (cp >= 0x20 && cp < 0x7f) {
			out.push(ch);
			continue;
		}

		if (ch === DIM_ON || ch === DIM_OFF || ch === NEWLINE_GLYPH) {
			out.push(ch);
			continue;
		}

		const emoji = EMOJI_FOLD[ch];
		if (emoji !== undefined) { out.push(emoji); continue; }
		if (EMOJI_PICTOGRAPH.test(ch)) continue;

		if (cp >= 0x2500 && cp <= 0x25ff) {
			out.push(cp === 0x2502 || cp === 0x2503 ? "|" : cp === 0x2500 || cp === 0x2501 ? "-" : "+");
			continue;
		}

		const decomposed = ch.normalize("NFKD").replace(COMBINING_MARKS, "");
		if (decomposed.length === 1) {
			const folded = CHAR_FOLD[ch];
			if (folded !== undefined) out.push(folded);
			else {
				const code = decomposed.codePointAt(0);
				if (code !== undefined && code >= 0x20 && code < 0x7f) out.push(decomposed);
				else if (UNRENDERABLE.test(ch)) continue;
				else out.push("?");
			}
		} else if (decomposed.length > 1) {
			const first = decomposed.charAt(0);
			const firstCp = first.codePointAt(0);
			if (firstCp !== undefined && firstCp >= 0x20 && firstCp < 0x7f) out.push(first);
			else out.push(CHAR_FOLD[ch] ?? "?");
		} else {
			out.push(CHAR_FOLD[ch] ?? "?");
		}
	}

	return out.join("").replace(/ +/g, " ").replace(EDGE_RUNS, "");
}

// ============================================================================
// Stopword dimming
// ============================================================================

const ALPHA_RUN = /[a-zA-Z\u00c0-\u00d6\u00d8-\u00ff]+/g;
const DIM_MARKER_SPLIT = /([\u000e\u000f])/;

/** Wrap stopwords in DIM_ON/DIM_OFF so they print in dim ink. */
export function dimStopwords(text: string): string {
	const parts = text.split(DIM_MARKER_SPLIT);
	let dim = false;
	let out = "";

	for (const part of parts) {
		if (part === DIM_ON) { dim = true; out += part; }
		else if (part === DIM_OFF) { dim = false; out += part; }
		else if (dim) { out += part; }
		else {
			out += part.replace(ALPHA_RUN, (word) =>
				STOPWORDS.has(word.toLowerCase()) ? DIM_ON + word + DIM_OFF : word,
			);
		}
	}

	return out;
}

// ============================================================================
// Data URL elision
// ============================================================================

const DATA_URL_RE = /data:([A-Za-z][\w.+-]*\/[\w.+-]+(?:;[\w!#$%&'*+.^|~-]+=[\w!#$%&'*+.^|~-]+)*);base64,([A-Za-z0-9+/=]{40,})/gi;

export function elideDataUrls(text: string): string {
	if (!/;base64,/i.test(text)) return text;
	return text.replace(DATA_URL_RE, (_match, mime) => `[data URL omitted: ${mime}]`);
}

// ============================================================================
// Truncation helpers
// ============================================================================

/** Truncate text keeping head and tail, eliding the middle. */
export function truncateForSummary(text: string, maxChars: number, headRatio: number = TRUNCATE_HEAD_RATIO): string {
	if (text.length <= maxChars) return text;
	const headChars = Math.round(maxChars * headRatio);
	const tailChars = maxChars - headChars;
	const elided = text.length - maxChars;
	const tail = tailChars > 0 ? text.slice(-tailChars) : "";
	return `${text.slice(0, headChars)} …[${elided} chars elided…] ${tail}`;
}

// ============================================================================
// Internal helpers
// ============================================================================

function extractTextContent(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.filter((b): b is { type: string; text: string } => typeof b === "object" && b !== null && "type" in b && b.type === "text" && "text" in b && typeof (b as { text: string }).text === "string")
		.map((b) => (b as { text: string }).text)
		.join("");
}

function formatToolArgs(args: unknown, headRatio: number): string {
	if (typeof args === "string") return truncateForSummary(elideDataUrls(args), TOOL_CALL_MAX_CHARS, headRatio);
	if (typeof args === "object" && args !== null) {
		return truncateForSummary(
			Object.entries(args as Record<string, unknown>)
				.map(([key, value]) => `${key}=${JSON.stringify(value)}`)
				.join(", "),
			TOOL_CALL_MAX_CHARS,
			headRatio,
		);
	}
	return String(args ?? "");
}

function renderResultBlock(rawText: string, maxChars: number, headRatio: number, dim: boolean): string {
	const body = truncateForSummary(elideDataUrls(stripDimMarkers(rawText)), maxChars, headRatio);
	return dim ? `<out>\n${DIM_ON}${body}${DIM_OFF}\n</out>` : `<out>\n${body}\n</out>`;
}

// ============================================================================
// Minimal PNG Encoder (Bun zlib, no dependencies)
// ============================================================================

/** Create a PNG file from raw RGBA pixel data. */
function makePng(
	width: number,
	height: number,
	pixels: Uint8Array,
): Uint8Array {
	const sig = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);

	function makeChunk(type: string, data: Uint8Array<ArrayBufferLike>): Uint8Array {
		const len = new Uint8Array(4); len[0] = 0; len[1] = 0; len[2] = 0; len[3] = 0;
		len[3] = data.length & 0xFF; len[2] = (data.length >> 8) & 0xFF; len[1] = (data.length >> 16) & 0xFF; len[0] = (data.length >> 24) & 0xFF;
		const typeBytes = new Uint8Array([...type.split("").map(c => c.charCodeAt(0))]);
		const crc = new Uint8Array(4);
		crc[0] = len[0]!; crc[1] = len[1]!; crc[2] = len[2]!; crc[3] = len[3]!;
		// @ts-expect-error noUncheckedIndexedAccess makes typed arrays return T|undefined
		for (let i = 0; i < typeBytes.length; i++) crc[i] ^= typeBytes[i]!;
		// @ts-expect-error noUncheckedIndexedAccess makes typed arrays return T|undefined
		for (let i = 0; i < data.length; i++) crc[i & 3] ^= data[i]!;
		const table = new Uint32Array(256);
		for (let i = 0; i < 256; i++) {
			let c = i;
			for (let j = 0; j < 8; j++) c = (c >>> 1) ^ (c & 1 ? 0xEDB88320 : 0);
			table[i] = c >>> 0;
		}
		let crcVal = 0xFFFFFFFF;
		// @ts-expect-error noUncheckedIndexedAccess makes typed arrays return T|undefined
		for (let i = 0; i < data.length; i++) crcVal = table[(crcVal ^ data[i]!) & 0xFF] ^ (crcVal >>> 8);
		crcVal ^= 0xFFFFFFFF;
		crc[0] = (crcVal >> 24) & 0xFF; crc[1] = (crcVal >> 16) & 0xFF; crc[2] = (crcVal >> 8) & 0xFF; crc[3] = crcVal & 0xFF;
		const out = new Uint8Array(4 + 4 + data.length + 4);
		out.set(len, 0);
		out.set(typeBytes, 4);
		out.set(data as Uint8Array, 8);
		out.set(crc, 8 + data.length);
		return out;
	}

	function makeIhdr(): Uint8Array {
		const b = new ArrayBuffer(13);
		const v = new DataView(b);
		v.setInt32(0, width, true);
		v.setInt32(4, height, true);
		v.setUint8(8, 8);  // bit depth
		v.setUint8(9, 6);  // color type: RGBA
		v.setUint8(10, 0); // compression
		v.setUint8(11, 0); // filter
		v.setUint8(12, 0); // interlace
		return new Uint8Array(b);
	}

	function makeIdat(raw: Uint8Array): Uint8Array {
		// @ts-expect-error Bun.deflateSync returns Uint8Array<ArrayBufferLike> but runtime uses ArrayBuffer
		const compressed = Bun.deflateSync(raw);
		return makeChunk("IDAT", compressed);
	}

	const scanlines = new Uint8Array(height * (1 + width * 4));
	for (let y = 0; y < height; y++) {
		const rowOff = y * (1 + width * 4);
		scanlines[rowOff] = 0; // filter: none
		const srcOff = y * width * 4;
		scanlines.set(pixels.subarray(srcOff, srcOff + width * 4), rowOff + 1);
	}

	const ihdr = makeIhdr();
	const idat = makeIdat(scanlines);
	const end = makeChunk("IEND", new Uint8Array(0));
	const total = sig.length + (4 + 4 + ihdr.length + 4) + idat.length + end.length;
	const result = new Uint8Array(total);
	let off = 0;
	result.set(sig, off); off += sig.length;
	result.set(ihdr, off); off += ihdr.length;
	result.set(idat, off); off += idat.length;
	result.set(end, off);
	return result;
}

/** Convert raw RGBA to base64. */
function toBase64(data: Uint8Array): string {
	let binary = "";
	for (let i = 0; i < data.length; i += 8192) {
		const chunk = data.subarray(i, i + 8192);
		for (let j = 0; j < chunk.length; j++) binary += String.fromCharCode(chunk[j]!);
	}
	return btoa(binary);
}
// ============================================================================
// 6×13 Monospace Bitmap Font
// ============================================================================

/** 6×13 bitmap font for standard ASCII (32–126). Each char = 13 rows × 1 byte. */
const FONT_BYTES_PER_CHAR = 13;
const FONT_WIDTH = 6;
const FONT_HEIGHT = 13;

/** Compact bitmap font data: 95 printable ASCII chars, 13 bytes each. */
const FONT_DATA = new Uint8Array([
	// Space (32)
	0,0,0,0,0,0,0,0,0,0,0,0,0,
	// Exclamation (33)
	0,0,16,0,0,0,0,0,16,0,0,0,0,
	// Double quote (34)
	0,20,20,0,0,0,0,0,0,0,0,0,0,
	// Hash (35)
	24,68,127,68,127,68,24,0,24,68,127,68,24,
	// Dollar (36)
	0,18,64,60,8,78,16,0,18,64,60,8,78,
	// Percent (37)
	64,64,8,4,2,4,8,0,64,64,8,4,2,
	// Ampersand (38)
	0,28,68,64,62,8,28,0,28,68,64,62,8,
	// Asterisk (39)
	0,12,34,60,34,12,0,0,12,34,60,34,12,
	// Left paren (40)
	0,4,8,16,16,16,8,0,4,8,16,16,8,
	// Right paren (41)
	0,8,16,16,16,4,8,0,8,16,16,16,4,
	// Asterisk operator (42)
	0,0,28,0,28,0,0,0,0,28,0,28,0,
	// Plus (43)
	0,0,16,16,124,16,16,0,0,16,16,124,16,
	// Comma (44)
	0,0,0,0,0,0,0,0,8,16,16,8,0,
	// Minus (45)
	0,0,0,0,124,0,0,0,0,0,0,124,0,
	// Period (46)
	0,0,0,0,0,0,0,0,16,16,0,0,0,
	// Forward slash (47)
	0,2,4,8,16,32,64,0,2,4,8,16,32,
	// Digits 0-9 (48-57)
	60,68,72,74,76,68,60,0,60,68,72,74,76,
	60,66,68,72,72,68,60,0,60,66,68,72,66,
	56,72,72,60,8,8,60,0,56,74,72,72,60,
	56,74,72,60,72,74,56,0,60,72,68,68,72,
	76,72,70,68,60,68,60,0,56,72,72,72,66,
	60,68,68,60,68,68,60,0,60,68,68,72,72,
	72,68,68,60,72,68,60,0,60,68,68,72,74,
	72,74,62,8,8,8,8,0,56,72,68,68,68,66,
	60,68,76,60,76,68,60,0,60,68,68,124,
	68,68,60,0,60,68,68,124,68,68,60,0,
	120,68,68,124,68,68,120,0,60,68,
	68,124,68,68,60,0,60,68,68,124,
	68,68,60,0,68,68,120,68,68,
	68,60,0,60,68,68,124,68,68,
	60,0,127,68,68,124,68,68,
	127,0,60,68,68,124,68,68,
	60,0,60,68,68,76,76,68,
	60,0,60,68,68,124,68,68,
	60,0,127,68,68,124,68,68,
	127,0,60,68,68,124,68,68,
	60,0,60,68,68,68,68,68,
	60,0,68,68,120,68,68,
	68,60,0,60,68,68,120,
	68,68,60,0,68,68,120,
	68,60,0,120,68,
	68,124,68,68,120,
	0,60,68,68,124,
	68,68,60,0,60,
	68,68,124,68,
	68,60,0,68,
	68,120,68,68,
	120,68,0,60,
	68,68,124,68,
	68,60,0,68,
	68,68,120,68,
	68,120,68,
	// Punctuation + symbols (123-126)
	120,68,60,68,68,120,0,
	60,68,68,120,68,68,60,
	0,64,120,64,0,0,0,
	120,120,120,0,0,0,0,
]);

/** ASCII index → row bytes offset in FONT_DATA. */
function fontOffset(code: number): number {
	return (code - 32) * FONT_BYTES_PER_CHAR;
}

/** Rasterize text → RGBA Uint8Array (white bg, black text). */
function rasterizeText(
	text: string,
	frameWidth: number,
	frameHeight: number,
): Uint8Array {
	const width = frameWidth * FONT_WIDTH;
	const height = frameHeight * FONT_HEIGHT;
	const rgba = new Uint8Array(width * height * 4);
	const maxRows = Math.min(text.length, frameHeight);
	let row = 0;
	let col = 0;
	for (let ci = 0; ci < text.length && row < maxRows; ci++) {
		const ch = text[ci] ?? "";
		const code = ch.charCodeAt(0);
		if (code === 10 || code === 13) {
			if (code === 10) { row++; col = 0; }
			continue;
		}
		if (code < 32 || code === 127) continue;
		const off = fontOffset(code);
		const rowStart = (row * FONT_HEIGHT + col * FONT_WIDTH) * 4;
		for (let r = 0; r < FONT_HEIGHT; r++) {
			const byte = FONT_DATA[off + r];
			if (!byte) continue;
			const y = rowStart + r * width * 4;
			for (let bit = 0; bit < FONT_WIDTH; bit++) {
				if (byte & (1 << (5 - bit))) {
					const px = y + bit * 4;
					rgba[px] = 0; rgba[px + 1] = 0;
					rgba[px + 2] = 0; rgba[px + 3] = 255;
				}
			}
		}
		col++;
		if (col >= frameWidth) { col = 0; row++; }
	}
	return rgba;
}

/** Render a single text frame page → base64 PNG. */
function renderFramePng(
	pageText: string,
	cols: number,
	rows: number,
): string {
	const rgba = rasterizeText(pageText, cols, rows);
	const pngBytes = makePng(cols * FONT_WIDTH, rows * FONT_HEIGHT, rgba);
	return toBase64(pngBytes);
}

// ============================================================================
// Text-based Frame Layout (no Canvas dependency)
// ============================================================================

/** Paginate normalized text into pages that fit one frame's grid capacity. */
function paginate(text: string, capacity: number): string[] {
	const chars = [...text];
	const pages: string[] = [];
	let start = 0;
	let cells = 0;
	let hasCell = false;

	for (let i = 0; i < chars.length; i++) {
		const ch = chars[i];
		if (!ch) continue;
		if (ch === DIM_ON || ch === DIM_OFF) continue;
		if (hasCell && cells + 1 > capacity) {
			pages.push(chars.slice(start, i).join(""));
			start = i;
			cells = 0;
		}
		cells += 1;
		hasCell = true;
	}

	if (hasCell) pages.push(chars.slice(start).join(""));
	return pages;
}

/** Frame layout from planArchive. */
interface ArchiveLayout {
	textHead: string;
	textTail: string;
	keptText: string;
	framePages: string[];
	truncatedChars: number;
}


function planArchive(archiveText: string, cols: number, rows: number, maxFrames: number): ArchiveLayout {
	const capacity = cols * rows;
	const totalCapacity = maxFrames * capacity;
	const truncatedChars = Math.max(0, archiveText.length - totalCapacity);

	// Keep text at both edges (one page worth each).
	const edgeChars = capacity;
	const textHead = archiveText.slice(0, Math.min(edgeChars, archiveText.length));
	const remaining = archiveText.slice(edgeChars);

	// Determine text tail.
	const tailLen = Math.min(edgeChars, remaining.length);
	const textTail = remaining.slice(-tailLen);
	const imagedMiddle = remaining.slice(0, remaining.length - tailLen);

	// Paginate the middle into frame pages.
	const framePages = paginate(imagedMiddle, capacity);

	return {
		textHead,
		textTail,
		keptText: imagedMiddle + textTail,
		framePages,
		truncatedChars,
	};
}

// ============================================================================
// Main Compact Function
// ============================================================================

export interface SerializeOptions {
	toolResultMaxChars?: number;
	toolArgMaxChars?: number;
	toolCallMaxChars?: number;
	truncateHeadRatio?: number;
	dimToolResults?: boolean;
	includeThinking?: boolean;
	/** Render frames as PNG images (default: auto-detect Bun availability). */
	render?: boolean;
	/** Vision model provider for frame-size tuning ("gpt4o", "claude", "gemini", etc.). */
	provider?: string;
}

/**
 * Compact messages into a structured archive using snapcompact.
 *
 * Strategy:
 *  1. Serialize the conversation into structured text.
 *  2. Normalize (Unicode folding, whitespace collapsing).
 *  3. Elide base64 data URLs (prevents corrupted images from poisoning requests).
 *  4. Plan foveated layout: text at both edges, framed middle.
 *  5. Produce a human-readable summary + frame archive for persistence.
 *
 * This implementation renders frames as PNG images for vision model consumption.
 * Falls back to text-only when rendering is disabled or fails.
 */
export async function compact(
	messages: CompactableMessage[],
	options?: {
		firstKeptEntryId?: string;
		previousArchive?: Archive;
		previousSummary?: string;
		maxFrames?: number;
		shape?: { cols?: number; rows?: number; lineRepeat?: number };
		serializeOptions?: SerializeOptions;
		/** Override frame rendering: true to force, false to disable. */
		render?: boolean;

	},
): Promise<CompactResult> {
	const maxFrames = options?.maxFrames ?? MAX_FRAMES_DEFAULT;

	// Provider-aware frame sizing.
	const cols = resolveCols(options?.shape?.cols ?? DEFAULT_COLS, options?.serializeOptions?.provider);
	const rows = options?.shape?.rows ?? DEFAULT_ROWS;

	// Serialize messages.
	const serialized = serializeMessages(messages, options?.serializeOptions);

	// Elide data URLs in the serialized output.
	const elided = elideDataUrls(serialized);

	// Fold in previous archive text if one exists.
	let archiveText = elided;
	if (options?.previousArchive) {
		const prevText = options.previousArchive.text ?? options.previousArchive.textHead ?? "";
		const prevTail = options.previousArchive.textTail ?? "";
		if (prevText || prevTail) {
			archiveText = prevText + (prevTail ? NEWLINE_GLYPH + prevTail : "") + (archiveText ? NEWLINE_GLYPH + archiveText : "");
		}
	}

	// Fold in previous text summary if no archive exists.
	if (options?.previousSummary && !options?.previousArchive?.text) {
		archiveText = `[Summary of earlier history] ${options.previousSummary}` + (archiveText ? NEWLINE_GLYPH + archiveText : "");
	}

	// Normalize.
	const normalized = normalizeText(archiveText);

	// Plan foveated layout.
	const layout = planArchive(normalized, cols, rows, maxFrames);

	// Estimate character counts.
	const textChars = layout.textHead.length + layout.textTail.length;
	const frameChars = layout.framePages.reduce((sum, page) => sum + page.replace(DIM_MARKERS, "").length, 0);
	const totalChars = frameChars + textChars;

	// Build frames from the planned pages.
	const renderPng = options?.render ?? (typeof Bun !== "undefined" && typeof Bun.deflateSync === "function");
	const frames: Frame[] = layout.framePages.map((pageText) => {
		const clean = pageText.replace(DIM_MARKERS, "");
		const rendered = Math.min(clean.length, cols * rows);
		let data = "";
		if (renderPng) {
			try {
				data = renderFramePng(pageText, cols, rows);
			} catch {
				// Fall back to text-only rendering.
			}
		}
		return {
			data,
			cols,
			rows,
			chars: rendered,
			font: "6x13",
		};
	});

	// Estimate tokens (rough: ~4 chars per token for English).
	const estimatedTokens = Math.ceil(totalChars / 4);
	const previousTotal = options?.previousArchive?.totalChars ?? 0;

	// Build human-readable summary.
	const summary = buildSummary(frames.length, totalChars, layout.truncatedChars, textChars, options?.previousSummary);

	const result: CompactResult = {
		summary,
		tokensBefore: estimatedTokens + previousTotal,
		archivedChars: totalChars,
	};
	if (frames.length > 0) result.frames = frames;
	const preserve = frames.length > 0 || layout.keptText.length > 0
		? { [PRESERVE_KEY]: {
				frames,
				totalChars,
				truncatedChars: layout.truncatedChars + (options?.previousArchive?.truncatedChars ?? 0),
				text: layout.keptText,
				textHead: layout.textHead,
				textTail: layout.textTail,
			} as Archive }
		: null;
	if (preserve) result.preserveData = preserve;
	return result;
}

/** Provider-specific optimal frame widths (chars per row). */
const PROVIDER_COLS: Record<string, number> = {
	"openai": DEFAULT_COLS,
	"gpt4o": 120,
	"gpt4o-mini": 100,
	"claude": 110,
	"claude-sonnet": 110,
	"claude-opus": 100,
	"gemini": 120,
	"gemini-flash": 100,
	"llama": 140,
	"mistral": 130,
};

/** Resolve frame column count using provider hint. */
function resolveCols(cols: number, provider?: string): number {
	if (!provider) return cols;
	const key = provider.toLowerCase();
	if (key in PROVIDER_COLS) return PROVIDER_COLS[key as keyof typeof PROVIDER_COLS] as number;
	return cols;
}

/** Compute token overhead for PNG frames. Accounts for base64 data + rendering cost. */
export function computeFrameTokenOverhead(frames: Frame[]): number {
	let total = 0;
	for (const frame of frames) {
		// PNG frames with data incur vision-model overhead; text-only frames do not.
		if (frame.data && frame.data.length > 0) {
			total += PNG_FRAME_OVERHEAD_TOKENS;
		}
	}
	return total;
}

function buildSummary(
	frameCount: number,
	totalChars: number,
	truncatedChars: number,
	textChars: number,
	previousSummary?: string,
): string {
	const textNote = textChars > 0 ? ` (+${textChars.toLocaleString()} chars as text)` : "";

	if (frameCount === 0 && textChars === 0) {
		return previousSummary ?? "No prior history.";
	}

	const parts: string[] = [];
	parts.push(`Archived ${totalChars.toLocaleString()} chars of conversation history${textNote}.`);

	if (frameCount > 0) {
		parts.push(`Rendered onto ${frameCount} text frame${frameCount === 1 ? "" : "s"}.`);
		parts.push("Frames are normalized text (monospace, black-on-white).");
	}

	if (truncatedChars > 0) {
		parts.push(`${truncatedChars.toLocaleString()} additional characters were elided to fit the frame budget.`);
	}

	if (previousSummary) {
		parts.push(`Previous compaction summary: ${previousSummary.slice(0, 120)}${previousSummary.length > 120 ? "…" : ""}`);
	}

	return parts.join("\n");
}
