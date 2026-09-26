// ── Native snapcompact renderer ────────────────────────────────────────────────
// The frame renderer lives in the Rust `snapcompact` crate (crates/snapcompact),
// linked into the shared @logician/log-natives addon. That crate registers its
// N-API functions itself at load time, so napi's codegen never sees them and
// the generated log-natives typings don't declare them. Their contract is
// declared here instead, next to the only code that uses it.

import * as natives from "@logician/log-natives";

/**
 * Render one snapcompact frame on a libuv worker: print pre-normalized text
 * onto a `size`-wide bitmap and encode it as PNG.
 *
 * The bitmap height hugs the rows the text actually occupies
 * (`usedRows * lineRepeat * cellHeight`), so a partially filled frame never
 * pays for blank padding rows. The glyph grid holds `floor(size/cellWidth) *
 * floor(size/cellHeight/lineRepeat)` characters; input beyond that is ignored.
 * Native-cell bitmap-font shapes encode as indexed PNG; stretched bitmap-font
 * shapes (target cell != font cell) encode as RGB. TrueType shapes encode RGB
 * directly from grayscale coverage.
 * `stretch: false` pins bitmap fonts to the indexed path, printing
 * natural-size glyphs on the requested cell box; `columns: 2` flows
 * pre-wrapped newline-separated lines down two newspaper columns.
 * `U+000E`/`U+000F` in `text` toggle dim-gray ink spans without occupying a
 * cell.
 * Returns a promise for the PNG encoded as base64, created as a one-byte
 * (Latin-1) JS string straight from native code — no `Uint8Array` hop or
 * JS-side re-encode.
 */
export type RenderSnapcompactPng = (
	text: string,
	options: SnapcompactRenderOptions,
) => Promise<string>;

/** Shape options for one snapcompact frame. */
export interface SnapcompactRenderOptions {
	/**
	 * Frame width in pixels; also bounds the grid rows
	 * (`floor(size/cellHeight/lineRepeat)`). Output height hugs the rows the
	 * text actually uses instead of padding to a square.
	 */
	size: number;
	/**
	 * Bundled font: `"5x8"`, `"6x12"`, `"8x13"` (X.org BDF), `"8x8"`
	 * (unscii-8), or `"silver"` (embedded TrueType). Default `"5x8"`.
	 */
	font?: string;
	/**
	 * Target cell advance in pixels. Differing from the font's natural cell
	 * triggers the Lanczos stretch path. Default: font natural width.
	 */
	cellWidth?: number;
	/** Target cell pitch in pixels. Default: font natural height. */
	cellHeight?: number;
	/**
	 * Ink variant: `"sent"` (six-hue sentence cycling) or `"bw"` (black).
	 * Default `"sent"`.
	 */
	variant?: string;
	/**
	 * Print each text line this many times; copies after the first sit on a
	 * pale highlight band. Default 1.
	 */
	lineRepeat?: number;
	/**
	 * Stretch behavior. Unset: auto — Lanczos-stretch whenever the target
	 * cell differs from the font's natural cell. `false`: never stretch —
	 * render indexed with glyphs at natural size on the requested cell box
	 * (e.g. 8x13 glyphs on an 8x16 pitch, the "8on16" shapes). `true`: force
	 * the stretch path (identical to auto; natural cells render indexed).
	 */
	stretch?: boolean;
	/**
	 * Layout columns: `1` (default) row-major grid; `2` two newspaper "doc"
	 * columns of pre-wrapped newline-separated lines.
	 */
	columns?: number;
}

/**
 * Return the subset of `chars` that the named snapcompact font can render.
 *
 * The TypeScript normalizer uses this to keep Unicode text intact only when
 * the selected native font has a glyph for it; renderer control codes are
 * considered renderable because they are interpreted outside font lookup.
 */
export type SnapcompactSupportedChars = (font: string, chars: string) => string;

interface SnapcompactNative {
	renderSnapcompactPng: RenderSnapcompactPng;
	snapcompactSupportedChars: SnapcompactSupportedChars;
}

const native = natives as unknown as SnapcompactNative;

export const renderSnapcompactPng: RenderSnapcompactPng = (text, options) =>
	native.renderSnapcompactPng(text, options);

export const snapcompactSupportedChars: SnapcompactSupportedChars = (
	font,
	chars,
) => native.snapcompactSupportedChars(font, chars);
