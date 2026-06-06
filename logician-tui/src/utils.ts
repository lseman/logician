// ── Utility primitives ────────────────────────────────────────────────────────
// Grapheme segmentation, visible width, text wrapping, fuzzy matching

// ── Grapheme segmenter (Unicode-aware) ────────────────────────────────────────

export const getGraphemeSegmenter = (): Intl.Segmenter => {
    try {
        return new (Intl as any).Segmenter(undefined, {
            usage: "segmentation",
            segmenter: "grapheme",
        });
    } catch {
        // Fallback: Intl.Segmenter not available — BMP-only splitter
        return {
            segment(
                text: string,
            ): Iterable<{ segment: string; segmented: boolean }> {
                const chars = [...text];
                return {
                    *[Symbol.iterator]() {
                        for (const ch of chars) {
                            yield { segment: ch, segmented: false };
                        }
                    },
                };
            },
        } as unknown as Intl.Segmenter;
    }
};

// ── Visible width ─────────────────────────────────────────────────────────────

export function visibleWidth(text: string): number {
    let width = 0;
    let inEscape = false;

    for (let i = 0; i < text.length; i++) {
        const ch = text[i];
        if (ch === "\x1b" && text[i + 1] === "[") {
            inEscape = true;
            i += 1;
        } else if (inEscape) {
            const c = ch.charCodeAt(0);
            if (c >= 0x40 && c <= 0x7e) inEscape = false;
        } else {
            const code = ch.charCodeAt(0);
            width +=
                code >= 0x1100 &&
                (code <= 0x115f ||
                    code === 0x2329 ||
                    code === 0x232a ||
                    (code >= 0x2e80 && code <= 0xa4cf && code !== 0x303f) ||
                    (code >= 0xac00 && code <= 0xd7a3) ||
                    (code >= 0xf900 && code <= 0xfaff) ||
                    (code >= 0xfe10 && code <= 0xfe19) ||
                    (code >= 0xfe30 && code <= 0xfe6f) ||
                    (code >= 0xff00 && code <= 0xff60) ||
                    (code >= 0xffe0 && code <= 0xffe6) ||
                    (code >= 0x20000 && code <= 0x2fffd) ||
                    (code >= 0x30000 && code <= 0x3fffd))
                    ? 2
                    : 1;
        }
    }
    return width;
}

// ── Slice by column width ─────────────────────────────────────────────────────

export function sliceByColumn(
    text: string,
    startCol: number,
    endCol: number,
    byGrapheme = false,
): string {
    const segmenter = byGrapheme ? getGraphemeSegmenter() : null;
    const segments = segmenter
        ? [...segmenter.segment(text)].map((s) => s.segment)
        : [...text];
    let col = 0;
    let inEscape = false;

    // We need character-level control to strip ANSI codes, so work on chars
    // but measure width properly.
    let result = "";
    let currentCol = 0;
    let started = false;
    let i = 0;
    const chars = [...text];

    while (i < chars.length) {
        const ch = chars[i];
        if (ch === "\x1b" && chars[i + 1] === "[") {
            if (!started) result += ch;
            i += 1;
            inEscape = true;
        } else if (inEscape) {
            if (!started) result += ch;
            if (ch.charCodeAt(0) >= 0x40 && ch.charCodeAt(0) <= 0x7e)
                inEscape = false;
            i += 1;
        } else {
            const w = visibleWidth(ch);
            if (currentCol >= endCol) break;
            if (currentCol >= startCol) {
                result += ch;
            }
            currentCol += w;
            i += 1;
        }
    }
    return result;
}

// ── Word navigation ───────────────────────────────────────────────────────────

export function findWordBackward(text: string, cursor: number): number {
    if (cursor === 0) return 0;
    const segmenter = getGraphemeSegmenter();
    const segments = [...segmenter.segment(text)].map((s) => s.segment);

    // Skip non-word chars going backward
    let i = cursor;
    while (i > 0 && !isWordChar(segments[i - 1])) i--;
    const start = i;

    // Skip word chars going backward
    while (i > 0 && isWordChar(segments[i - 1])) i--;
    return i;
}

export function findWordForward(text: string, cursor: number): number {
    if (cursor >= text.length) return text.length;
    const segmenter = getGraphemeSegmenter();
    const segments = [...segmenter.segment(text)].map((s) => s.segment);

    // Skip non-word chars going forward
    let i = cursor;
    while (i < segments.length && !isWordChar(segments[i])) i++;
    const start = i;

    // Skip word chars going forward
    while (i < segments.length && isWordChar(segments[i])) i++;
    return i;
}

function isWordChar(seg: string): boolean {
    const ch = seg.trim();
    if (!ch) return false;
    const c = ch.charCodeAt(0);
    return (
        (c >= 0x30 && c <= 0x39) || // 0-9
        (c >= 0x41 && c <= 0x5a) || // A-Z
        (c >= 0x61 && c <= 0x7a) || // a-z
        c === 0x5f || // _
        c >= 0x0100
    ); // Unicode letter/digit
}

export function isWhitespaceChar(ch: string): boolean {
    return ch.trim().length === 0;
}

// ── Word boundaries (for grapheme cursor) ─────────────────────────────────────

export function graphemeLength(text: string): number {
    const segmenter = getGraphemeSegmenter();
    return [...segmenter.segment(text)].length;
}

export function graphemeSlice(text: string, from: number, to?: number): string {
    const segmenter = getGraphemeSegmenter();
    const segments = [...segmenter.segment(text)];
    const end = to !== undefined ? to : segments.length;
    return segments
        .slice(from, end)
        .map((s) => s.segment)
        .join("");
}

// ── Text wrapping ─────────────────────────────────────────────────────────────

export function wrapText(text: string, maxLineLength: number): string[] {
    const lines: string[] = [];
    const rawLines = text.split("\n");

    for (const rawLine of rawLines) {
        if (rawLine.length <= maxLineLength) {
            lines.push(rawLine);
        } else {
            const words = rawLine.split(/\s+/);
            let current = "";
            for (const word of words) {
                if (current.length === 0) {
                    current = word;
                } else if (current.length + 1 + word.length <= maxLineLength) {
                    current += " " + word;
                } else {
                    lines.push(current);
                    current = word;
                }
            }
            if (current) lines.push(current);
        }
    }

    return lines;
}

// ── Fuzzy matching (for slash commands) ───────────────────────────────────────

export interface FuzzyScore {
    index: number;
    score: number;
}

export function fuzzyMatch(
    query: string,
    candidate: string,
): FuzzyScore | null {
    const q = query.toLowerCase();
    const c = candidate.toLowerCase();

    if (!q) return { index: 0, score: 1000 };

    // Exact match
    if (c === q) return { index: 0, score: 2000 };

    // Prefix match
    if (c.startsWith(q)) {
        return { index: 0, score: 1800 - (c.length - q.length) };
    }

    // Subsequence match
    let qi = 0;
    let gap = 0;
    let lastMatch = -1;

    for (let ci = 0; ci < c.length && qi < q.length; ci++) {
        if (c[ci] === q[qi]) {
            if (lastMatch >= 0) {
                gap += ci - lastMatch - 1;
            }
            lastMatch = ci;
            qi++;
        }
    }

    if (qi === q.length) {
        // Bonus for matching at start
        const startBonus = c.indexOf(q[0]) === 0 ? 500 : 0;
        return { index: 0, score: 1200 - gap * 10 + startBonus };
    }

    // Description match (lower priority)
    if (c.includes(q)) {
        return { index: 0, score: 600 - c.indexOf(q) };
    }

    return null;
}

export function fuzzyFilter<T>(
    items: T[],
    query: string,
    getKey: (item: T) => string,
): (T & { score: number })[] {
    const scored = items
        .map((item, idx) => {
            const key = getKey(item);
            const result = fuzzyMatch(query, key);
            if (!result) {
                // Check description too
                const descMatch = fuzzyMatch(query, key.split(" ").pop() || "");
                if (!descMatch) return null;
                return { item, score: descMatch.score };
            }
            return { item, score: result.score };
        })
        .filter(Boolean) as unknown as (T & { score: number })[];

    scored.sort((a, b) => b.score - a.score);
    return scored;
}

// ── Clamp string to width (preserving ANSI) ───────────────────────────────────

export function clampLineToWidth(text: string, width: number): string {
    let result = "";
    let visible = 0;
    let i = 0;

    while (i < text.length) {
        const ch = text[i];
        if (ch === "\x1b") {
            const next = text[i + 1];
            if (next === "[") {
                let j = i + 2;
                while (
                    j < text.length &&
                    !(text.charCodeAt(j) >= 0x40 && text.charCodeAt(j) <= 0x7e)
                )
                    j++;
                result += text.slice(i, j + 1);
                i = j + 1;
                continue;
            }
            if (next === "]") {
                let j = i + 2;
                while (
                    j < text.length &&
                    text[j] !== "\x07" &&
                    !(text[j] === "\x1b" && text[j + 1] === "\\")
                )
                    j++;
                const end = text[j] === "\x07" ? j + 1 : j + 2;
                result += text.slice(i, end);
                i = end;
                continue;
            }
            result += ch;
            i++;
            continue;
        }
        const w = visibleWidth(ch);
        if (visible + w > width) break;
        result += ch;
        visible += w;
        i++;
    }
    return result;
}
