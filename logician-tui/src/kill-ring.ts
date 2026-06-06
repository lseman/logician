// ── Kill ring (clipboard) ────────────────────────────────────────────────────
// Emacs-style kill ring: stores deleted text, supports yank and yank-pop.
// Coalesces consecutive kills into a single ring entry.

export interface KillEntry {
    text: string;
}

export class KillRing {
    private entries: KillEntry[] = [];
    private readonly maxSize: number;

    constructor(maxSize = 60) {
        this.maxSize = maxSize;
    }

    push(
        text: string,
        opts?: { prepend?: boolean; accumulate?: boolean },
    ): void {
        if (!text) return;

        if (opts?.accumulate && this.entries.length > 0) {
            // Append to last entry (consecutive kills)
            const last = this.entries[this.entries.length - 1];
            last.text = opts.prepend ? text + last.text : last.text + text;
        } else {
            const entry: KillEntry = { text };
            if (opts?.prepend) {
                entry.text = text;
            }
            this.entries.push(entry);
            if (this.entries.length > this.maxSize) this.entries.shift();
        }
    }

    pop(): string | null {
        if (this.entries.length === 0) return null;
        const entry = this.entries.pop()!;
        return entry.text;
    }

    peek(): string | null {
        if (this.entries.length === 0) return null;
        return this.entries[this.entries.length - 1].text;
    }

    rotate(): void {
        if (this.entries.length < 2) return;
        const last = this.entries.pop()!;
        this.entries.unshift(last);
    }

    get length(): number {
        return this.entries.length;
    }

    clear(): void {
        this.entries = [];
    }
}
