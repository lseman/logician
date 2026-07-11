// ── Deterministic ID generation ──────────────────────────────────────────
// SHA-256 based, 12-char hex. Matches pi-observational-memory.

import { createHash } from "node:crypto";

export function hashId(content: string): string {
	return createHash("sha256").update(content).digest("hex").slice(0, 12);
}
