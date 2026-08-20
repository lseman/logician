// ── JSONL decode errors ───────────────────────────────────────────────────
// Ported from pi coding agent's harness/session/jsonl/errors.ts.

import type { FileError } from "../../../core/errors.ts";
import type { Result } from "../../../core/result.ts";
import { SessionError } from "../types.ts";

export class JsonlDecodeError extends Error {
	readonly kind: "syntax" | "schema";

	constructor(kind: "syntax" | "schema", message: string, cause?: Error) {
		super(message, cause === undefined ? undefined : { cause });
		this.name = "JsonlDecodeError";
		this.kind = kind;
	}
}

export function fileResult<T>(
	result: Result<T, FileError>,
	message: string,
): T {
	if (!result.ok) {
		throw new SessionError(
			result.error.code === "not_found" ? "not_found" : "storage",
			`${message}: ${result.error.message}`,
			result.error,
		);
	}
	return result.value;
}

export function invalidFile(
	path: string,
	line: number,
	cause: Error,
): SessionError {
	return new SessionError(
		"invalid_entry",
		`Invalid JSONL session ${path}: line ${line} ${cause.message}`,
		cause,
	);
}
