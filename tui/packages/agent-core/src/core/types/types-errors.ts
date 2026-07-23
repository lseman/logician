// ── Error types ───────────────────────────────────────────────────────────

/** Result of a fallible operation. Expected failures are returned as `ok: false` instead of thrown. */
export type Result<TValue, TError> = { ok: true; value: TValue } | { ok: false; error: TError };

/** Create a successful Result. */
export function ok<TValue, TError>(value: TValue): Result<TValue, TError> {
	return { ok: true, value };
}

/** Create a failed Result. */
export function err<TValue, TError>(error: TError): Result<TValue, TError> {
	return { ok: false, error };
}

/** Return the success value or throw the failure error. */
export function getOrThrow<TValue, TError>(result: Result<TValue, TError>): TValue {
	if (!result.ok) throw result.error;
	return result.value;
}

/** Return the success value or undefined. Only object values are allowed. */
export function getOrUndefined<TValue extends object, TError>(result: Result<TValue, TError>): TValue | undefined {
	return result.ok ? result.value : undefined;
}

/** Normalize unknown thrown values into Error instances. */
export function toError(error: unknown): Error {
	if (error instanceof Error) return error;
	if (typeof error === "string") return new Error(error);
	try {
		return new Error(JSON.stringify(error));
	} catch (e: unknown) {
		return new Error(String(error));
	}
}

/** Stable, backend-independent file error codes. */
export type FileErrorCode =
	| "aborted"
	| "not_found"
	| "permission_denied"
	| "not_directory"
	| "is_directory"
	| "invalid"
	| "not_supported"
	| "unknown";

/** Error returned by file operations. */
export class FileError extends Error {
	public code: FileErrorCode;
	public path?: string;

	constructor(code: FileErrorCode, message: string, path?: string, cause?: Error) {
		super(message, cause === undefined ? undefined : { cause });
		this.name = "FileError";
		this.code = code;
		this.path = path;
	}
}

/** Stable, backend-independent execution error codes. */
export type ExecutionErrorCode =
	| "aborted"
	| "timeout"
	| "shell_unavailable"
	| "spawn_error"
	| "callback_error"
	| "unknown";

/** Error returned by execution env. */
export class ExecutionError extends Error {
	public code: ExecutionErrorCode;

	constructor(code: ExecutionErrorCode, message: string, cause?: Error) {
		super(message, cause === undefined ? undefined : { cause });
		this.name = "ExecutionError";
		this.code = code;
	}
}

/** Stable compaction error codes. */
export type CompactionErrorCode = "aborted" | "summarization_failed" | "invalid_session" | "unknown";

/** Error returned by compaction helpers. */
export class CompactionError extends Error {
	public code: CompactionErrorCode;

	constructor(code: CompactionErrorCode, message: string, cause?: Error) {
		super(message, cause === undefined ? undefined : { cause });
		this.name = "CompactionError";
		this.code = code;
	}
}

/** Stable branch-summary error codes. */
export type BranchSummaryErrorCode = "aborted" | "summarization_failed" | "invalid_session";

/** Error returned by branch summarization helpers. */
export class BranchSummaryError extends Error {
	public code: BranchSummaryErrorCode;

	constructor(code: BranchSummaryErrorCode, message: string, cause?: Error) {
		super(message, cause === undefined ? undefined : { cause });
		this.name = "BranchSummaryError";
		this.code = code;
	}
}

/** Stable session error codes. */
export type SessionErrorCode =
	| "not_found"
	| "invalid_session"
	| "invalid_entry"
	| "invalid_fork_target"
	| "storage"
	| "unknown";

/** Error thrown by session storage and session operations. */
export class SessionError extends Error {
	public code: SessionErrorCode;

	constructor(code: SessionErrorCode, message: string, cause?: Error) {
		super(message, cause === undefined ? undefined : { cause });
		this.name = "SessionError";
		this.code = code;
	}
}

export enum AgentErrorType {
	TURN_TIMEOUT = "turn_timeout",
	CONTEXT_FULL = "context_full",
	PROVIDER_ERROR = "provider_error",
	ABORTED = "aborted",
	TOOL_EXECUTION_FAILED = "tool_execution_failed",
	TOOL_ARGUMENT_ERROR = "tool_argument_error",
	TOOL_DUPLICATE_CALL = "tool_duplicate_call",
	TOOL_FAILURE_LOOP = "tool_failure_loop",
	HOOK_FAILED = "hook_failed",
	INVALID_CONFIG = "invalid_config",
}

export interface AgentErrorOptions {
	type: AgentErrorType;
	message: string;
	cause?: unknown;
	turnId?: string;
	toolName?: string;
	retryable?: boolean;
}

export class AgentError extends Error {
	readonly type: AgentErrorType;
	readonly cause?: unknown;
	readonly turnId?: string;
	readonly toolName?: string;
	readonly retryable: boolean;

	constructor(options: AgentErrorOptions) {
		super(options.message);
		if (options.cause) {
			Object.defineProperty(this, "cause", {
				value: options.cause,
				writable: true,
				enumerable: false,
			});
		}
		this.name = "AgentError";
		this.type = options.type;
		this.cause = options.cause;
		this.turnId = options.turnId;
		this.toolName = options.toolName;
		this.retryable = options.retryable ?? this.isDefaultRetryable(options.type);
	}

	private isDefaultRetryable(type: AgentErrorType): boolean {
		return (
			type === AgentErrorType.PROVIDER_ERROR ||
			type === AgentErrorType.CONTEXT_FULL
		);
	}
}

export function wrapError(
	type: AgentErrorType,
	original: Error,
	extra?: Partial<AgentErrorOptions>,
): AgentError {
	return new AgentError({
		type,
		message: original.message,
		cause: original,
		...extra,
	});
}
