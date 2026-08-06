import { createHash, randomUUID } from "node:crypto";
import type { Writable } from "node:stream";
import type { ParsedBridgeEvent } from "@logician/coding-agent/runtime";

export const EXEC_STREAM_SCHEMA = "logician.exec-stream";
export const EXEC_STREAM_SCHEMA_VERSION = 1;

export interface ExecBridge {
	on(callback: (event: ParsedBridgeEvent) => void): () => void;
	onError(callback: (error: Error) => void): void;
	init(): Promise<Record<string, unknown>>;
	sendMessage(message: string): Promise<void>;
	stop(): Promise<void>;
	getConfig(): { baseUrl: string; model: string };
	respondToPermission(
		toolCallId: string,
		decision: "allow" | "deny" | "always",
	): boolean;
	respondToQuestion(questionId: string, answer: string): boolean;
}

export interface HeadlessExecOptions {
	prompt: string;
	jsonl: boolean;
	cwd: string;
	stdout: Pick<Writable, "write">;
	stderr: Pick<Writable, "write">;
	now?: () => number;
	runId?: string;
}

type StreamRecord = Record<string, unknown> & { type: string };

function sha256(value: string): string {
	return `sha256:${createHash("sha256").update(value).digest("hex")}`;
}

function errorMessage(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

export async function runHeadlessExec(
	bridge: ExecBridge,
	options: HeadlessExecOptions,
): Promise<number> {
	const now = options.now ?? Date.now;
	const startedAt = now();
	const runId = options.runId ?? `exec_${randomUUID()}`;
	let output = "";
	let lastError: string | undefined;
	let contextTokens: number | undefined;
	let maxContextTokens: number | undefined;
	const toolStarts = new Map<string, number>();

	const emit = (record: StreamRecord): void => {
		if (!options.jsonl) return;
		options.stdout.write(
			`${JSON.stringify({
				...record,
				schema: EXEC_STREAM_SCHEMA,
				schema_version: EXEC_STREAM_SCHEMA_VERSION,
				run_id: runId,
			})}\n`,
		);
	};

	const unsubscribe = bridge.on(event => {
		switch (event.type) {
			case "token":
				output += event.token;
				if (options.jsonl) emit({ type: "content", content: event.token });
				else options.stdout.write(event.token);
				break;
			case "tool_start":
			case "tool_execution_start": {
				const id =
					event.tool_call_id ?? `${event.tool_name}:${toolStarts.size}`;
				toolStarts.set(id, now());
				emit({
					type: "tool_use",
					id,
					name: event.tool_name,
					input: event.tool_args ?? {},
					started_at: new Date(toolStarts.get(id) ?? now()).toISOString(),
				});
				if (!options.jsonl) options.stderr.write(`tool: ${event.tool_name}\n`);
				break;
			}
			case "tool_end":
			case "tool_execution_end": {
				const id = event.tool_call_id ?? event.tool_name;
				const completedAt = now();
				const toolStartedAt = toolStarts.get(id) ?? completedAt;
				emit({
					type: "tool_result",
					id,
					name: event.tool_name,
					output: event.result ?? "",
					status: event.is_error ? "error" : "success",
					started_at: new Date(toolStartedAt).toISOString(),
					completed_at: new Date(completedAt).toISOString(),
					duration_ms: Math.max(0, completedAt - toolStartedAt),
					...(event.details ? { result_metadata: event.details } : {}),
				});
				break;
			}
			case "context_update":
				contextTokens = event.tokens;
				maxContextTokens = event.max_tokens;
				break;
			case "permission_request":
				lastError = `Headless execution denied interactive permission for ${event.tool_name}`;
				bridge.respondToPermission(event.tool_call_id, "deny");
				emit({ type: "error", error: lastError });
				break;
			case "question_request":
				lastError = "Headless execution cannot answer an interactive question";
				bridge.respondToQuestion(event.question_id, "__dismissed__");
				emit({ type: "error", error: lastError });
				break;
			case "notice":
				if (event.level === "error") {
					lastError = event.text;
					emit({ type: "error", error: event.text });
				}
				break;
			default:
				break;
		}
	});

	bridge.onError(error => {
		lastError = error.message;
	});

	try {
		await bridge.init();
		await bridge.sendMessage(options.prompt);
	} catch (error: unknown) {
		lastError = errorMessage(error);
		emit({ type: "error", error: lastError });
	} finally {
		unsubscribe();
		try {
			await bridge.stop();
		} catch (error: unknown) {
			lastError ??= errorMessage(error);
			emit({ type: "error", error: errorMessage(error) });
		}
	}

	if (!options.jsonl && output.length > 0 && !output.endsWith("\n")) {
		options.stdout.write("\n");
	}

	const config = bridge.getConfig();
	emit({
		type: "metadata",
		meta: {
			receipt_kind: "terminal",
			model: config.model,
			workspace: options.cwd,
			duration_ms: Math.max(0, now() - startedAt),
			prompt_sha256: sha256(options.prompt),
			visible_final_answer_chars: [...output].length,
			status: lastError ? "failed" : "completed",
			...(lastError ? { error: lastError } : {}),
			...(contextTokens === undefined ? {} : { context_tokens: contextTokens }),
			...(maxContextTokens === undefined
				? {}
				: { context_max_tokens: maxContextTokens }),
		},
	});
	emit({ type: "done" });

	if (lastError && !options.jsonl) {
		options.stderr.write(`error: ${lastError}\n`);
	}
	return lastError ? 1 : 0;
}

export interface ParsedExecArgs {
	jsonl: boolean;
	prompt: string;
}

export function parseExecArgs(args: readonly string[]): ParsedExecArgs {
	let jsonl = false;
	let optionsEnded = false;
	const promptParts: string[] = [];
	for (const argument of args) {
		if (!optionsEnded && argument === "--jsonl") {
			jsonl = true;
			continue;
		}
		if (!optionsEnded && argument === "--") {
			optionsEnded = true;
			continue;
		}
		if (!optionsEnded && argument.startsWith("-")) {
			throw new Error(`Unknown exec option: ${argument}`);
		}
		promptParts.push(argument);
	}
	const prompt = promptParts.join(" ").trim();
	if (!prompt) {
		throw new Error("Usage: logician exec [--jsonl] <prompt>");
	}
	return { jsonl, prompt };
}
