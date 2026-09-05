// ── Acceptance Contract ───────────────────────────────────────────────────
// Outcome verification via deterministic commands and stop rules.
// Self-reporting removed — the model does not reliably self-assess.
// Verification commands are the authoritative check.

import type {
	AcceptanceConfig,
	AcceptanceCriterion,
	AcceptanceVerification,
} from "../../system/types/acceptance.ts";

export type AcceptanceLevel = "none" | "verified";

export interface ResolvedAcceptance {
	level: AcceptanceLevel;
	explicit: boolean;
	criteria: AcceptanceCriterion[];
	verify: AcceptanceVerification[];
	stopRules?: string[];
}

export interface AcceptanceLedger {
	status: "passed" | "failed" | "timeout" | "not-required";
	verification?: Array<{
		command: string;
		result: "passed" | "failed";
		summary?: string;
	}>;
}

export interface AcceptanceVerificationResult {
	command: string;
	result: "passed" | "failed";
	summary?: string;
}

export function formatVerificationRepair(
	results: readonly AcceptanceVerificationResult[],
): string {
	const failures = results.filter(result => result.result === "failed");
	return [
		"[verification-repair] Deterministic verification failed. Fix the underlying issue, then finish normally.",
		...failures.map(
			failure =>
				`- ${failure.command}: ${failure.summary?.trim() || "non-zero exit"}`,
		),
		"Do not merely rewrite the acceptance report; change or diagnose the workspace and rerun relevant checks.",
	].join("\n");
}


export function resolveEffectiveAcceptance(params: {
	explicit?: AcceptanceConfig;
}): ResolvedAcceptance {
	const explicit = params.explicit;
	if (!explicit) {
		return {
			level: "none",
			explicit: false,
			criteria: [],
			verify: [],
			stopRules: [],
		};
	}

	const criteria = normalizeCriteria(explicit.criteria ?? []);
	const verify = explicit.verify ?? [];

	let level: AcceptanceLevel = "none";
	if (verify.length > 0) level = "verified";

	return {
		level,
		explicit: true,
		criteria,
		verify,
		stopRules: explicit.stopRules ?? [],
	};
}

function normalizeCriteria(
	input: string[] | AcceptanceCriterion[],
): AcceptanceCriterion[] {
	if (input.length === 0) return [];
	const result: AcceptanceCriterion[] = [];
	for (let i = 0; i < input.length; i++) {
		const item = input[i];
		if (typeof item === "string") {
			result.push({
				id: `criterion-${i + 1}`,
				must: item,
				severity: "required",
			});
		} else {
			result.push({
				id: item.id || `criterion-${i + 1}`,
				must: item.must,
				severity: item.severity ?? "required",
			});
		}
	}
	return result;
}

export function shouldRunAcceptanceFinalization(
	resolved: ResolvedAcceptance,
): boolean {
	return resolved.verify.length > 0;
}

export async function verifyAcceptanceCommands(
	resolved: ResolvedAcceptance,
	options: { cwd?: string; signal?: AbortSignal } = {},
): Promise<AcceptanceVerificationResult[]> {
	if (!resolved.verify.length) return [];
	const { execFile } = await import("node:child_process");
	const { promisify } = await import("node:util");
	const execFileAsync = promisify(execFile);

	return Promise.all(
		resolved.verify.map(
			verification =>
				new Promise<AcceptanceVerificationResult>(resolve => {
					const timeout = verification.timeoutMs ?? 30_000;
					const timeoutId = setTimeout(() => {
						resolve({
							command: verification.command,
							result: "failed",
							summary: `Timeout after ${timeout}ms`,
						});
					}, timeout);
					if (options.signal?.aborted) {
						clearTimeout(timeoutId);
						resolve({
							command: verification.command,
							result: "failed",
							summary: "Aborted",
						});
						return;
					}

					execFileAsync("bash", ["-c", verification.command], {
						cwd: verification.cwd ?? options.cwd,
						timeout,
						maxBuffer: 1024 * 1024,
					}).then(
						(output: { stdout?: string; stderr?: string }) => {
							clearTimeout(timeoutId);
							resolve({
								command: verification.command,
								result: "passed",
								summary: (output.stdout ?? "").trim().slice(0, 500),
							});
						},
						(error: NodeJS.ErrnoException) => {
							clearTimeout(timeoutId);
							resolve({
								command: verification.command,
								result: verification.allowFailure
									? "passed"
									: "failed",
								summary: error.message.slice(0, 500),
							});
						},
					);
				}),
		),
	).then(results => results.flat());
}

export function validateAcceptanceInput(config: AcceptanceConfig): string[] {
	const errors: string[] = [];
	const validKeys = new Set(["criteria", "verify", "stopRules"]);
	for (const key of Object.keys(config)) {
		if (!validKeys.has(key)) {
			errors.push(`Unknown acceptance config key: ${key}`);
		}
	}
	if (!config.criteria && !config.verify) {
		errors.push("Must specify at least one of: criteria, verify");
	}
	if (config.criteria) {
		const items = config.criteria;
		for (let i = 0; i < items.length; i++) {
			const item = items[i];
			if (typeof item === "string") {
				if (!item.trim()) errors.push(`criteria[${i}] is empty`);
			} else {
				if (!item.id?.trim()) errors.push(`criteria[${i}]: id is required`);
				if (!item.must.trim()) errors.push(`criteria[${i}]: must is required`);
			}
		}
	}
	return errors;
}
