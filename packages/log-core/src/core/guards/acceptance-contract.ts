// ── Acceptance Contract ───────────────────────────────────────────────────
// Outcome verification gated by an acceptance contract defined in AgentConfig.
// Deterministic commands are authoritative; the final report supplies evidence
// for criteria that cannot be checked mechanically.

export type EvidenceKind =
	| "changed-files"
	| "tests-added"
	| "commands-run"
	| "validation-output"
	| "residual-risks"
	| "no-staged-files"
	| "diff-summary"
	| "manual-notes";

export type CriterionSeverity = "required" | "recommended";

export type AcceptanceLevel = "none" | "checked" | "verified";

export interface AcceptanceCriterion {
	id: string;
	must: string;
	evidence?: EvidenceKind[];
	severity?: CriterionSeverity;
}

export interface AcceptanceVerification {
	id: string;
	command: string;
	cwd?: string;
	timeoutMs?: number;
	allowFailure?: boolean;
}

export interface AcceptanceConfig {
	criteria?: string[] | AcceptanceCriterion[];
	evidence?: EvidenceKind[];
	verify?: AcceptanceVerification[];
	stopRules?: string[];
}

export interface ResolvedAcceptance {
	level: AcceptanceLevel;
	explicit: boolean;
	criteria: AcceptanceCriterion[];
	evidence: EvidenceKind[];
	verify: AcceptanceVerification[];
	stopRules?: string[];
}

export interface AcceptanceReport {
	criteriaSatisfied: Array<{
		id: string;
		status: "satisfied" | "failed" | "partial";
		evidence?: string;
	}>;
	changedFiles?: string[];
	commandsRun?: Array<{
		command: string;
		result: "passed" | "failed" | "skipped";
		summary?: string;
	}>;
	residualRisks?: string[];
	noStagedFiles?: boolean;
}

export interface AcceptanceLedger {
	status: "passed" | "failed" | "timeout" | "not-required";
	report?: AcceptanceReport;
	config?: ResolvedAcceptance;
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

export interface AcceptanceEvaluation {
	status: "passed" | "failed" | "timeout";
	ledger: {
		status: string;
		report?: AcceptanceReport;
		verification?: string[];
	};
}

const ACCEPTANCE_FENCE = "```\nacceptance-report";
const ACCEPTANCE_FENCE_NO_NEWLINE = "```acceptance-report";
const ACCEPTANCE_FENCE_END = "```";

const DEFAULT_EVIDENCE: EvidenceKind[] = [
	"changed-files",
	"commands-run",
	"validation-output",
];

export function resolveEffectiveAcceptance(params: {
	explicit?: AcceptanceConfig;
}): ResolvedAcceptance {
	const explicit = params.explicit;
	if (!explicit) {
		return {
			level: "none",
			explicit: false,
			criteria: [],
			evidence: [],
			verify: [],
			stopRules: [],
		};
	}

	const evidence = explicit.evidence ?? DEFAULT_EVIDENCE;
	const criteria = normalizeCriteria(explicit.criteria ?? [], evidence);
	const verify = explicit.verify ?? [];

	let level: AcceptanceLevel = "checked";
	if (verify.length > 0) level = "verified";

	return {
		level,
		explicit: true,
		criteria,
		evidence,
		verify,
		stopRules: explicit.stopRules ?? [],
	};
}

function normalizeCriteria(
	input: string[] | AcceptanceCriterion[],
	evidence?: EvidenceKind[],
): AcceptanceCriterion[] {
	if (input.length === 0) return [];
	const result: AcceptanceCriterion[] = [];
	for (let i = 0; i < input.length; i++) {
		const item = input[i];
		if (typeof item === "string") {
			result.push({
				id: `criterion-${i + 1}`,
				must: item,
				evidence: evidence ?? undefined,
				severity: "required",
			});
		} else {
			result.push({
				id: item.id || `criterion-${i + 1}`,
				must: item.must,
				evidence: item.evidence ?? evidence ?? undefined,
				severity: item.severity ?? "required",
			});
		}
	}
	return result;
}

export function shouldRunAcceptanceFinalization(
	resolved: ResolvedAcceptance,
): boolean {
	return resolved.level !== "none";
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
				new Promise<AcceptanceVerificationResult[]>(resolve => {
					const timeout = verification.timeoutMs ?? 30_000;
					const timeoutId = setTimeout(() => {
						resolve([
							{
								command: verification.command,
								result: "failed",
								summary: `Timeout after ${timeout}ms`,
							},
						]);
					}, timeout);
					if (options.signal?.aborted) {
						clearTimeout(timeoutId);
						resolve([
							{
								command: verification.command,
								result: "failed",
								summary: "Aborted",
							},
						]);
						return;
					}

					execFileAsync("bash", ["-c", verification.command], {
						cwd: verification.cwd ?? options.cwd,
						timeout,
						maxBuffer: 1024 * 1024,
					}).then(
						(output: { stdout?: string; stderr?: string }) => {
							clearTimeout(timeoutId);
							resolve([
								{
									command: verification.command,
									result: "passed",
									summary: (output.stdout ?? "").trim().slice(0, 500),
								},
							]);
						},
						(error: NodeJS.ErrnoException) => {
							clearTimeout(timeoutId);
							resolve([
								verification.allowFailure
									? {
											command: verification.command,
											result: "passed",
											summary: `Non-zero exit ${error.code ?? "unknown"} (allowed)`,
										}
									: {
											command: verification.command,
											result: "failed",
											summary: error.message.slice(0, 500),
										},
							]);
						},
					);
				}),
		),
	).then(results => results.flat());
}

export function formatAcceptancePrompt(resolved: ResolvedAcceptance): string {
	if (resolved.level === "none") return "";

	const lines: string[] = [];
	lines.push("# Acceptance Contract");
	lines.push("");
	if (resolved.criteria.length > 0) {
		lines.push("## Criteria");
		lines.push("");
		for (const c of resolved.criteria) {
			lines.push(`- **${c.id}** (${c.severity}): ${c.must}`);
			if (c.evidence?.length) {
				lines.push(`  Evidence: ${c.evidence.join(", ")}`);
			}
		}
		lines.push("");
	}
	if (resolved.verify.length > 0) {
		lines.push("## Verification Commands");
		lines.push("");
		for (const v of resolved.verify) {
			lines.push(`- \`${v.command}\` (id: ${v.id})`);
			if (v.cwd) lines.push(`  cwd: \`${v.cwd}\``);
		}
		lines.push("");
	}
	lines.push(
		"Finish by producing a JSON report inside an acceptance-report fence:",
	);
	lines.push("");
	lines.push("```acceptance-report");
	lines.push("{");
	lines.push(
		'  "criteriaSatisfied": [{ "id": "criterion-1", "status": "satisfied", "evidence": "..." }],',
	);
	lines.push('  "changedFiles": ["file.ts"],');
	lines.push(
		'  "commandsRun": [{ "command": "npm test", "result": "passed", "summary": "all pass" }],',
	);
	lines.push('  "residualRisks": []');
	lines.push("}");
	lines.push("```");
	lines.push("");
	return lines.join("\n");
}

export function parseAcceptanceReport(output: string): {
	report?: AcceptanceReport;
	error?: string;
} {
	let fenceStart = output.indexOf(ACCEPTANCE_FENCE);
	if (fenceStart === -1) {
		fenceStart = output.indexOf(ACCEPTANCE_FENCE_NO_NEWLINE);
		if (fenceStart === -1) {
			return { error: "No acceptance report fence found" };
		}
	}
	const afterStart = fenceStart + ACCEPTANCE_FENCE.length;
	const fenceEnd = output.indexOf(ACCEPTANCE_FENCE_END, afterStart);
	if (fenceEnd === -1) {
		return { error: "Missing closing fence" };
	}
	const jsonStr = output.slice(afterStart, fenceEnd).trim();
	try {
		const report = JSON.parse(jsonStr) as AcceptanceReport;
		return { report };
	} catch (_e: unknown) {
		return { error: "Malformed acceptance report JSON" };
	}
}

export function evaluateAcceptanceReport(
	finalText: string,
	resolved: ResolvedAcceptance,
	verificationResults: AcceptanceVerificationResult[],
): AcceptanceEvaluation {
	const parsed = parseAcceptanceReport(finalText);
	if (!parsed.report && !parsed.error) {
		return {
			status: "failed",
			ledger: { status: "failed", verification: [] },
		};
	}

	const verificationSummary = verificationResults.map(
		verification =>
			`[${verification.result.toUpperCase()}] ${verification.command}${verification.summary ? ` → ${verification.summary.slice(0, 100)}` : ""}`,
	);
	const report = parsed.report;
	const criteriaResults = resolved.criteria.map(criterion => ({
		id: criterion.id,
		status: (report?.criteriaSatisfied?.some(
			item =>
				item.id === criterion.id &&
				(item.status === "satisfied" ||
					(criterion.severity === "recommended" && item.status === "partial")),
		)
			? "satisfied"
			: "failed") as "satisfied" | "failed",
		evidence: criterion.must,
	}));
	const allCriteriaPass = criteriaResults.every(
		criterion => criterion.status === "satisfied",
	);
	const allVerificationsPass = verificationResults.every(
		verification => verification.result === "passed",
	);

	return {
		status: allCriteriaPass && allVerificationsPass ? "passed" : "failed",
		ledger: {
			status: allCriteriaPass && allVerificationsPass ? "passed" : "failed",
			report: {
				...report,
				criteriaSatisfied: criteriaResults,
			},
			verification:
				verificationSummary.length > 0 ? verificationSummary : undefined,
		},
	};
}

export function stripAcceptanceReport(output: string): string {
	let fenceStart = output.indexOf(ACCEPTANCE_FENCE);
	if (fenceStart === -1) {
		fenceStart = output.indexOf(ACCEPTANCE_FENCE_NO_NEWLINE);
		if (fenceStart === -1) return output;
	}
	const afterStart = fenceStart + ACCEPTANCE_FENCE.length;
	const fenceEnd = output.indexOf(ACCEPTANCE_FENCE_END, afterStart);
	if (fenceEnd === -1) return output;
	return (
		output.slice(0, fenceStart) +
		output.slice(fenceEnd + ACCEPTANCE_FENCE_END.length)
	);
}

export function validateAcceptanceInput(config: AcceptanceConfig): string[] {
	const errors: string[] = [];
	const validKeys = new Set(["criteria", "evidence", "verify", "stopRules"]);
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
