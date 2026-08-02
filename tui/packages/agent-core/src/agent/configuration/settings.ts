// ── Settings command handler ─────────────────────────────────────────────────
// /settings — view and modify harness configuration at runtime.

/** Readable string for a boolean setting. */
function boolStr(v: boolean | undefined): string {
	return v ? "on" : "off";
}

/** Format a single setting line for the settings view. */
function fmtSetting(label: string, value: string): string {
	return `  ${label.padEnd(28)} ${value}`;
}

/** Build a human-readable settings snapshot. */
export function buildSettingsSnapshot(opts: {
	currentModel: string;
	models: string[];
	temperature: number;
	maxTokens: number | undefined;
	maxIterations: number;
	contextWindowTokens: number | undefined;
	thinkingLevel: string;
	inferenceMode: string;
	proactiveCompactionEnabled: boolean | undefined;
	proactiveCompactionFraction: number | undefined;
	guardsEnabled: boolean | undefined;
	continuationEnabled: boolean | undefined;
	budgetStopEnabled: boolean | undefined;
	toolExecution: string | undefined;
	steeringQueueMode: string | undefined;
	followUpQueueMode: string | undefined;
	autoRetryEnabled: boolean | undefined;
	maxRetries: number | undefined;
	retryBaseDelayMs: number | undefined;
	turnTimeoutMs: number | undefined;
	acceptAllPermissions: boolean;
	rtkProxyEnabled: boolean | undefined;
}): string {
	const lines: string[] = ["── Runtime Settings ──"];

	lines.push(fmtSetting("Model", opts.currentModel));
	if (opts.models.length > 0) {
		lines.push(fmtSetting("  Models (cycle)", opts.models.join(", ")));
	}

	lines.push(fmtSetting("Temperature", opts.temperature.toFixed(2)));
	lines.push(fmtSetting("Max Tokens", opts.maxTokens?.toString() ?? "unset"));
	lines.push(fmtSetting("Max Iterations", opts.maxIterations.toString()));
	lines.push(fmtSetting("Context Window (tokens)", opts.contextWindowTokens?.toString() ?? "unset"));

	lines.push("");
	lines.push("── Reasoning ──");
	lines.push(fmtSetting("Thinking Level", opts.thinkingLevel));
	lines.push(fmtSetting("Inference Mode", opts.inferenceMode));

	lines.push("");
	lines.push("── Guardrails ──");
	lines.push(fmtSetting("Guards", boolStr(opts.guardsEnabled)));
	lines.push(fmtSetting("Continuation", boolStr(opts.continuationEnabled)));
	lines.push(fmtSetting("Budget Stop", boolStr(opts.budgetStopEnabled)));

	lines.push("");
	lines.push("── Compaction ──");
	lines.push(fmtSetting("Proactive Compaction", boolStr(opts.proactiveCompactionEnabled)));
	if (opts.proactiveCompactionFraction !== undefined) {
		lines.push(fmtSetting("  Activation threshold", `${(opts.proactiveCompactionFraction * 100).toFixed(0)}%`));
	}

	lines.push("");
	lines.push("── Execution ──");
	lines.push(fmtSetting("Tool Execution", opts.toolExecution ?? "sequential"));
	lines.push(fmtSetting("Steering Queue Mode", opts.steeringQueueMode ?? "one-at-a-time"));
	lines.push(fmtSetting("Follow-up Queue Mode", opts.followUpQueueMode ?? "one-at-a-time"));
	lines.push(fmtSetting("Auto Retry", boolStr(opts.autoRetryEnabled)));
	if (opts.maxRetries !== undefined) {
		lines.push(fmtSetting("  Max Retries", opts.maxRetries.toString()));
	}
	if (opts.retryBaseDelayMs !== undefined) {
		lines.push(fmtSetting("  Base Retry Delay", `${opts.retryBaseDelayMs}ms`));
	}
	if (opts.turnTimeoutMs !== undefined) {
		lines.push(fmtSetting("Turn Timeout", `${opts.turnTimeoutMs}ms`));
	}

	lines.push("");
	lines.push("── Permissions ──");
	lines.push(fmtSetting("Default", opts.acceptAllPermissions ? "Accept All" : "Ask"));

	lines.push("");
	lines.push("── RTK Proxy ──");
	lines.push(fmtSetting("RTK CLI Proxy", boolStr(opts.rtkProxyEnabled)));
	if (opts.rtkProxyEnabled) {
		lines.push("  All bash commands prefixed with `rtk` for 60-90% output compression.");
	}

	lines.push("");
	lines.push("── Quick Changes ──");
	lines.push("  /settings thinking <level>     → off|minimal|low|medium|high|xhigh");
	lines.push("  /settings model <name>         → set a specific model");
	lines.push("  /settings model-cycle          → cycle to next model");
	lines.push("  /settings temp <n>             → set temperature (0.0–2.0)");
	lines.push("  /settings max-tokens <n>       → set max tokens");
	lines.push("  /settings max-iterations <n>   → set max iterations per turn");
	lines.push("  /settings guards [on]          → toggle output guards");
	lines.push("  /settings compaction [on]      → toggle proactive compaction");
	lines.push("  /settings permissions <mode>   → acceptAll|acceptEdits|ask|plan");
	lines.push("  /settings inference-mode <m>   → thinking-general|thinking-coding|instruct-general|instruct-reasoning|instruct-coding|deterministic|creative|analytical");
	lines.push("  /rtk                           → toggle RTK CLI proxy on/off");

	return lines.join("\n");
}

/** Parse a settings subcommand and invoke the appropriate action. */
export type SettingsAction =
	| { type: "view"; snapshot: string }
	| { type: "change"; key: string; value: string }
	| { type: "cycle" }
	| { type: "error"; message: string };

export function parseSettingsCommand(
	args: string,
): SettingsAction {
	const trimmed = args.trim();

	// No args → view all
	if (!trimmed) {
		return { type: "view", snapshot: "" }; // filled in by caller
	}

	const parts = trimmed.split(/\s+/);
	const sub = parts[0].toLowerCase();

	switch (sub) {
		case "thinking": {
			const level = parts[1]?.trim().toLowerCase();
			const valid = ["off", "minimal", "low", "medium", "high", "xhigh"];
			if (!level) {
				return {
					type: "error",
					message: `Usage: /settings thinking <level>\n\nValid levels: ${valid.join(", ")}`,
				};
			}
			if (!valid.includes(level)) {
				return {
					type: "error",
					message: `Invalid level "${level}". Valid: ${valid.join(", ")}`,
				};
			}
			return { type: "change", key: "thinking_level", value: level };
		}

		case "model": {
			const model = parts[1]?.trim();
			if (!model) {
				return { type: "error", message: "Usage: /settings model <name>" };
			}
			return { type: "change", key: "model", value: model };
		}

		case "model-cycle":
		case "model_cycle":
			return { type: "cycle" };

		case "temp": {
			const n = parseFloat(parts[1]?.trim() ?? "");
			if (isNaN(n) || n < 0 || n > 2.0) {
				return {
					type: "error",
					message: `Invalid temperature "${parts[1]}". Must be 0.0–2.0`,
				};
			}
			return { type: "change", key: "temperature", value: n.toString() };
		}

		case "max-tokens":
		case "max_tokens": {
			const n = parseInt(parts[1]?.trim() ?? "", 10);
			if (isNaN(n) || n < 1) {
				return {
					type: "error",
					message: `Invalid value "${parts[1]}". Must be a positive integer`,
				};
			}
			return { type: "change", key: "max_tokens", value: n.toString() };
		}

		case "max-iterations":
		case "max_iterations": {
			const n = parseInt(parts[1]?.trim() ?? "", 10);
			if (isNaN(n) || n < 1) {
				return {
					type: "error",
					message: `Invalid value "${parts[1]}". Must be a positive integer`,
				};
			}
			return { type: "change", key: "max_iterations", value: n.toString() };
		}

		case "guards": {
			const state = parts[1]?.trim().toLowerCase();
			if (state === "on" || state === "off") {
				return { type: "change", key: "guards", value: state };
			}
			return {
				type: "error",
				message: "Usage: /settings guards [on|off]\n\nToggle output guards (context recovery, loop detection).",
			};
		}

		case "compaction": {
			const state = parts[1]?.trim().toLowerCase();
			if (state === "on" || state === "off") {
				return { type: "change", key: "compaction", value: state };
			}
			return {
				type: "error",
				message: "Usage: /settings compaction [on|off]\n\nToggle proactive compaction at 80% context window.",
			};
		}

		case "permissions": {
			const mode = parts[1]?.trim().toLowerCase();
			const valid = ["acceptall", "acceptedits", "ask", "plan"];
			if (!valid.includes(mode)) {
				return {
					type: "error",
					message: `Invalid mode "${mode}". Valid: acceptAll, acceptEdits, ask, plan`,
				};
			}
			return { type: "change", key: "permissions", value: mode };
		}

		case "inference-mode":
		case "inference_mode": {
			const mode = parts[1]?.trim().toLowerCase();
			const valid = [
				"thinking-general",
				"thinking-coding",
				"instruct-general",
				"instruct-reasoning",
			];
			if (!mode) {
				return {
					type: "error",
					message: `Usage: /settings inference-mode <mode>\n\nValid modes: ${valid.join(", ")}`,
				};
			}
			if (!valid.includes(mode)) {
				return {
					type: "error",
					message: `Invalid mode "${mode}". Valid: ${valid.join(", ")}`,
				};
			}
			return { type: "change", key: "inference_mode", value: mode };
		}

		default:
			return {
				type: "error",
				message: `Unknown subcommand: /settings ${sub}\n\nRun /settings for the full settings view.`,
			};
	}
}

export type RtkProxyAction = "toggle" | "error";

export function parseRtkCommand(
	args: string,
): RtkProxyAction {
	const trimmed = args.trim();
	if (trimmed === "") {
		return "toggle";
	}
	return "error";
}
