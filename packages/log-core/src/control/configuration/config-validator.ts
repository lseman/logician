import type { AgentConfig } from "../../system/types/types-config.ts";
import {
	INFERENCE_MODE_ORDER,
	isValidInferenceMode,
	QUEUE_MODES,
	THINKING_LEVELS,
	VALID_TOOL_EXECUTION,
} from "../../system/types/types-config.ts";

export interface AgentConfigValidationError {
	field: string;
	message: string;
}

export function validateAgentConfig(
	config: AgentConfig,
): AgentConfigValidationError[] {
	const errors: AgentConfigValidationError[] = [];
	const range = (
		field: keyof AgentConfig,
		value: number | undefined,
		minimum: number,
		maximum?: number,
		inclusive = false,
	) => {
		if (
			value !== undefined &&
			((inclusive ? value < minimum : value <= minimum) ||
				(maximum !== undefined && value > maximum))
		)
			errors.push({
				field: String(field),
				message:
					maximum === undefined
						? `must be ${inclusive ? ">=" : ">"} ${minimum}`
						: `must be ${minimum}-${maximum}`,
			});
	};
	if (!config.baseUrl) errors.push({ field: "baseUrl", message: "required" });
	if (!config.model) errors.push({ field: "model", message: "required" });
	if (
		config.temperature !== undefined &&
		(config.temperature < 0 || config.temperature > 2)
	)
		errors.push({ field: "temperature", message: "must be 0-2" });
	range("maxTokens", config.maxTokens, 0);
	range("contextWindowTokens", config.contextWindowTokens, 0);
	range("maxIterations", config.maxIterations, 0);
	range("maxRetries", config.maxRetries, 0, undefined, true);
	range("retryBaseDelayMs", config.retryBaseDelayMs, 0, undefined, true);
	range("turnTimeoutMs", config.turnTimeoutMs, 0);
	range("cacheSize", config.cacheSize, 0);
	range("cacheTtlMs", config.cacheTtlMs, 0);
	range("loopDetectionWindow", config.loopDetectionWindow, 0);
	range(
		"degenerateLoopThreshold",
		config.degenerateLoopThreshold,
		0,
		undefined,
		true,
	);
	range("stagnationThreshold", config.stagnationThreshold, 0, undefined, true);
	if (
		config.proactiveCompactionFraction !== undefined &&
		(config.proactiveCompactionFraction <= 0 ||
			config.proactiveCompactionFraction > 1)
	)
		errors.push({
			field: "proactiveCompactionFraction",
			message: "must be 0-1",
		});
	if (config.thinkingLevel && !THINKING_LEVELS.includes(config.thinkingLevel))
		errors.push({
			field: "thinkingLevel",
			message: `must be one of: ${THINKING_LEVELS.join(", ")}`,
		});
	for (const [field, value] of [
		["steeringQueueMode", config.steeringQueueMode],
		["followUpQueueMode", config.followUpQueueMode],
	] as const)
		if (value && !QUEUE_MODES.includes(value))
			errors.push({
				field,
				message: `must be one of: ${QUEUE_MODES.join(", ")}`,
			});
	if (
		config.toolExecution &&
		!VALID_TOOL_EXECUTION.includes(config.toolExecution)
	)
		errors.push({
			field: "toolExecution",
			message: `must be one of: ${VALID_TOOL_EXECUTION.join(", ")}`,
		});
	if (config.inferenceMode && !isValidInferenceMode(config.inferenceMode))
		errors.push({
			field: "inferenceMode",
			message: `must be one of: ${INFERENCE_MODE_ORDER.join(", ")}`,
		});
	return errors;
}

export function throwOnAgentConfigErrors(
	errors: readonly AgentConfigValidationError[],
): void {
	if (errors.length > 0)
		throw new Error(
			`Invalid config: ${errors.map(({ field, message }) => `${field}: ${message}`).join("; ")}`,
		);
}
