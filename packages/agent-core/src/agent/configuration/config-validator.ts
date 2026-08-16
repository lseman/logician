// Config schema validation and defaults.

import type {
	AgentConfig,
	QueueMode,
	ThinkingLevel,
} from "../types.ts";
import {
	INFERENCE_MODE_ORDER,
	isValidInferenceMode,
	QUEUE_MODES,
	THINKING_LEVELS,
	VALID_TOOL_EXECUTION,
} from "../types.ts";

interface ValidationError {
	field: string;
	message: string;
}

export function validateConfig(config: AgentConfig): ValidationError[] {
	const errors: ValidationError[] = [];

	if (!config.baseUrl) {
		errors.push({ field: "baseUrl", message: "required" });
	}
	if (!config.model) {
		errors.push({ field: "model", message: "required" });
	}

	if (
		config.temperature !== undefined &&
		(config.temperature < 0 || config.temperature > 2)
	) {
		errors.push({ field: "temperature", message: "must be 0-2" });
	}

	if (config.maxTokens !== undefined && config.maxTokens <= 0) {
		errors.push({ field: "maxTokens", message: "must be > 0" });
	}

	if (
		config.contextWindowTokens !== undefined &&
		config.contextWindowTokens <= 0
	) {
		errors.push({ field: "contextWindowTokens", message: "must be > 0" });
	}

	if (
		config.thinkingLevel &&
		!THINKING_LEVELS.includes(config.thinkingLevel)
	) {
		errors.push({
			field: "thinkingLevel",
			message: `must be one of: ${THINKING_LEVELS.join(", ")}`,
		});
	}

	if (
		config.steeringQueueMode &&
		!QUEUE_MODES.includes(config.steeringQueueMode)
	) {
		errors.push({
			field: "steeringQueueMode",
			message: `must be one of: ${QUEUE_MODES.join(", ")}`,
		});
	}

	if (
		config.followUpQueueMode &&
		!QUEUE_MODES.includes(config.followUpQueueMode)
	) {
		errors.push({
			field: "followUpQueueMode",
			message: `must be one of: ${QUEUE_MODES.join(", ")}`,
		});
	}

	if (
		config.toolExecution &&
		!VALID_TOOL_EXECUTION.includes(config.toolExecution)
	) {
		errors.push({
			field: "toolExecution",
			message: `must be one of: ${VALID_TOOL_EXECUTION.join(", ")}`,
		});
	}

	if (
		config.inferenceMode &&
		!isValidInferenceMode(config.inferenceMode)
	) {
		errors.push({
			field: "inferenceMode",
			message: `must be one of: ${INFERENCE_MODE_ORDER.join(", ")}`,
		});
	}

	if (config.maxIterations !== undefined && config.maxIterations <= 0) {
		errors.push({ field: "maxIterations", message: "must be > 0" });
	}

	if (config.maxRetries !== undefined && config.maxRetries < 0) {
		errors.push({ field: "maxRetries", message: "must be >= 0" });
	}

	if (config.retryBaseDelayMs !== undefined && config.retryBaseDelayMs < 0) {
		errors.push({ field: "retryBaseDelayMs", message: "must be >= 0" });
	}

	if (config.turnTimeoutMs !== undefined && config.turnTimeoutMs <= 0) {
		errors.push({ field: "turnTimeoutMs", message: "must be > 0" });
	}

	if (config.cacheSize !== undefined && config.cacheSize <= 0) {
		errors.push({ field: "cacheSize", message: "must be > 0" });
	}

	if (config.cacheTtlMs !== undefined && config.cacheTtlMs <= 0) {
		errors.push({ field: "cacheTtlMs", message: "must be > 0" });
	}

	if (
		config.loopDetectionWindow !== undefined &&
		config.loopDetectionWindow <= 0
	) {
		errors.push({ field: "loopDetectionWindow", message: "must be > 0" });
	}

	if (
		config.degenerateLoopThreshold !== undefined &&
		config.degenerateLoopThreshold < 0
	) {
		errors.push({ field: "degenerateLoopThreshold", message: "must be >= 0" });
	}

	if (
		config.stagnationThreshold !== undefined &&
		config.stagnationThreshold < 0
	) {
		errors.push({ field: "stagnationThreshold", message: "must be >= 0" });
	}

	if (
		config.proactiveCompactionFraction !== undefined &&
		(config.proactiveCompactionFraction <= 0 ||
			config.proactiveCompactionFraction > 1)
	) {
		errors.push({
			field: "proactiveCompactionFraction",
			message: "must be 0-1",
		});
	}

	return errors;
}

export function throwOnValidationErrors(errors: ValidationError[]): void {
	if (errors.length === 0) return;
	const msg = errors.map(e => `${e.field}: ${e.message}`).join("; ");
	throw new Error(`Invalid config: ${msg}`);
}
