import type { LLMBackend } from "../provider/backend.ts";
import { convertToChatFormat } from "../provider/messages.ts";
import type { AgentEvent, Message } from "../types/index.ts";

export interface ReflectionConfig {
	enabled?: boolean;
	maxReflections?: number;
	prompt?: string;
}

export interface ReflectionResult {
	assessment: "complete" | "incomplete";
	reasoning: string;
	issues: string[];
	needsMoreWork: boolean;
	suggestedSteps: string[];
}

type EventSink = (event: AgentEvent) => Promise<void> | void;

const DEFAULT_REFLECTION_PROMPT = `
You have just completed a task. Before finalizing, perform a structured self-evaluation.

Review your work against these criteria:
1. **Completeness**: Did you fully address the task? Are there any loose ends?
2. **Correctness**: Are your changes/code/logic sound? Any bugs or mistakes?
3. **Edge cases**: Did you consider error handling, edge cases, or failure modes?
4. **Quality**: Is the output clean, well-structured, and production-ready?
5. **Next steps**: If incomplete, what specific steps are needed?

Respond with a JSON report in a reflection-report fence:
\`\`\`reflection-report
{
  "assessment": "complete" | "incomplete",
  "reasoning": "Brief explanation of your assessment",
  "issues": ["List any issues found, or empty array if none"],
  "needsMoreWork": boolean,
  "suggestedSteps": ["Steps needed if incomplete, or empty array"]
}
\`\`\`

If "assessment" is "complete" and "needsMoreWork" is false, the task is done.
If "assessment" is "incomplete" or "needsMoreWork" is true, you will be asked to continue.`;

export async function runReflection(
	currentMessages: Message[],
	backend: LLMBackend,
	reflectionConfig: ReflectionConfig,
	emit: EventSink,
	signal?: AbortSignal,
): Promise<{
	result: ReflectionResult;
	turnId: string;
	messages: Message[];
}> {
	const reflectionPrompt: Message = {
		role: "user",
		content: reflectionConfig.prompt ?? DEFAULT_REFLECTION_PROMPT,
	};
	const turnId = "reflection";
	await emit({ type: "reflection_start", turnId });
	const response = await backend.generate(
		convertToChatFormat([
			...currentMessages,
			reflectionPrompt,
		]) as unknown as Record<string, unknown>[],
		{
			tools: [],
			temperature: 0.1,
			maxTokens: 2048,
			signal,
		},
	);
	const reflectionContent = (response?.content as string) ?? "";
	const result = parseReflectionResult(reflectionContent);
	const assistantMessage: Message = {
		role: "assistant",
		content: reflectionContent,
		timestamp: Date.now(),
	};
	await emit({
		type: "reflection_end",
		turnId,
		assessment: result.assessment,
		needsMoreWork: result.needsMoreWork,
		issues: result.issues,
	});
	return {
		result,
		turnId,
		messages: [...currentMessages, reflectionPrompt, assistantMessage],
	};
}

function parseReflectionResult(content: string): ReflectionResult {
	const fenceStart = content.indexOf("```");
	const fenceEnd = content.indexOf("```", fenceStart + 3);
	if (fenceStart >= 0 && fenceEnd > fenceStart) {
		const json = content
			.slice(fenceStart + 3, fenceEnd)
			.trim()
			.replace(/^reflection-report\s*/, "");
		try {
			const parsed = JSON.parse(json) as Partial<ReflectionResult>;
			if (isReflectionResult(parsed)) return parsed;
		} catch {
			// Invalid structured reflection fails closed below.
		}
	}
	return {
		assessment: "incomplete",
		reasoning: content
			? content.slice(0, 200)
			: "No valid structured reflection was produced.",
		issues: ["Reflection output could not be validated."],
		needsMoreWork: true,
		suggestedSteps: [],
	};
}

function isReflectionResult(
	value: Partial<ReflectionResult>,
): value is ReflectionResult {
	return (
		(value.assessment === "complete" || value.assessment === "incomplete") &&
		typeof value.needsMoreWork === "boolean" &&
		typeof value.reasoning === "string" &&
		Array.isArray(value.issues) &&
		value.issues.every(issue => typeof issue === "string") &&
		Array.isArray(value.suggestedSteps) &&
		value.suggestedSteps.every(step => typeof step === "string")
	);
}
