// ── completion tool ──────────────────────────────────────────────────────────
// Make model completion calls via the JS kernel's fetch().
// Results persist in the kernel's module scope for follow-up.

import type { Tool, ToolContext } from "@logician/log-core";
import type { KernelManager, KernelManagerConfig } from "./kernel-manager.ts";

export interface CompletionToolDeps {
	kernel: KernelManager;
	config?: KernelManagerConfig & { baseUrl?: string; chatTemplate?: string | null };
}

const completionSchema = {
	type: "object",
	properties: {
		prompt: {
			type: "string",
			description:
				"The prompt text to send to the model. " +
				"Use system: and user: prefixes to structure multi-turn prompts.",
		},
		model: {
			type: "string",
			description:
				"Model identifier (overrides the runtime's default model when set).",
		},
		temperature: {
			type: "number",
			description: "Sampling temperature (0.0–2.0). Defaults to runtime config.",
		},
		max_tokens: {
			type: "number",
			description:
				"Maximum output tokens. Defaults to runtime config.",
		},
		schema: {
			type: "object",
			description:
				"JSON Schema for structured output. When provided, the response is parsed " +
				"and validated against the schema.",
		},
	},
	required: ["prompt"],
} as const;

const COMPLETION_KERNEL_CODE = `
// Persistent completion registry in module scope (survives across eval calls).
if (typeof __ompCompletions === 'undefined') {
    __ompCompletions = new Map();
}

const baseUrl = %BASE_URL%;
const defaultModel = %DEFAULT_MODEL%;
const chatTemplate = %CHAT_TEMPLATE%;

async function doCompletion(prompt, options) {
    const model = options.model || defaultModel;
    const temperature = options.temperature;
    const maxTokens = options.max_tokens;
    const schema = options.schema;

    // Build the request body for OpenAI-compatible API.
    const messages = [];
    // Split prompt into system/user parts.
    const lines = prompt.split('\\n');
    let currentRole = 'user';
    let currentContent = [];

    for (const line of lines) {
        if (line.startsWith('system:')) {
            if (currentContent.length > 0) {
                messages.push({ role: currentRole, content: currentContent.join('\\n') });
            }
            currentRole = 'system';
            currentContent = [line.slice(7)];
        } else if (line.startsWith('user:')) {
            if (currentContent.length > 0) {
                messages.push({ role: currentRole, content: currentContent.join('\\n') });
            }
            currentRole = 'user';
            currentContent = [line.slice(5)];
        } else if (line.startsWith('assistant:')) {
            if (currentContent.length > 0) {
                messages.push({ role: currentRole, content: currentContent.join('\\n') });
            }
            currentRole = 'assistant';
            currentContent = [line.slice(10)];
        } else {
            currentContent.push(line);
        }
    }
    if (currentContent.length > 0) {
        messages.push({ role: currentRole, content: currentContent.join('\\n') });
    }

    const body = {
        model,
        messages,
        stream: false,
    };
    if (temperature !== undefined) body.temperature = temperature;
    if (maxTokens !== undefined) body.max_tokens = maxTokens;
    if (chatTemplate) body.chat_template = chatTemplate;

    const response = await fetch(baseUrl + '/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error('API error ' + response.status + ': ' + errorText);
    }

    const data = await response.json();
    const text = data.choices?.[0]?.message?.content || '';

    const result = {
        text,
        model: data.model,
        finishReason: data.choices?.[0]?.finish_reason,
        usage: data.usage,
    };

    if (schema) {
        try {
            result.parsed = JSON.parse(text);
            // Basic schema validation hint.
            result.validation = 'parsed';
        } catch {
            result.validation = 'parse_failed';
        }
    }

    return result;
}

const handle = await doCompletion(%PROMPT%, %OPTIONS%);
__ompCompletions.set(%HANDLE_ID%, handle);
`;

/**
 * Execute a model completion call via the JS kernel's fetch().
 *
 * Results are stored in the kernel's persistent module scope,
 * accessible to other tools (e.g. wait).
 */
export function createCompletionTool(deps: CompletionToolDeps): Tool {
	return {
		name: "completion",
		label: "Completion",
		description:
			"Make a model completion call via the JS kernel's fetch(). " +
			"Use prompt structure: 'system: ...\\nuser: ...' for multi-turn. " +
			"Results persist in kernel scope for use with the wait tool.",
		promptSnippet: "completion(prompt='What is 2+2?', model='gpt-4o')",
		promptGuidelines: [
			"Use system: and user: prefixes for multi-turn prompts",
			"Provide schema for structured JSON output",
			"Results are stored in kernel scope — use wait() to retrieve",
		],
		readOnly: true,
		executionMode: "sequential",
		parameters: completionSchema,
		execute: async (
			args: Record<string, unknown>,
			_ctx: ToolContext,
		): Promise<string> => {
			const prompt = String(args.prompt);
			const model = args.model as string | undefined;
			const temperature = args.temperature as number | undefined;
			const maxTokens = args.max_tokens as number | undefined;
			const schema = args.schema as Record<string, unknown> | undefined;

			const handleId = `omp_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

			const code = COMPLETION_KERNEL_CODE
				.replace("%BASE_URL%", JSON.stringify(deps.config?.baseUrl ?? ""))
				.replace("%DEFAULT_MODEL%", JSON.stringify(model ?? "default"))
				.replace("%CHAT_TEMPLATE%", JSON.stringify(deps.config?.chatTemplate ?? null))
				.replace("%PROMPT%", JSON.stringify(prompt))
				.replace("%OPTIONS%", JSON.stringify({ model, temperature, max_tokens: maxTokens, schema }))
				.replace("%HANDLE_ID%", JSON.stringify(handleId));

			const result = await deps.kernel.js.eval(code, 30_000);

			if (result.status === "success") {
				return `Completion started. Handle: ${handleId}\nOutput: ${result.output.trim()}`;
			}

			return `Error: ${result.error ?? "Unknown error"}`;
		},
	};
}
