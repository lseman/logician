// ── Message handling ──────────────────────────────────────────────────────────────
// Message creation and chat format conversion.

import type { Message, MessageRole } from "./types.ts";

export function createUserMessage(content: string): Message {
    return { role: "user", content, timestamp: Date.now() };
}

export function createSystemMessage(content: string): Message {
    return { role: "system", content, timestamp: Date.now() };
}

export function createAssistantMessage(
    content: string,
    toolCalls?: Array<{ id: string; name: string; arguments: string }>,
): Message {
    return {
        role: "assistant",
        content: content || null,
        tool_calls: toolCalls,
        timestamp: Date.now(),
    };
}

export function createToolResultMessage(
    toolCallId: string,
    toolName: string,
    result: string,
    isError: boolean = false,
): Message {
    return {
        role: "tool",
        content: result,
        tool_call_id: toolCallId,
        name: toolName,
        timestamp: Date.now(),
    };
}

export function convertToChatFormat(
    messages: Message[],
): Record<string, unknown>[] {
    return messages.map((m) => {
        const obj: Record<string, unknown> = { role: m.role };
        if (m.content !== null && m.content !== undefined) {
            obj.content = m.content;
        }
        if (m.tool_call_id) obj.tool_call_id = m.tool_call_id;
        if (m.tool_calls?.length) {
            obj.tool_calls = m.tool_calls.map((tc) => ({
                id: tc.id,
                type: "function",
                function: {
                    name: tc.name,
                    arguments: tc.arguments,
                },
            }));
        }
        if (m.name) obj.name = m.name;
        return obj;
    });
}

export function estimateTokens(text: string): number {
    return Math.max(1, Math.ceil(text.length / 4));
}

export function estimateMessageTokens(messages: Message[]): number {
    return estimateTokens(JSON.stringify(messages));
}

export function estimateChatPayloadTokens(
    messages: Message[],
    tools?: Record<string, unknown>[],
): number {
    return estimateTokens(
        JSON.stringify({
            messages: convertToChatFormat(messages),
            tools: tools || [],
        }),
    );
}

export interface CompactionResult {
    messages: Message[];
    tokensBefore: number;
    tokensAfter: number;
    changed: boolean;
}

export function compactMessagesForContext(
    messages: Message[],
    options: {
        targetTokens?: number;
        keepRecentMessages?: number;
        maxSummaryChars?: number;
    } = {},
): CompactionResult {
    const tokensBefore = estimateMessageTokens(messages);
    const contentCompacted = messages.map((message) =>
        compactLargeMessageContent(message),
    );
    const contentTokens = estimateMessageTokens(contentCompacted);

    const keepRecentMessages = Math.max(2, options.keepRecentMessages || 8);
    const systemMessages = contentCompacted.filter(
        (message) => message.role === "system",
    );
    const nonSystem = contentCompacted.filter(
        (message) => message.role !== "system",
    );
    let tailStart = Math.max(0, nonSystem.length - keepRecentMessages);
    tailStart = adjustTailStartForToolPairs(nonSystem, tailStart);

    const older = nonSystem.slice(0, tailStart);
    const recent = nonSystem.slice(tailStart);
    if (!older.length) {
        return {
            messages: contentCompacted,
            tokensBefore,
            tokensAfter: contentTokens,
            changed: contentTokens < tokensBefore,
        };
    }

    const summary = summarizeMessages(older, options.maxSummaryChars || 12000);
    const compacted = [
        ...systemMessages,
        createUserMessage(
            `<context-compaction reason="context_full">\n${summary}\n</context-compaction>`,
        ),
        ...recent,
    ];
    const tokensAfter = estimateMessageTokens(compacted);
    return {
        messages: compacted,
        tokensBefore,
        tokensAfter,
        changed: tokensAfter < tokensBefore,
    };
}

function compactLargeMessageContent(message: Message): Message {
    if (typeof message.content !== "string") return message;
    const maxChars =
        message.role === "tool"
            ? 6000
            : message.role === "assistant"
              ? 10000
              : 14000;
    if (message.content.length <= maxChars) return message;
    return {
        ...message,
        content: truncateMiddle(message.content, maxChars),
    };
}

function adjustTailStartForToolPairs(
    messages: Message[],
    start: number,
): number {
    let adjusted = start;
    while (adjusted > 0 && messages[adjusted]?.role === "tool") {
        adjusted--;
    }
    return adjusted;
}

function summarizeMessages(messages: Message[], maxChars: number): string {
    const lines = [
        "The earlier conversation was compacted automatically because the model context was full.",
        "Important prior messages and tool results are summarized below.",
        "",
    ];

    for (const message of messages) {
        if (message.role === "system") continue;
        const label =
            message.role === "tool"
                ? `tool:${message.name || "unknown"}`
                : message.role;
        const text = messageToText(message);
        if (!text) continue;
        lines.push(`## ${label}`);
        lines.push(truncateMiddle(text, 1800));
        if (message.tool_calls?.length) {
            lines.push(
                `Tool calls: ${message.tool_calls
                    .map(
                        (tool) =>
                            `${tool.name}(${truncateMiddle(tool.arguments || "{}", 500)})`,
                    )
                    .join("; ")}`,
            );
        }
        lines.push("");
    }

    return truncateMiddle(lines.join("\n"), maxChars);
}

function messageToText(message: Message): string {
    if (typeof message.content === "string") return message.content.trim();
    if (message.content === null || message.content === undefined) return "";
    return String(message.content).trim();
}

function truncateMiddle(text: string, maxChars: number): string {
    if (text.length <= maxChars) return text;
    const half = Math.max(1, Math.floor((maxChars - 32) / 2));
    return `${text.slice(0, half)}\n...[compacted ${text.length - half * 2} chars]...\n${text.slice(-half)}`;
}
