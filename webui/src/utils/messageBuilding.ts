import { type ChatMessage, type ToolCallItem } from "../types";
import {
  isToolCallMarkup,
  extractToolCallsFromMarkup,
  extractToolCallsFromObject,
  parseToolArguments,
  pushUniqueToolCall,
  makeParsedToolCall,
  isValidJsonString,
  shouldReplaceToolResponse,
} from "./toolParsing";
import { normalizeThinkingLog } from "./messageFormatting";

function attachToolsToLastAssistant(messages: ChatMessage[], tools: ToolCallItem[]): ChatMessage[] {
  if (tools.length === 0) return messages;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].role !== "assistant") continue;
    return messages.map((message, messageIndex) =>
      messageIndex === index ? { ...message, tool_calls: tools } : message,
    );
  }
  return messages;
}

function attachThinkingToLastAssistant(messages: ChatMessage[], thinkingLog: string[]): ChatMessage[] {
  const normalizedThinking = normalizeThinkingLog(thinkingLog);
  if (normalizedThinking.length === 0) {
    return messages;
  }

  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].role !== "assistant") {
      continue;
    }
    return messages.map((message, messageIndex) =>
      messageIndex === index
        ? { ...message, thinking_log: normalizedThinking }
        : message,
    );
  }

  return messages;
}

function attachFallbackContentToLastAssistant(messages: ChatMessage[], fallbackContent = ""): ChatMessage[] {
  const trimmedFallback = fallbackContent.trim();
  if (!trimmedFallback) {
    return messages;
  }

  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].role !== "assistant") {
      continue;
    }
    const currentContent = String(messages[index].content || "").trim();
    if (currentContent && !isToolCallMarkup(currentContent)) {
      return messages;
    }
    return messages.map((message, messageIndex) =>
      messageIndex === index
        ? { ...message, content: trimmedFallback }
        : message,
    );
  }

  return messages;
}

function messageMatchKey(message: ChatMessage): string {
  if (message.id !== undefined) {
    return `id:${message.id}`;
  }
  return [
    message.role,
    message.timestamp || "",
    String(message.content || "").trim(),
  ].join("\n");
}

function preserveAssistantMetadata(
  messages: ChatMessage[],
  previousMessages: ChatMessage[],
): ChatMessage[] {
  if (previousMessages.length === 0) {
    return messages;
  }

  const previousAssistantMeta = new Map<
    string,
    Pick<ChatMessage, "thinking_log" | "tool_calls">
  >();

  for (const message of previousMessages) {
    if (message.role !== "assistant") {
      continue;
    }
    if (!message.thinking_log?.length && !message.tool_calls?.length) {
      continue;
    }
    previousAssistantMeta.set(messageMatchKey(message), {
      thinking_log: message.thinking_log,
      tool_calls: message.tool_calls,
    });
  }

  if (previousAssistantMeta.size === 0) {
    return messages;
  }

  return messages.map((message) => {
    if (message.role !== "assistant") {
      return message;
    }
    const preserved = previousAssistantMeta.get(messageMatchKey(message));
    if (!preserved) {
      return message;
    }
    return {
      ...message,
      thinking_log: message.thinking_log?.length ? message.thinking_log : preserved.thinking_log,
      tool_calls: message.tool_calls?.length ? message.tool_calls : preserved.tool_calls,
    };
  });
}

export function reconcileAssistantMessage(
  messages: ChatMessage[],
  thinkingLog: string[],
  tools: ToolCallItem[],
  fallbackContent?: string,
  previousMessages: ChatMessage[] = [],
): ChatMessage[] {
  const preservedMessages = preserveAssistantMetadata(messages, previousMessages);
  const withThinking = attachThinkingToLastAssistant(preservedMessages, thinkingLog);
  const withTools = attachToolsToLastAssistant(withThinking, tools);
  const withFallbackContent = attachFallbackContentToLastAssistant(withTools, fallbackContent);
  const hasAssistant = withFallbackContent.some((message) => message.role === "assistant");
  if (hasAssistant || !fallbackContent?.trim()) {
    return withFallbackContent;
  }
  return [
    ...withFallbackContent,
    {
      role: "assistant",
      content: fallbackContent.trim(),
      timestamp: new Date().toISOString(),
      thinking_log: normalizeThinkingLog(thinkingLog),
      tool_calls: tools.length > 0 ? tools : undefined,
    },
  ];
}

/**
 * Collapse every assistant→(tool→assistant)* chain into ONE display message.
 *
 * Strategy:
 *  1. Walk the chain, collecting streaming tool_calls (from any assistant in
 *     the chain — they carry full args + response from the streaming path).
 *  2. Collect DB tool messages separately.
 *  3. For each DB tool message, find the first streaming entry with the same
 *     tool name that has no response yet and patch it in.  If no streaming
 *     entry exists, add a minimal entry (only if the name is known).
 *  4. Use the last non-markup assistant content as the display text.
 */
export function buildDisplayMessages(messages: ChatMessage[]): ChatMessage[] {
  const result: ChatMessage[] = [];
  let i = 0;

  while (i < messages.length) {
    const msg = messages[i];

    if (msg.role !== "assistant") {
      if (msg.role !== "tool") result.push(msg);
      i++;
      continue;
    }

    // Pass 1: walk the whole chain, collecting streaming tool_calls + DB tool msgs
    const streamingCalls: ToolCallItem[] = [];
    const markupCalls: ToolCallItem[] = [];
    const dbToolMsgs: ChatMessage[] = [];
    let finalContent = "";
    let finalThinking: string[] | undefined;
    let j = i;
    let hasStructuredToolCalls = false;

    while (j < messages.length && (messages[j].role === "assistant" || messages[j].role === "tool")) {
      const cur = messages[j];
      if (cur.role === "assistant") {
        // Streaming tool_calls attached via attachToolsToLastAssistant
        if (cur.tool_calls?.length) {
          hasStructuredToolCalls = true;
          for (const tc of cur.tool_calls) {
            pushUniqueToolCall(streamingCalls, { ...tc });
          }
        }
        if (isToolCallMarkup(cur.content)) {
          for (const tc of extractToolCallsFromMarkup(cur.content)) {
            pushUniqueToolCall(markupCalls, tc);
          }
        }
        // Keep the last non-empty, non-markup content as the display text
        if (cur.content.trim() && !isToolCallMarkup(cur.content)) {
          finalContent = cur.content;
        }
        if (cur.thinking_log?.length) finalThinking = cur.thinking_log;
      } else {
        dbToolMsgs.push(cur);
      }
      j++;
    }

    // Pass 2: patch DB tool responses into streaming entries by name-order
    const toolCalls: ToolCallItem[] = hasStructuredToolCalls
      ? [...streamingCalls]
      : [...streamingCalls, ...markupCalls];
    for (const toolMsg of dbToolMsgs) {
      const name = String(toolMsg.name || "").trim();
      const response = String(toolMsg.content || "").trim();
      const callId = String(toolMsg.tool_call_id || "");

      // Prefer upgrading an existing tool card with the fuller DB response.
      const matchIdx = toolCalls.findIndex((tc) =>
        (!name || tc.tool === name) && shouldReplaceToolResponse(tc.response, response),
      );
      if (matchIdx >= 0) {
        toolCalls[matchIdx] = { ...toolCalls[matchIdx], response };
      } else if (!toolCalls.some((tc) => tc.tool === name)) {
        // No streaming entry at all — add from DB (args unknown)
        if (name) {
          toolCalls.push({ id: callId || `db-${j}`, tool: name, response });
        }
      }
      // If streaming already has a complete entry for this name, skip (no duplicate)
    }

    const finalToolCalls = toolCalls.filter((tc) => tc.tool.trim());
    result.push({
      ...msg,
      content: finalContent,
      thinking_log: finalThinking,
      tool_calls: finalToolCalls.length > 0 ? finalToolCalls : undefined,
    });
    i = j;
  }

  return result;
}
