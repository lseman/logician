import { type ToolCallItem } from "../components/ToolCallCard";

function parseToolArguments(value: unknown): Record<string, unknown> | undefined {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  if (!trimmed.startsWith("{")) {
    return undefined;
  }
  try {
    const parsed = JSON.parse(trimmed) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    return undefined;
  }
  return undefined;
}

function toolCallSignature(item: Pick<ToolCallItem, "tool" | "args" | "response">): string {
  return `${item.tool}\n${JSON.stringify(item.args ?? {})}\n${item.response ?? ""}`;
}

function pushUniqueToolCall(target: ToolCallItem[], item: ToolCallItem | null) {
  if (!item) {
    return;
  }
  const signature = toolCallSignature(item);
  if (target.some((existing) => existing.id === item.id || toolCallSignature(existing) === signature)) {
    return;
  }
  target.push(item);
}

function hashString(value: string) {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash << 5) - hash + value.charCodeAt(index);
    hash |= 0;
  }
  return Math.abs(hash);
}

function makeParsedToolCall(
  seed: string,
  index: number,
  name: unknown,
  args?: unknown,
): ToolCallItem | null {
  const tool = String(name || "").trim();
  if (!tool) {
    return null;
  }
  return {
    id: `parsed-${hashString(`${seed}:${index}:${tool}`)}`,
    tool,
    args: parseToolArguments(args),
  };
}

function extractToolCallsFromObject(content: unknown, seed: string): ToolCallItem[] {
  if (!content || typeof content !== "object") {
    return [];
  }

  const results: ToolCallItem[] = [];
  const pushCall = (name: unknown, args?: unknown) => {
    pushUniqueToolCall(results, makeParsedToolCall(seed, results.length, name, args));
  };

  if (Array.isArray(content)) {
    content.forEach((item, index) => {
      for (const toolCall of extractToolCallsFromObject(item, `${seed}:${index}`)) {
        pushUniqueToolCall(results, toolCall);
      }
    });
    return results;
  }

  const record = content as Record<string, unknown>;

  if (Array.isArray(record.tool_calls)) {
    record.tool_calls.forEach((call, index) => {
      for (const toolCall of extractToolCallsFromObject(call, `${seed}:tool_calls:${index}`)) {
        pushUniqueToolCall(results, toolCall);
      }
    });
  }

  if (record.tool_call) {
    for (const toolCall of extractToolCallsFromObject(record.tool_call, `${seed}:tool_call`)) {
      pushUniqueToolCall(results, toolCall);
    }
  }

  if (record.function && typeof record.function === "object" && !Array.isArray(record.function)) {
    const fn = record.function as Record<string, unknown>;
    pushCall(fn.name, fn.arguments);
  }

  if (typeof record.name === "string") {
    pushCall(record.name, record.arguments ?? record.input);
  }

  if (Array.isArray(record.content)) {
    record.content.forEach((block, index) => {
      if (!block || typeof block !== "object" || Array.isArray(block)) {
        return;
      }
      const toolUse = block as Record<string, unknown>;
      if (toolUse.type === "tool_use") {
        pushCall(toolUse.name, toolUse.input);
      } else {
        for (const toolCall of extractToolCallsFromObject(toolUse, `${seed}:content:${index}`)) {
          pushUniqueToolCall(results, toolCall);
        }
      }
    });
  }

  return results;
}

/**
 * Return true if the content is raw tool-call markup that should be suppressed.
 * Uses pattern matching so it works even when JSON.parse fails due to unescaped
 * characters inside argument values (a common LLM output artefact).
 */
function isToolCallMarkup(content: string): boolean {
  const t = content.trim();
  if (!t) return false;
  // JSON-style: {"tool_call": ...} or {"name": ..., "arguments": ...}
  if (t.startsWith("{")) {
    if (/^\{\s*"tool_call"\s*:/.test(t)) return true;
    if (/^\{\s*"name"\s*:/.test(t) && /"arguments"\s*:/.test(t)) return true;
    // Also try full parse for well-formed JSON
    try {
      const p = JSON.parse(t) as Record<string, unknown>;
      if ("tool_call" in p || ("name" in p && "arguments" in p)) return true;
    } catch { /* fall through */ }
  }
  // YAML / TOON style: starts with tool_call: anywhere as a block
  if (/^tool_call\s*:/m.test(t)) return true;
  return false;
}

function extractToolCallsFromMarkup(content: string): ToolCallItem[] {
  const text = content.trim();
  if (!text) {
    return [];
  }

  if (text.startsWith("{") || text.startsWith("[")) {
    try {
      return extractToolCallsFromObject(JSON.parse(text) as unknown, text);
    } catch {
      // Fall through to regex-based parsing for malformed model output.
    }
  }

  const directCall = text.match(/^\[direct_tool_call\]\s+([^\s]+)(?:\s+(.*))?$/s);
  if (directCall) {
    const [, name, rawArgs = ""] = directCall;
    const toolCall = makeParsedToolCall(text, 0, name, rawArgs.trim());
    return toolCall ? [toolCall] : [];
  }

  if (!/^tool_call\s*:/m.test(text)) {
    return [];
  }

  const segments = text
    .split(/(?:^|\n)\s*tool_call\s*:/)
    .map((segment) => segment.trim())
    .filter(Boolean);

  return segments.flatMap((segment, index) => {
    const nameMatch = segment.match(/name\s*:\s*([^\n]+?)(?=\s+arguments\s*:|\s*$)/);
    const inlineArgsMatch = segment.match(/arguments\s*:\s*([A-Za-z_][\w-]*)\s*:\s*([^\n]+)/);
    const toolCall = makeParsedToolCall(
      `${text}:${index}`,
      index,
      nameMatch?.[1],
      inlineArgsMatch
        ? { [inlineArgsMatch[1]]: inlineArgsMatch[2].trim().replace(/^['"]|['"]$/g, "") }
        : undefined,
    );
    return toolCall ? [toolCall] : [];
  });
}

function isValidJsonString(value: string): boolean {
  const text = value.trim();
  if (!(text.startsWith("{") || text.startsWith("["))) {
    return false;
  }
  try {
    JSON.parse(text);
    return true;
  } catch {
    return false;
  }
}

function shouldReplaceToolResponse(existing: string | undefined, next: string): boolean {
  const incoming = next.trim();
  if (!incoming) {
    return false;
  }
  const current = String(existing || "").trim();
  if (!current) {
    return true;
  }
  const incomingValidJson = isValidJsonString(incoming);
  const currentValidJson = isValidJsonString(current);
  if (incomingValidJson && !currentValidJson) {
    return true;
  }
  if (incoming.length > current.length + 24) {
    return true;
  }
  return false;
}

export {
  isToolCallMarkup,
  parseToolArguments,
  toolCallSignature,
  pushUniqueToolCall,
  makeParsedToolCall,
  extractToolCallsFromObject,
  extractToolCallsFromMarkup,
  isValidJsonString,
  shouldReplaceToolResponse,
};
