import { useState } from "react";
import {
  DiffViewer,
  JsonViewer,
  isUnifiedDiffText,
  tryParseJson,
} from "./MarkdownMessage";

function prettyJsonLikeString(value: string): string {
  let depth = 0;
  let inString = false;
  let escaping = false;
  let out = "";

  const indent = () => "  ".repeat(Math.max(depth, 0));

  for (const char of value) {
    if (escaping) {
      out += char;
      escaping = false;
      continue;
    }

    if (char === "\\") {
      out += char;
      escaping = true;
      continue;
    }

    if (char === "\"") {
      inString = !inString;
      out += char;
      continue;
    }

    if (inString) {
      out += char;
      continue;
    }

    if (char === "{" || char === "[") {
      depth += 1;
      out += `${char}\n${indent()}`;
      continue;
    }

    if (char === "}" || char === "]") {
      depth -= 1;
      out += `\n${indent()}${char}`;
      continue;
    }

    if (char === ",") {
      out += `${char}\n${indent()}`;
      continue;
    }

    out += char;
  }

  return out;
}

function prettyJson(value: unknown): string {
  if (typeof value === "string") {
    try {
      return JSON.stringify(JSON.parse(value), null, 2);
    } catch {
      const trimmed = value.trim();
      if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
        return prettyJsonLikeString(trimmed);
      }
      return value;
    }
  }
  return JSON.stringify(value, null, 2);
}

export type ToolCallItem = {
  id: string;
  tool: string;
  args?: Record<string, unknown>;
  response?: string;
  status?: "running" | "ok" | "error";
  duration_ms?: number;
  cache_hit?: boolean;
};

function fileOperationPayload(
  toolName: string,
  payload: unknown,
): { summary: string[]; diff?: string; raw: unknown } | null {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return null;
  }
  const record = payload as Record<string, unknown>;
  const normalizedTool = toolName.trim();
  const isFileTool = new Set([
    "read_file",
    "write_file",
    "edit_file",
    "apply_edit_block",
    "smart_edit",
    "edit_file_libcst",
    "replace_function_body",
    "replace_docstring",
    "replace_decorators",
    "replace_argument",
    "insert_after_function",
    "delete_function",
  ]).has(normalizedTool);
  const diff = typeof record.diff === "string"
    ? record.diff
    : typeof record.preview === "string"
      ? record.preview
      : undefined;
  if (!isFileTool && !diff) {
    return null;
  }

  const summary: string[] = [];
  if (typeof record.path === "string" && record.path.trim()) {
    summary.push(record.path.trim());
  }
  if (typeof record.status === "string" && record.status.trim()) {
    summary.push(record.status.trim());
  }
  if (typeof record.matches_replaced === "number") {
    summary.push(`${record.matches_replaced} matches`);
  }
  if (typeof record.blocks_applied === "number") {
    summary.push(`${record.blocks_applied} blocks`);
  }
  if (typeof record.edits_applied === "number") {
    summary.push(`${record.edits_applied} edits`);
  }
  if (typeof record.newline === "string" && record.newline.trim()) {
    summary.push(record.newline.trim());
  }
  return { summary, diff, raw: payload };
}

function runtimeOperationPayload(
  toolName: string,
  payload: unknown,
): { summary: string[]; details?: string[]; raw: unknown } | null {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return null;
  }
  const record = payload as Record<string, unknown>;
  const normalizedTool = toolName.trim();
  const runtimeTools = new Set([
    "set_venv",
    "set_working_directory",
    "install_packages",
    "show_coding_config",
    "start_background_process",
    "send_input_to_process",
    "get_process_output",
    "kill_process",
    "list_processes",
  ]);
  if (!runtimeTools.has(normalizedTool)) {
    return null;
  }

  const summary: string[] = [];
  const details: string[] = [];
  for (const key of ["status", "name", "cwd", "venv_path", "python", "python_bin"]) {
    if (typeof record[key] === "string" && record[key]?.trim()) {
      summary.push(String(record[key]).trim());
    }
  }
  if (typeof record.pid === "number") {
    summary.push(`pid ${record.pid}`);
  }
  if (typeof record.running === "boolean") {
    summary.push(record.running ? "running" : "stopped");
  }
  if (typeof record.exit_code === "number") {
    summary.push(`exit ${record.exit_code}`);
  }
  if (typeof record.bytes_written === "number") {
    summary.push(`${record.bytes_written} bytes`);
  }
  if (Array.isArray(record.processes)) {
    summary.push(`${record.processes.length} tracked process${record.processes.length === 1 ? "" : "es"}`);
  }
  if (typeof record.message === "string" && record.message.trim()) {
    details.push(record.message.trim());
  }
  if (typeof record.output === "string" && record.output.trim()) {
    details.push(compactPreview(record.output, 220));
  }
  if (typeof record.command_preview === "string" && record.command_preview.trim()) {
    details.push(compactPreview(record.command_preview, 220));
  }
  return { summary, details, raw: payload };
}

function compactPreview(value: string | undefined, limit = 180): string {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (!text) {
    return "";
  }
  if (text.length <= limit) {
    return text;
  }
  return `${text.slice(0, limit).trimEnd()}...`;
}

function SectionToggle({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="tool-section">
      <button
        type="button"
        className="tool-section-toggle"
        onClick={() => setOpen((v) => !v)}
      >
        <span className={`thinking-caret ${open ? "is-open" : ""}`}>▸</span>
        <span className="tool-section-label">{label}</span>
      </button>
      {open ? <div className="tool-section-body">{children}</div> : null}
    </div>
  );
}

export function ToolCallCard({
  item,
  done = false,
}: {
  item: ToolCallItem;
  done?: boolean;
}) {
  const hasArgs = Boolean(item.args && Object.keys(item.args).length > 0);
  const hasResponse = Boolean(item.response);
  const argsText = hasArgs ? prettyJson(item.args) : "";
  const parsedArgs = tryParseJson(argsText);
  const state = item.status === "error"
    ? "error"
    : done || item.status === "ok" || hasResponse
      ? "done"
      : "live";
  const isDone = state !== "live";
  const toolLabel = item.tool.trim() || "tool";
  const preview = compactPreview(item.response);
  const parsedResponse = tryParseJson(String(item.response || ""));
  const responseText = String(item.response || "");
  const filePayload = fileOperationPayload(toolLabel, parsedResponse);
  const runtimePayload = runtimeOperationPayload(toolLabel, parsedResponse);
  const meta: string[] = [];

  if (item.cache_hit) {
    meta.push("cached");
  }
  if (item.duration_ms !== undefined) {
    meta.push(`${item.duration_ms}ms`);
  }
  if (state === "error") {
    meta.push("error");
  } else if (state === "live") {
    meta.push("running");
  }

  return (
    <div className={`tool-card tool-card-${state}`}>
      <div className="tool-card-header">
        <span className="tool-card-spinner" aria-hidden="true">
          {isDone ? (
            <svg
              width="11"
              height="11"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="20 6 9 17 4 12" />
            </svg>
          ) : (
            <svg className="animate-spin" width="11" height="11" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity="0.25" />
              <path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" opacity="0.75" />
            </svg>
          )}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="tool-card-name">{toolLabel}</span>
            <span className={`tool-card-state tool-card-state-${state}`}>
              {state === "live" ? "live" : state}
            </span>
          </div>
          {preview ? <div className="tool-card-preview">{preview}</div> : null}
        </div>
        {meta.length > 0 ? (
          <span className="tool-card-meta">{meta.join(" · ")}</span>
        ) : null}
      </div>

      {hasArgs ? (
        <SectionToggle label="tool_call">
          {parsedArgs !== null ? (
            <JsonViewer value={parsedArgs} raw={argsText} label="tool call" />
          ) : (
            <pre className="tool-card-args">{prettyJson(item.args)}</pre>
          )}
        </SectionToggle>
      ) : null}

      {hasResponse ? (
        <SectionToggle label="tool_response">
          {filePayload ? (
            <div className="space-y-3">
              {filePayload.summary.length > 0 ? (
                <div className="tool-card-preview">{filePayload.summary.join(" · ")}</div>
              ) : null}
              {filePayload.diff ? (
                <DiffViewer diff={filePayload.diff} label="file diff" />
              ) : (
                <JsonViewer value={filePayload.raw} raw={responseText} label="tool response" />
              )}
            </div>
          ) : runtimePayload ? (
            <div className="space-y-3">
              {runtimePayload.summary.length > 0 ? (
                <div className="tool-card-preview">{runtimePayload.summary.join(" · ")}</div>
              ) : null}
              {runtimePayload.details?.map((detail) => (
                <div key={detail} className="tool-card-preview">{detail}</div>
              ))}
              <JsonViewer value={runtimePayload.raw} raw={responseText} label="tool response" />
            </div>
          ) : parsedResponse !== null ? (
            <JsonViewer value={parsedResponse} raw={responseText} label="tool response" />
          ) : isUnifiedDiffText(responseText) ? (
            <DiffViewer diff={responseText} label="tool response diff" />
          ) : (
            <pre className="tool-card-args">{prettyJson(item.response)}</pre>
          )}
        </SectionToggle>
      ) : null}
    </div>
  );
}
