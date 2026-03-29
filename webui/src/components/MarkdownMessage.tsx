import { Children, isValidElement, ReactNode, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue };

export function isUnifiedDiffText(text: string) {
  const normalized = String(text || "").trim();
  if (!normalized) {
    return false;
  }
  const lines = normalized.split(/\r?\n/);
  let signals = 0;
  if (lines.some((line) => line.startsWith("@@ "))) {
    signals += 1;
  }
  if (lines.some((line) => line.startsWith("--- ") || line.startsWith("+++ "))) {
    signals += 1;
  }
  if (lines.some((line) => /^diff --git |^index /.test(line))) {
    signals += 1;
  }
  if (lines.some((line) => line.startsWith("+") || line.startsWith("-"))) {
    signals += 1;
  }
  return signals >= 2;
}

export function isDiffKey(label?: string) {
  const key = String(label || "").trim().toLowerCase();
  return (
    key === "diff" ||
    key === "patch" ||
    key === "preview" ||
    key.endsWith("_diff") ||
    key.endsWith("_patch") ||
    key.endsWith("_preview")
  );
}

function flattenText(children: ReactNode): string {
  return Children.toArray(children)
    .map((child) => {
      if (typeof child === "string") {
        return child;
      }
      if (typeof child === "number") {
        return String(child);
      }
      if (isValidElement(child)) {
        return flattenText(child.props.children);
      }
      return "";
    })
    .join("");
}

function normalizeStreamingMarkdown(content: string, streaming: boolean) {
  if (!streaming) {
    return content;
  }
  let normalized = content;
  const fenceCount = (normalized.match(/```/g) ?? []).length;
  if (fenceCount % 2 === 1) {
    normalized += "\n```";
  }
  return normalized;
}

export function tryParseJson(text: string): JsonValue | null {
  const trimmed = String(text || "").trim();
  if (!trimmed || (!trimmed.startsWith("{") && !trimmed.startsWith("["))) {
    return null;
  }
  try {
    return JSON.parse(trimmed) as JsonValue;
  } catch {
    return null;
  }
}

function jsonPreview(value: JsonValue): string {
  if (Array.isArray(value)) {
    const preview = value.slice(0, 3).map((item) => jsonPreview(item)).join(", ");
    return `[${preview}${value.length > 3 ? ", ..." : ""}]`;
  }
  if (value && typeof value === "object") {
    const entries = Object.entries(value).slice(0, 3);
    const preview = entries
      .map(([key, item]) => `${key}: ${jsonPreview(item)}`)
      .join(", ");
    return `{${preview}${Object.keys(value).length > 3 ? ", ..." : ""}}`;
  }
  if (typeof value === "string") {
    return `"${value}"`;
  }
  return String(value);
}

function JsonLeaf({ value }: { value: JsonPrimitive }) {
  if (value === null) {
    return <span className="json-null">null</span>;
  }
  if (typeof value === "string") {
    return <span className="json-string">"{value}"</span>;
  }
  if (typeof value === "number") {
    return <span className="json-number">{value}</span>;
  }
  return <span className="json-boolean">{String(value)}</span>;
}

export function DiffViewer({
  diff,
  label,
}: {
  diff: string;
  label: string;
}) {
  const [copied, setCopied] = useState(false);
  const lines = useMemo(
    () => String(diff || "").replace(/\n$/, "").split(/\r?\n/),
    [diff],
  );

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(diff);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="diff-shell">
      <div className="diff-toolbar">
        <span className="diff-toolbar-label">{label}</span>
        <button className="code-block-copy" onClick={handleCopy} type="button">
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <div className="diff-body">
        {lines.map((line, index) => {
          const kind =
            line.startsWith("@@ ") ||
            line.startsWith("diff --git ") ||
            line.startsWith("index ") ||
            line.startsWith("--- ") ||
            line.startsWith("+++ ")
              ? "meta"
              : line.startsWith("+")
                ? "addition"
                : line.startsWith("-")
                  ? "deletion"
                  : "context";
          return (
            <div key={`${index}-${line.slice(0, 24)}`} className={`diff-line diff-line-${kind}`}>
              <span className="diff-line-number">{index + 1}</span>
              <code className="diff-line-text">{line || " "}</code>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function JsonNode({
  label,
  value,
  depth = 0,
}: {
  label?: string;
  value: JsonValue;
  depth?: number;
}) {
  const isObject = Boolean(value) && typeof value === "object";
  const [expanded, setExpanded] = useState(depth < 1);

  if (!isObject) {
    if (typeof value === "string" && isDiffKey(label) && isUnifiedDiffText(value)) {
      return (
        <div className="json-row space-y-2">
          {label ? (
            <div>
              <span className="json-key">"{label}"</span>
              <span className="json-punct">: </span>
            </div>
          ) : null}
          <DiffViewer diff={value} label={label || "diff"} />
        </div>
      );
    }
    return (
      <div className="json-row">
        {label ? <span className="json-key">"{label}"</span> : null}
        {label ? <span className="json-punct">: </span> : null}
        <JsonLeaf value={value as JsonPrimitive} />
      </div>
    );
  }

  const isArray = Array.isArray(value);
  const entries = isArray
    ? (value as JsonValue[]).map((item, index) => [String(index), item] as const)
    : Object.entries(value as Record<string, JsonValue>);
  const open = isArray ? "[" : "{";
  const close = isArray ? "]" : "}";

  return (
    <div className="json-node">
      <button
        type="button"
        className="json-toggle"
        onClick={() => setExpanded((current) => !current)}
      >
        <span className={`json-caret ${expanded ? "is-open" : ""}`}>▸</span>
        {label ? <span className="json-key">"{label}"</span> : <span className="json-root">root</span>}
        {label ? <span className="json-punct">: </span> : <span className="json-punct"> </span>}
        <span className="json-bracket">{open}</span>
        <span className="json-meta">
          {isArray ? `${entries.length} items` : `${entries.length} keys`}
        </span>
        {!expanded ? (
          <>
            <span className="json-punct"> </span>
            <span className="json-preview">{jsonPreview(value)}</span>
          </>
        ) : null}
      </button>
      {expanded ? (
        <div className="json-children">
          {entries.map(([key, item]) => (
            <JsonNode key={`${depth}-${key}`} label={key} value={item} depth={depth + 1} />
          ))}
          <div className="json-row">
            <span className="json-bracket">{close}</span>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export function JsonViewer({
  value,
  raw,
  label,
}: {
  value: JsonValue;
  raw: string;
  label: string;
}) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(raw);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="json-shell">
      <div className="json-toolbar">
        <span className="json-toolbar-label">{label}</span>
        <button className="code-block-copy" onClick={handleCopy} type="button">
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <div className="json-tree">
        <JsonNode value={value} />
      </div>
    </div>
  );
}

function CodeBlock({
  className,
  children,
  onOpenArtifact,
}: {
  className?: string;
  children: ReactNode;
  onOpenArtifact?: (content: string, language: string) => void;
}) {
  const [copied, setCopied] = useState(false);
  const codeText = useMemo(() => flattenText(children).replace(/\n$/, ""), [children]);
  const language = String(className || "")
    .replace("hljs", "")
    .replace("language-", "")
    .trim() || "code";
  const parsedJson = useMemo(
    () => (language === "json" ? tryParseJson(codeText) : null),
    [codeText, language],
  );

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(codeText);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  const canOpenAsArtifact = onOpenArtifact && (
    language === "html" ||
    language === "svg" ||
    language === "mermaid" ||
    codeText.split("\n").length > 15
  );

  if (parsedJson !== null) {
    return <JsonViewer value={parsedJson} raw={codeText} label="json" />;
  }

  if (language === "diff" || language === "patch" || isUnifiedDiffText(codeText)) {
    return <DiffViewer diff={codeText} label={language} />;
  }

  return (
    <div className="code-block-shell">
      <div className="code-block-toolbar">
        <span className="code-block-language">{language}</span>
        {canOpenAsArtifact ? (
          <button
            className="code-block-copy code-block-open-artifact"
            onClick={() => onOpenArtifact!(codeText, language)}
            type="button"
            title="Open in Artifacts panel"
          >
            Open
          </button>
        ) : null}
        <button className="code-block-copy" onClick={handleCopy} type="button">
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto rounded-b-[22px] border border-t-0 border-white/10 bg-[#0a1215] px-4 py-4 font-mono text-[13px] leading-6 text-sand/90 shadow-inner">
        <code className={className}>{children}</code>
      </pre>
    </div>
  );
}

export function MarkdownMessage({
  content,
  streaming = false,
  onOpenArtifact,
}: {
  content: string;
  streaming?: boolean;
  onOpenArtifact?: (content: string, language: string) => void;
}) {
  const renderedContent = useMemo(
    () => normalizeStreamingMarkdown(content, streaming),
    [content, streaming],
  );
  const parsedJson = useMemo(
    () => (!streaming ? tryParseJson(renderedContent) : null),
    [renderedContent, streaming],
  );

  if (parsedJson !== null) {
    return (
      <div className="markdown-body mt-3 text-[15px] leading-7 text-sand/90">
        <JsonViewer value={parsedJson} raw={renderedContent} label="json response" />
      </div>
    );
  }

  return (
    <div className="markdown-body mt-3 text-[15px] leading-7 text-sand/90">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          a: ({ ...props }) => (
            <a
              {...props}
              target="_blank"
              rel="noreferrer"
              className="text-tide underline decoration-tide/40 underline-offset-4 transition hover:text-white"
            />
          ),
          code: ({ className, children, ...props }) => {
            const text = flattenText(children ?? "");
            const isBlock = text.includes("\n") || Boolean(className);
            if (!isBlock) {
              return (
                <code
                  {...props}
                  className="rounded-md bg-black/30 px-1.5 py-0.5 font-mono text-[0.92em] text-ember"
                >
                  {children}
                </code>
              );
            }
            return <CodeBlock className={className} onOpenArtifact={onOpenArtifact} children={children} />;
          },
          pre: ({ children }) => <>{children}</>,
          table: ({ children }) => (
            <div className="overflow-x-auto">
              <table className="min-w-full border-collapse overflow-hidden rounded-[18px] border border-white/10">
                {children}
              </table>
            </div>
          ),
          th: ({ children }) => (
            <th className="border border-white/10 bg-white/5 px-3 py-2 text-left text-xs font-semibold uppercase tracking-[0.14em] text-sand/70">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="border border-white/10 px-3 py-2 align-top text-sm text-sand/75">
              {children}
            </td>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-tide/40 pl-4 italic text-sand/70">
              {children}
            </blockquote>
          ),
        }}
      >
        {renderedContent}
      </ReactMarkdown>
      {streaming ? <span className="streaming-cursor" /> : null}
    </div>
  );
}
