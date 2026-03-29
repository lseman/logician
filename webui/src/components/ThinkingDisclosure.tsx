import { useEffect, useMemo, useState } from "react";
import { MarkdownMessage } from "./MarkdownMessage";

function normalizeThinkingEntries(entries: string[]) {
  return entries
    .map((entry) => String(entry || "").trim())
    .filter((entry) => entry.length > 0);
}

export function ThinkingDisclosure({
  entries,
  streaming = false,
}: {
  entries: string[];
  streaming?: boolean;
}) {
  const normalizedEntries = useMemo(() => normalizeThinkingEntries(entries), [entries]);
  const [expanded, setExpanded] = useState(normalizedEntries.length > 0);

  useEffect(() => {
    if (normalizedEntries.length > 0) {
      setExpanded(true);
    }
  }, [normalizedEntries.length]);

  if (normalizedEntries.length === 0) {
    return null;
  }

  return (
    <div className="thinking-shell">
      <button
        type="button"
        className="thinking-toggle"
        onClick={() => setExpanded((current) => !current)}
        aria-expanded={expanded}
      >
        <div className="flex items-center gap-3">
          <span className={`thinking-caret ${expanded ? "is-open" : ""}`}>▸</span>
          <span className="thinking-label">Thinking...</span>
          {streaming ? <span className="thinking-live-dot" aria-hidden="true" /> : null}
        </div>
        <div className="thinking-meta">
          {streaming
            ? "live"
            : `${normalizedEntries.length} ${normalizedEntries.length === 1 ? "note" : "notes"}`}
        </div>
      </button>

      {expanded ? (
        <div className="thinking-body">
          {normalizedEntries.map((entry, index) => (
            <div key={`${index}-${entry.slice(0, 24)}`} className="thinking-entry">
              <MarkdownMessage content={entry} />
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
