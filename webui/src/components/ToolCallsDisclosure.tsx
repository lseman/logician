import { useEffect, useState } from "react";
import { ToolCallCard, type ToolCallItem } from "./ToolCallCard";

export function ToolCallsDisclosure({
  items,
  streaming = false,
}: {
  items: ToolCallItem[];
  streaming?: boolean;
}) {
  const [expanded, setExpanded] = useState(streaming);

  useEffect(() => {
    if (streaming && items.length > 0) {
      setExpanded(true);
    }
  }, [streaming, items.length]);

  if (items.length === 0) return null;

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
          <span className="thinking-label">
            {streaming ? "Running tools" : "Tools used"}
          </span>
          {streaming ? <span className="thinking-live-dot" aria-hidden="true" /> : null}
        </div>
        <div className="thinking-meta">
          {streaming ? "live" : `${items.length} ${items.length === 1 ? "call" : "calls"}`}
        </div>
      </button>

      {expanded ? (
        <div className="thinking-body">
          <div className="space-y-1.5 py-1">
            {items.map((item) => (
              <ToolCallCard key={item.id} item={item} done={!streaming} />
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
