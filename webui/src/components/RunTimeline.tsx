import type { RunEvent } from "../types";

export function RunTimeline({
  items,
  live = false,
  title = "Run timeline",
  emptyLabel = "Waiting for streaming events...",
}: {
  items: RunEvent[];
  live?: boolean;
  title?: string;
  emptyLabel?: string;
}) {
  return (
    <div className="run-timeline">
      <div className="run-timeline-header">
        <div className="flex items-center gap-2">
          <span className="run-timeline-title">{title}</span>
          {live ? <span className="thinking-live-dot" aria-hidden="true" /> : null}
        </div>
        <span className="run-timeline-count">
          {items.length === 0 ? "idle" : `${items.length} events`}
        </span>
      </div>

      {items.length === 0 ? (
        <div className="run-timeline-empty">{emptyLabel}</div>
      ) : (
        <div className="run-timeline-list">
          {items.map((item) => (
            <div key={item.id} className={`run-timeline-item run-timeline-item-${item.state || "done"}`}>
              <div className={`run-timeline-dot run-timeline-dot-${item.state || "done"}`} />
              <div className="min-w-0 flex-1">
                <div className="run-timeline-row">
                  <span className="run-timeline-label">{item.label}</span>
                  {item.meta ? <span className="run-timeline-meta">{item.meta}</span> : null}
                </div>
                {item.detail ? <div className="run-timeline-detail">{item.detail}</div> : null}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
