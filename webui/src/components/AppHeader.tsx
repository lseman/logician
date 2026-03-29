export function AppHeader({
  sessionsCount,
  todosCount,
  memoryCount,
  repoName,
  graphLabel,
  sessionTitle,
  backendStatus,
}: {
  sessionsCount: number;
  todosCount: number;
  memoryCount: number;
  repoName: string;
  graphLabel: string;
  sessionTitle: string;
  backendStatus: {
    llm_url: string;
    llm_reachable: boolean;
  } | null;
}) {
  return (
    <header className="topbar-shell">
      <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
        <div className="flex items-center gap-4">
          <div className="brand-lockup">
            <div className="brand-wordmark" aria-label="Logician">
              Logician
            </div>
          </div>
          <div className="topbar-status">
            <span className="topbar-stat">{sessionsCount} conversations</span>
            <span className="topbar-stat">{todosCount} todos</span>
            <span className="topbar-stat">{memoryCount} memory notes</span>
          </div>
        </div>

        <div className="topbar-meta">
          <span className="topbar-meta-item">
            <span className="topbar-meta-label">Repo</span>
            <span>{repoName}</span>
          </span>
          <span className="topbar-meta-item">
            <span className="topbar-meta-label">Graph</span>
            <span>{graphLabel}</span>
          </span>
          <span className="topbar-meta-item">
            <span className="topbar-meta-label">Session</span>
            <span>{sessionTitle}</span>
          </span>
          <span
            className={`topbar-meta-item ${
              backendStatus?.llm_reachable ? "topbar-meta-item-online" : "topbar-meta-item-offline"
            }`}
          >
            {backendStatus?.llm_reachable ? "LLM online" : "LLM offline"}
          </span>
        </div>
      </div>

      {backendStatus && !backendStatus.llm_reachable ? (
        <div className="hero-alert mt-3 rounded-[18px] px-4 py-2.5 text-sm text-sand/80">
          No model backend is reachable at <span className="font-mono">{backendStatus.llm_url}</span>.
        </div>
      ) : null}
    </header>
  );
}
