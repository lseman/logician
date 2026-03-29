import { useCallback } from "react";
import type {
  Artifact,
  GraphPayload,
  MemoryPayload,
  RagResult,
  RepoEntry,
  RunEvent,
  TodoItem,
} from "./types";
import { compactText, formatSessionStamp } from "./utils/messageFormatting";
import { RunTimeline } from "./components/RunTimeline";
import { ToolCallRenderer, type ToolCallItem } from "./components/ToolCallRenderer";

type Pane = "graph" | "todos" | "memory" | "artifacts";

type Props = {
  activePane: Pane;
  setActivePane: (v: Pane) => void;
  repos: RepoEntry[];
  selectedRepoId: string;
  setSelectedRepoId: (v: string) => void;
  graph: GraphPayload;
  ragQuery: string;
  setRagQuery: (v: string) => void;
  ragLoading: boolean;
  ragResults: RagResult[];
  ragError: string;
  runRagSearch: (query: string) => Promise<void>;
  todos: TodoItem[];
  todoDraft: string;
  setTodoDraft: (v: string) => void;
  handleAddTodo: (event: React.FormEvent) => Promise<void>;
  updateTodo: (id: number, status: TodoItem["status"]) => Promise<void>;
  memoryData: MemoryPayload | null;
  memoryQuery: string;
  setMemoryQuery: (v: string) => void;
  memoryType: string;
  setMemoryType: (v: string) => void;
  graphStats: { highlighted: number; nodes: number; edges: number };
  artifacts: Artifact[];
  activeArtifact: Artifact | null;
  setActiveArtifact: (v: Artifact | null) => void;
  liveTimeline: RunEvent[];
  streamingTools: ToolCallItem[];
  streamingThinking: string[];
  sending: boolean;
  waitingFirstToken: boolean;
};

function CurrentReasoningPanel({
  liveTimeline,
  streamingTools,
  streamingThinking,
  sending,
  waitingFirstToken,
}: {
  liveTimeline: RunEvent[];
  streamingTools: ToolCallItem[];
  streamingThinking: string[];
  sending: boolean;
  waitingFirstToken: boolean;
}) {
  const recentThinking = streamingThinking.slice(-3).reverse();

  return (
    <div className="utility-block">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="panel-title">Current Reasoning</div>
          <div className="mt-1 text-sm text-sand/60">
            {sending
              ? waitingFirstToken && liveTimeline.length === 0 && streamingTools.length === 0
                ? "Waiting for the first action..."
                : "Tracking live reasoning steps"
              : "No active reasoning run"}
          </div>
        </div>
        <span className={`agent-status-chip ${sending ? "agent-status-active" : "agent-status-idle"}`}>
          {sending ? <span className="agent-status-ring" /> : null}
          <span>{sending ? "Live" : "Idle"}</span>
        </span>
      </div>

      {recentThinking.length > 0 ? (
        <div className="mt-4 space-y-2">
          {recentThinking.map((entry, index) => (
            <div
              key={`${index}-${entry.slice(0, 24)}`}
              className="rounded-2xl border border-white/8 bg-white/5 px-3 py-2 text-sm text-sand/65"
            >
              {compactText(entry, 120)}
            </div>
          ))}
        </div>
      ) : null}

      {streamingTools.length > 0 ? (
        <div className="mt-4">
          <ToolCallRenderer items={streamingTools} streaming />
        </div>
      ) : null}

      <div className="mt-4">
        <RunTimeline
          items={liveTimeline}
          live={sending}
          title="Action history"
          emptyLabel={sending ? "Waiting for actions..." : "No actions yet."}
        />
      </div>
    </div>
  );
}

function TabButton({
  active,
  label,
  count,
  onClick,
}: {
  active: boolean;
  label: string;
  count?: number;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`tab-toggle ${active ? "tab-toggle-active" : "tab-toggle-idle"}`}
    >
      <span>{label}</span>
      {typeof count === "number" ? <span className="ml-1 opacity-70">{count}</span> : null}
    </button>
  );
}

function SearchResultItem({ result }: { result: RagResult }) {
  const lang = String(result.metadata?.language || result.metadata?.ext || "code");
  return (
    <div className="utility-block">
      <div className="mb-3 flex items-center justify-between gap-2">
        <span className="chip">{lang}</span>
        <span className="text-[11px] uppercase tracking-[0.16em] text-sand/40">match</span>
      </div>
      <div className="mb-2">
        <div className="text-sm font-semibold text-white">{result.source}</div>
        <div className="text-xs text-sand/50">{result.rel_path}</div>
      </div>
      <pre className="max-h-28 overflow-auto rounded-2xl border border-white/10 bg-black/20 p-3 text-xs leading-5 text-sand/60">
        <code>{result.content}</code>
      </pre>
    </div>
  );
}

function TodoCard({
  item,
  onUpdate,
}: {
  item: TodoItem;
  onUpdate: (id: number, status: TodoItem["status"]) => Promise<void>;
}) {
  return (
    <div className="utility-block">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-white">{item.title}</div>
          {item.note ? <div className="mt-1 text-xs text-sand/50">{item.note}</div> : null}
        </div>
        <span className="chip">{item.status}</span>
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        <button type="button" className="action-button !px-3 !py-1.5 !text-xs" onClick={() => void onUpdate(item.id, "not-started")}>
          Todo
        </button>
        <button type="button" className="action-button !px-3 !py-1.5 !text-xs" onClick={() => void onUpdate(item.id, "in-progress")}>
          Doing
        </button>
        <button type="button" className="action-button !px-3 !py-1.5 !text-xs" onClick={() => void onUpdate(item.id, "completed")}>
          Done
        </button>
        <button type="button" className="action-button !px-3 !py-1.5 !text-xs" onClick={() => void onUpdate(item.id, "blocked")}>
          Blocked
        </button>
      </div>
    </div>
  );
}

export function UtilityRail({
  activePane,
  setActivePane,
  repos,
  selectedRepoId,
  setSelectedRepoId,
  graph,
  ragQuery,
  setRagQuery,
  ragLoading,
  ragResults,
  ragError,
  runRagSearch,
  todos,
  todoDraft,
  setTodoDraft,
  handleAddTodo,
  updateTodo,
  memoryData,
  memoryQuery,
  setMemoryQuery,
  memoryType,
  setMemoryType,
  graphStats,
  artifacts,
  activeArtifact,
  setActiveArtifact,
  liveTimeline,
  streamingTools,
  streamingThinking,
  sending,
  waitingFirstToken,
}: Props) {
  const memoryTypeEntries = Object.entries(memoryData?.overview.types ?? {});
  const selectedRepo = repos.find((repo) => repo.id === selectedRepoId) ?? null;

  const submitRagSearch = useCallback(async (event: React.FormEvent) => {
    event.preventDefault();
    await runRagSearch(ragQuery);
  }, [ragQuery, runRagSearch]);

  return (
    <aside className="utility-rail flex h-full flex-col p-3">
      <CurrentReasoningPanel
        liveTimeline={liveTimeline}
        streamingTools={streamingTools}
        streamingThinking={streamingThinking}
        sending={sending}
        waitingFirstToken={waitingFirstToken}
      />

      <div className="mt-3 utility-block">
        <div className="panel-title">Workspace Rail</div>
        <div className="mt-3 flex flex-wrap gap-2">
          <TabButton active={activePane === "graph"} label="Graph" onClick={() => setActivePane("graph")} />
          <TabButton active={activePane === "todos"} label="Todos" count={todos.length} onClick={() => setActivePane("todos")} />
          <TabButton active={activePane === "memory"} label="Memory" count={memoryData?.overview.count ?? 0} onClick={() => setActivePane("memory")} />
          <TabButton active={activePane === "artifacts"} label="Artifacts" count={artifacts.length} onClick={() => setActivePane("artifacts")} />
        </div>
      </div>

      <div className="mt-3 flex-1 overflow-y-auto space-y-3 pr-1">
        {activePane === "graph" ? (
          <>
            <div className="utility-block">
              <div className="panel-title">Repository Graph</div>
              <div className="mt-3 space-y-3">
                {repos.length > 0 ? (
                  <>
                    <select
                      value={selectedRepoId}
                      onChange={(e) => setSelectedRepoId(e.target.value)}
                      className="field"
                    >
                      <option value="">None (global)</option>
                      {repos.map((repo) => (
                        <option key={repo.id} value={repo.id}>
                          {repo.name}
                        </option>
                      ))}
                    </select>
                    <div className="grid grid-cols-3 gap-2">
                      <div className="mini-stat-card compact-stat-card">
                        <div className="text-[11px] uppercase tracking-[0.16em] text-sand/40">Nodes</div>
                        <div className="mt-1 text-lg font-semibold text-white">{graphStats.nodes}</div>
                      </div>
                      <div className="mini-stat-card compact-stat-card">
                        <div className="text-[11px] uppercase tracking-[0.16em] text-sand/40">Edges</div>
                        <div className="mt-1 text-lg font-semibold text-white">{graphStats.edges}</div>
                      </div>
                      <div className="mini-stat-card compact-stat-card">
                        <div className="text-[11px] uppercase tracking-[0.16em] text-sand/40">Focus</div>
                        <div className="mt-1 text-lg font-semibold text-white">{graphStats.highlighted}</div>
                      </div>
                    </div>
                    <div className="rounded-[22px] border border-white/10 bg-[radial-gradient(circle_at_top_left,rgba(103,183,200,0.16),transparent_32%),linear-gradient(180deg,rgba(255,255,255,0.03),rgba(255,255,255,0.01))] p-4">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <div className="text-sm font-semibold text-white">
                            {selectedRepo?.name ?? "Global workspace"}
                          </div>
                          <div className="mt-1 text-xs text-sand/50">
                            {graph.focus_paths?.length ?? 0} focused paths
                          </div>
                        </div>
                        <span className="chip">{selectedRepo?.git?.branch ?? "repo"}</span>
                      </div>
                      <div className="mt-4 aspect-[16/10] rounded-[18px] border border-white/10 bg-[radial-gradient(circle_at_30%_20%,rgba(103,183,200,0.12),transparent_24%),radial-gradient(circle_at_70%_70%,rgba(210,163,93,0.12),transparent_28%),linear-gradient(180deg,rgba(7,12,18,0.6),rgba(7,12,18,0.9))]" />
                    </div>
                  </>
                ) : (
                  <div className="text-sm text-sand/50">No repositories indexed yet.</div>
                )}
              </div>
            </div>

            <div className="utility-block">
              <div className="panel-title">Semantic Search</div>
              <form className="mt-3 space-y-3" onSubmit={submitRagSearch}>
                <input
                  type="text"
                  value={ragQuery}
                  onChange={(e) => setRagQuery(e.target.value)}
                  placeholder="Find file, function, or concept..."
                  className="field"
                />
                <button type="submit" className="action-button-primary w-full justify-center" disabled={ragLoading || !ragQuery.trim()}>
                  {ragLoading ? "Searching..." : "Search codebase"}
                </button>
              </form>
              {ragError ? <div className="mt-3 text-sm text-rose">{ragError}</div> : null}
            </div>

            {ragResults.length > 0 ? (
              <div className="space-y-3">
                {ragResults.map((result, index) => (
                  <SearchResultItem key={`${result.rel_path}-${index}`} result={result} />
                ))}
              </div>
            ) : null}
          </>
        ) : null}

        {activePane === "todos" ? (
          <>
            <div className="utility-block">
              <div className="panel-title">Current Plan</div>
              <form className="mt-3 space-y-3" onSubmit={handleAddTodo}>
                <input
                  type="text"
                  value={todoDraft}
                  onChange={(e) => setTodoDraft(e.target.value)}
                  placeholder="Add a next step..."
                  className="field"
                />
                <button type="submit" className="action-button-primary w-full justify-center" disabled={!todoDraft.trim()}>
                  Add todo
                </button>
              </form>
            </div>
            {todos.length > 0 ? (
              todos.map((item) => <TodoCard key={item.id} item={item} onUpdate={updateTodo} />)
            ) : (
              <div className="utility-block text-sm text-sand/50">No todos yet.</div>
            )}
          </>
        ) : null}

        {activePane === "memory" ? (
          <>
            <div className="utility-block">
              <div className="panel-title">Project Memory</div>
              <div className="mt-3 grid grid-cols-2 gap-2">
                <div className="mini-stat-card compact-stat-card">
                  <div className="text-[11px] uppercase tracking-[0.16em] text-sand/40">Notes</div>
                  <div className="mt-1 text-lg font-semibold text-white">{memoryData?.overview.count ?? 0}</div>
                </div>
                <div className="mini-stat-card compact-stat-card">
                  <div className="text-[11px] uppercase tracking-[0.16em] text-sand/40">Facts</div>
                  <div className="mt-1 text-lg font-semibold text-white">{memoryData?.facts.length ?? 0}</div>
                </div>
              </div>
              <div className="mt-3 space-y-3">
                <input
                  type="text"
                  value={memoryQuery}
                  onChange={(e) => setMemoryQuery(e.target.value)}
                  placeholder="Filter memory..."
                  className="field"
                />
                <select value={memoryType} onChange={(e) => setMemoryType(e.target.value)} className="field">
                  <option value="">All memory types</option>
                  {memoryTypeEntries.map(([type, count]) => (
                    <option key={type} value={type}>
                      {type} ({count})
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {memoryData?.overview.summary_markdown ? (
              <div className="utility-block">
                <div className="panel-title">Summary</div>
                <p className="mt-3 whitespace-pre-wrap text-sm leading-6 text-sand/70">
                  {compactText(memoryData.overview.summary_markdown, 520)}
                </p>
              </div>
            ) : null}

            {memoryData?.facts.slice(0, 6).map((fact) => (
              <div key={`${fact.name}:${fact.path}`} className="utility-block">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm font-semibold text-white">{fact.name}</div>
                  <span className="chip">{fact.type}</span>
                </div>
                <div className="mt-2 text-sm text-sand/60">{fact.description}</div>
                <div className="mt-2 text-xs text-sand/40">{fact.path}</div>
              </div>
            ))}

            {memoryData?.observations.slice(0, 5).map((observation) => (
              <div key={observation.id} className="utility-block">
                <div className="flex items-start justify-between gap-3">
                  <div className="text-sm font-semibold text-white">{observation.title}</div>
                  <span className="chip">{observation.type}</span>
                </div>
                <div className="mt-2 text-sm text-sand/60">
                  {compactText(observation.preview || observation.content, 180)}
                </div>
                <div className="mt-2 text-xs text-sand/40">
                  {formatSessionStamp(observation.timestamp)} · {observation.path}
                </div>
              </div>
            ))}
          </>
        ) : null}

        {activePane === "artifacts" ? (
          artifacts.length > 0 ? (
            artifacts.map((artifact) => {
              const isActive = activeArtifact?.id === artifact.id;
              return (
                <button
                  key={artifact.id}
                  type="button"
                  onClick={() => setActiveArtifact(artifact)}
                  className={`utility-block w-full text-left transition ${isActive ? "border-tide/30 bg-tide/10" : ""}`}
                >
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <div className="text-sm font-semibold text-white">{artifact.title}</div>
                      <div className="mt-1 text-xs text-sand/50">{artifact.kind} · {artifact.language || "plain"}</div>
                    </div>
                    <span className="chip">{artifact.kind}</span>
                  </div>
                </button>
              );
            })
          ) : (
            <div className="utility-block text-sm text-sand/50">No artifacts generated yet.</div>
          )
        ) : null}
      </div>
    </aside>
  );
}

export default UtilityRail;
