import { useState } from "react";
import { type ChatSession, type RepoEntry } from "./types";
import { compactText, formatSessionStamp } from "./utils/messageFormatting";

type Props = {
  sessions: ChatSession[];
  selectedSessionId: string | null;
  repos: RepoEntry[];
  selectedRepoId: string;
  sessionFilter: string;
  setSessionFilter: (value: string) => void;
  onSessionSelect: (sessionId: string) => Promise<void>;
  onRepoSelect: (id: string) => void;
  onRefresh: () => void;
  sidebarCollapsed: boolean;
  setSidebarCollapsed: (v: boolean) => void;
  filteredSessions: ChatSession[];
  groupedSessions: { label: string; sessions: ChatSession[] }[];
  editingTitle: string | null;
  titleDraft: string;
  setEditingTitle: (v: string | null) => void;
  titleDraftChanged: (v: string) => void;
  handleRenameSession: (sessionId: string, draft: string) => Promise<void>;
  selectedRepo: RepoEntry | null;
  onNewSession: () => void;
};

function SidebarIconButton({
  title,
  onClick,
  children,
}: {
  title: string;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className="flex h-10 w-10 items-center justify-center rounded-2xl border border-white/10 bg-white/5 text-sand/60 transition hover:border-tide/30 hover:bg-tide/10 hover:text-white"
    >
      {children}
    </button>
  );
}

function SessionItem({
  session,
  isSelected,
  onClick,
  onRename,
  onDelete,
  editingTitle,
  titleDraft,
  setEditingTitle,
  titleDraftChanged,
}: {
  session: ChatSession;
  isSelected: boolean;
  onClick: () => void;
  onRename: (sessionId: string, draft: string) => Promise<void>;
  onDelete: (sessionId: string) => void;
  editingTitle: string | null;
  titleDraft: string;
  setEditingTitle: (v: string | null) => void;
  titleDraftChanged: (v: string) => void;
}) {
  const isEditing = editingTitle === session.id;

  return (
    <div
      onClick={onClick}
      className={`session-pill ${isSelected ? "session-pill-active" : "session-pill-flat"} group cursor-pointer`}
    >
      {isEditing ? (
        <div className="space-y-2">
          <input
            type="text"
            value={titleDraft}
            onChange={(e) => titleDraftChanged(e.target.value)}
            onBlur={() => void onRename(session.id, titleDraft)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                void onRename(session.id, titleDraft);
              } else if (e.key === "Escape") {
                setEditingTitle(null);
              }
            }}
            className="field !rounded-2xl !px-3 !py-2"
            autoFocus
          />
          <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.16em] text-sand/40">
            <span>{formatSessionStamp(session.last_updated)}</span>
            <span>{session.message_count} msgs</span>
          </div>
        </div>
      ) : (
        <div className="space-y-2">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold text-sand/90">{session.title}</div>
              <div className="session-pill-meta mt-1 flex items-center gap-2">
                <span>{formatSessionStamp(session.last_updated)}</span>
                <span className="text-sand/25">•</span>
                <span>{session.message_count} msgs</span>
              </div>
            </div>
            <div className="flex items-center gap-1 opacity-0 transition group-hover:opacity-100">
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  titleDraftChanged(session.title);
                  setEditingTitle(session.id);
                }}
                className="rounded-full p-1.5 text-sand/40 transition hover:bg-white/5 hover:text-white"
                title="Rename"
              >
                <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
              </button>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(session.id);
                }}
                className="rounded-full p-1.5 text-sand/40 transition hover:bg-rose-500/10 hover:text-rose"
                title="Delete"
              >
                <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            </div>
          </div>
          {session.preview ? (
            <p className="line-clamp-2 text-xs leading-5 text-sand/50">
              {compactText(session.preview, 112)}
            </p>
          ) : null}
        </div>
      )}
    </div>
  );
}

function SessionGroup({
  label,
  sessions,
  selectedSessionId,
  onSessionSelect,
  onRename,
  onDelete,
  editingTitle,
  titleDraft,
  setEditingTitle,
  titleDraftChanged,
}: {
  label: string;
  sessions: ChatSession[];
  selectedSessionId: string | null;
  onSessionSelect: (id: string) => void;
  onRename: (sessionId: string, draft: string) => Promise<void>;
  onDelete: (sessionId: string) => void;
  editingTitle: string | null;
  titleDraft: string;
  setEditingTitle: (v: string | null) => void;
  titleDraftChanged: (v: string) => void;
}) {
  if (sessions.length === 0) {
    return null;
  }

  return (
    <section className="sidebar-group">
      <div className="session-group-heading mb-2 px-1">
        <span>{label}</span>
        <span>{sessions.length}</span>
      </div>
      <div className="space-y-2">
        {sessions.map((session) => (
          <SessionItem
            key={session.id}
            session={session}
            isSelected={session.id === selectedSessionId}
            onClick={() => onSessionSelect(session.id)}
            onRename={onRename}
            onDelete={onDelete}
            editingTitle={editingTitle}
            titleDraft={titleDraft}
            setEditingTitle={setEditingTitle}
            titleDraftChanged={titleDraftChanged}
          />
        ))}
      </div>
    </section>
  );
}

function ClearAllButton({ onClick }: { onClick: () => void }) {
  const [confirming, setConfirming] = useState(false);

  return (
    <button
      type="button"
      onClick={() => {
        if (confirming) {
          onClick();
          setConfirming(false);
          return;
        }
        setConfirming(true);
        window.setTimeout(() => setConfirming(false), 2200);
      }}
      className="action-button w-full justify-center !rounded-2xl !border-rose/20 !text-rose hover:!border-rose/40 hover:!bg-rose/10"
    >
      {confirming ? "Click again to confirm" : "Clear all sessions"}
    </button>
  );
}

export function SessionSidebar({
  sessions,
  selectedSessionId,
  repos,
  selectedRepoId,
  sessionFilter,
  setSessionFilter,
  onSessionSelect,
  onRepoSelect,
  onRefresh,
  sidebarCollapsed,
  setSidebarCollapsed,
  filteredSessions,
  groupedSessions,
  editingTitle,
  titleDraft,
  setEditingTitle,
  titleDraftChanged,
  handleRenameSession,
  selectedRepo,
  onNewSession,
}: Props) {
  const recentSessions = sessions.slice(0, 4);

  if (sidebarCollapsed) {
    return (
      <aside className="sidebar-shell">
        <div className="sidebar-frame sidebar-frame-collapsed h-full items-center">
          <div className="flex flex-col items-center gap-3">
            <button
              type="button"
              onClick={() => setSidebarCollapsed(false)}
              className="flex h-11 w-11 items-center justify-center rounded-[18px] border border-tide/20 bg-tide/10 text-white transition hover:border-tide/40 hover:bg-tide/20"
              title="Expand sidebar"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 12h16m-7-7 7 7-7 7" />
              </svg>
            </button>
            <div className="text-[10px] font-semibold uppercase tracking-[0.28em] text-sand/40">
              Chat
            </div>
            <SidebarIconButton title="Fresh thread" onClick={onNewSession}>
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            </SidebarIconButton>
          </div>

          <div className="sidebar-scroll !space-y-2">
            {recentSessions.map((session) => (
              <button
                key={session.id}
                type="button"
                onClick={() => void onSessionSelect(session.id)}
                title={session.title}
                className={`flex h-11 w-11 items-center justify-center rounded-[16px] border text-xs font-semibold transition ${
                  session.id === selectedSessionId
                    ? "border-tide/30 bg-tide/10 text-white"
                    : "border-white/10 bg-white/5 text-sand/60 hover:border-white/20 hover:bg-white/10"
                }`}
              >
                {session.title.trim().charAt(0).toUpperCase() || "S"}
              </button>
            ))}
          </div>

          <div className="sidebar-footer w-full !border-t-0 !pt-0">
            <SidebarIconButton title="Refresh sessions" onClick={onRefresh}>
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </SidebarIconButton>
          </div>
        </div>
      </aside>
    );
  }

  return (
    <aside className="sidebar-shell">
      <div className="sidebar-frame h-full">
        <div className="flex items-center gap-2 px-1">
          <button
            type="button"
            onClick={onNewSession}
            className="action-button-primary flex-1 !rounded-2xl !px-3.5 !py-2"
            title="Fresh thread"
          >
            <svg className="mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            New
          </button>
          <SidebarIconButton title="Refresh sessions" onClick={onRefresh}>
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </SidebarIconButton>
          <SidebarIconButton title="Collapse sidebar" onClick={() => setSidebarCollapsed(true)}>
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4m7-7-7 7 7 7" />
            </svg>
          </SidebarIconButton>
        </div>

        <div className="sidebar-section">
          <div>
            <div className="panel-title">History</div>
            <h2 className="mt-1 font-['Space_Grotesk'] text-2xl font-semibold tracking-[-0.04em] text-white">
              Conversations
            </h2>
            <p className="mt-1 text-sm text-sand/50">
              Browse recent threads, switch repo context, and jump back into earlier work.
            </p>
          </div>

          <div className="mt-4 grid grid-cols-2 gap-2">
            <div className="mini-stat-card compact-stat-card">
              <div className="text-[11px] uppercase tracking-[0.16em] text-sand/40">Visible</div>
              <div className="mt-1 text-lg font-semibold text-white">{filteredSessions.length}</div>
            </div>
            <div className="mini-stat-card compact-stat-card">
              <div className="text-[11px] uppercase tracking-[0.16em] text-sand/40">Total</div>
              <div className="mt-1 text-lg font-semibold text-white">{sessions.length}</div>
            </div>
          </div>
        </div>

        <div className="sidebar-section">
          <label htmlFor="session-filter" className="panel-title block pb-2">
            Search History
          </label>
          <input
            id="session-filter"
            type="text"
            value={sessionFilter}
            onChange={(e) => setSessionFilter(e.target.value)}
            placeholder="Search title or preview..."
            className="field"
          />
        </div>

        <div className="sidebar-section">
          <div className="panel-title pb-2">Repository Scope</div>
          {repos.length > 0 ? (
            <div className="space-y-3">
              <div className="space-y-2">
                <button
                  type="button"
                  onClick={() => onRepoSelect("")}
                  className={`repo-chip ${selectedRepoId ? "repo-chip-flat" : "repo-chip-active"}`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate text-sm font-semibold text-white">None</div>
                      <div className="mt-1 text-xs text-sand/45">
                        Global workspace context
                      </div>
                    </div>
                    <span className="chip">global</span>
                  </div>
                </button>

                {repos.map((repo) => {
                  const isActive = repo.id === selectedRepoId;
                  return (
                    <button
                      key={repo.id}
                      type="button"
                      onClick={() => onRepoSelect(repo.id)}
                      className={`repo-chip ${isActive ? "repo-chip-active" : "repo-chip-flat"}`}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <div className="truncate text-sm font-semibold text-white">{repo.name}</div>
                          <div className="mt-1 text-xs text-sand/45">
                            {repo.graph_nodes} nodes · {repo.graph_edges} edges
                          </div>
                        </div>
                        <span className="chip">{repo.git?.branch ?? "repo"}</span>
                      </div>
                    </button>
                  );
                })}
              </div>

              {selectedRepo ? (
                <div className="soft-panel rounded-[20px] border border-white/10 px-4 py-3">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <div className="text-sm font-semibold text-white">{selectedRepo.name}</div>
                      <div className="mt-1 text-xs text-sand/50">
                        {selectedRepo.graph_nodes} nodes · {selectedRepo.graph_edges} edges
                      </div>
                    </div>
                    <span className="chip">{selectedRepo.git?.branch ?? "repo"}</span>
                  </div>
                </div>
              ) : null}
            </div>
          ) : (
            <p className="text-sm text-sand/50">No repositories indexed yet.</p>
          )}
        </div>

        <div className="sidebar-scroll">
          {groupedSessions.length > 0 ? (
            groupedSessions.map((group) => (
              <SessionGroup
                key={group.label}
                label={group.label}
                sessions={group.sessions}
                selectedSessionId={selectedSessionId}
                onSessionSelect={onSessionSelect}
                onRename={handleRenameSession}
                onDelete={() => {}}
                editingTitle={editingTitle}
                titleDraft={titleDraft}
                setEditingTitle={setEditingTitle}
                titleDraftChanged={titleDraftChanged}
              />
            ))
          ) : (
            <div className="sidebar-section">
              <div className="text-sm font-semibold text-white">
                {sessions.length === 0 ? "No conversations yet" : "No matches"}
              </div>
              <p className="mt-2 text-sm text-sand/50">
                {sessions.length === 0
                  ? "Start a conversation and it will show up here."
                  : "Try a broader search term to find older threads."}
              </p>
            </div>
          )}
        </div>

        <div className="sidebar-footer">
          <div className="space-y-2">
            <ClearAllButton onClick={() => {}} />
            <p className="px-1 text-xs leading-5 text-sand/35">
              Session delete and clear-all are not exposed by the backend yet, so these remain visual placeholders for now.
            </p>
          </div>
        </div>
      </div>
    </aside>
  );
}

export default SessionSidebar;
