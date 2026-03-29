import {
  FormEvent,
  UIEvent,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
  startTransition,
  useCallback,
} from "react";
import { api, streamSse } from "./api";
import { type ToolCallItem } from "./components/ToolCallRenderer";
import { ChatInterface } from "./ChatInterface";
import { SessionSidebar } from "./SessionSidebar";
import { UtilityRail } from "./UtilityRail";
import { AppHeader } from "./components/AppHeader";
import { ToastStack } from "./components/ToastStack";
import type { Artifact } from "./types";
import {
  ChatMessage,
  ChatSession,
  GraphPayload,
  MemoryPayload,
  RagResult,
  RepoEntry,
  RunEvent,
  TodoItem,
} from "./types";
import {
  compactText,
  groupSessionsByAge,
  buildMessageWithAttachments,
} from "./utils/messageFormatting";
import { buildDisplayMessages } from "./utils/messageBuilding";
import { useAttachments } from "./hooks/useAttachments";

type Pane = "graph" | "todos" | "memory" | "artifacts";

type OverviewResponse = {
  sessions: ChatSession[];
  todos: TodoItem[];
  memory: MemoryPayload["overview"];
  repos: RepoEntry[];
};

type BackendStatus = {
  web: string;
  llm_url: string;
  llm_reachable: boolean;
  detail: string;
};

type Toast = { id: number; message: string; kind: "error" | "info" };

function updateRunEvents(
  events: RunEvent[],
  next: RunEvent,
  matcher?: (event: RunEvent) => boolean,
): RunEvent[] {
  const nextEvents = [...events];
  const matchIndex = matcher ? nextEvents.findIndex(matcher) : -1;
  if (matchIndex >= 0) {
    nextEvents[matchIndex] = { ...nextEvents[matchIndex], ...next, id: nextEvents[matchIndex].id };
    return nextEvents.slice(0, 18);
  }
  return [next, ...nextEvents].slice(0, 18);
}

export default function App() {
  const streamRef = useRef<AbortController | null>(null);
  const messagesRef = useRef<ChatMessage[]>([]);
  const composerFormRef = useRef<HTMLFormElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [streamingAssistant, setStreamingAssistant] = useState("");
  const [streamingThinking, setStreamingThinking] = useState<string[]>([]);
  const [activity, setActivity] = useState<RunEvent[]>([]);
  const [streamError, setStreamError] = useState("");
  const [backendStatus, setBackendStatus] = useState<BackendStatus | null>(null);
  const [todos, setTodos] = useState<TodoItem[]>([]);
  const [todoDraft, setTodoDraft] = useState("");
  const [memoryData, setMemoryData] = useState<MemoryPayload | null>(null);
  const [memoryQuery, setMemoryQuery] = useState("");
  const [memoryType, setMemoryType] = useState("");
  const [repos, setRepos] = useState<RepoEntry[]>([]);
  const [selectedRepoId, setSelectedRepoId] = useState("");
  const [graph, setGraph] = useState<GraphPayload>({ nodes: [], edges: [] });
  const [ragQuery, setRagQuery] = useState("");
  const [ragLoading, setRagLoading] = useState(false);
  const [ragResults, setRagResults] = useState<RagResult[]>([]);
  const [ragError, setRagError] = useState("");
  const [activePane, setActivePane] = useState<Pane>("graph");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    try { return localStorage.getItem("logician_sidebar") === "closed"; } catch { return false; }
  });
  const [sessionFilter, setSessionFilter] = useState("");
  const [streamingTools, setStreamingTools] = useState<ToolCallItem[]>([]);
  const [waitingFirstToken, setWaitingFirstToken] = useState(false);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [activeArtifact, setActiveArtifact] = useState<Artifact | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [editingTitle, setEditingTitle] = useState<string | null>(null);
  const [titleDraft, setTitleDraft] = useState("");
  const [editTarget, setEditTarget] = useState<{ index: number; content: string } | null>(null);

  const deferredMemoryQuery = useDeferredValue(memoryQuery);
  const messagesAtBottom = useRef(true);

  const selectedRepo = repos.find((repo) => repo.id === selectedRepoId) ?? null;
  const selectedSession =
    sessions.find((session) => session.id === selectedSessionId) ?? null;
  const displayMessages = useMemo(() => buildDisplayMessages(messages), [messages]);
  const filteredSessions = useMemo(() => {
    const q = sessionFilter.trim().toLowerCase();
    if (!q) return sessions;
    return sessions.filter((s) => s.title.toLowerCase().includes(q) || s.preview.toLowerCase().includes(q));
  }, [sessions, sessionFilter]);
  const groupedSessions = useMemo(() => groupSessionsByAge(filteredSessions), [filteredSessions]);
  const liveTimeline = useMemo(() => activity.slice(0, 8), [activity]);
  const hasLiveRunPanel = sending && Boolean(
    streamingAssistant ||
    streamingThinking.length > 0 ||
    streamingTools.length > 0 ||
    activity.length > 0,
  );
  const graphStats = useMemo(() => {
    const highlighted = graph.nodes.filter((node) => node.highlight).length;
    return {
      nodes: graph.nodes.length,
      edges: graph.edges.length,
      highlighted,
    };
  }, [graph]);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    try {
      localStorage.setItem("logician_sidebar", sidebarCollapsed ? "closed" : "open");
    } catch {
      // ignore persistence failures
    }
  }, [sidebarCollapsed]);

  useEffect(() => {
    return () => {
      streamRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (messagesAtBottom.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, streamingAssistant, streamingThinking, streamingTools, activity]);

  const handleScrollContainer = useCallback((event: UIEvent<HTMLDivElement>) => {
    const el = event.currentTarget;
    messagesAtBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  }, []);

  const scrollToBottom = useCallback(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      messagesAtBottom.current = true;
    }
  }, []);

  const abortStream = useCallback(() => {
    streamRef.current?.abort();
    streamRef.current = null;
    setSending(false);
    setWaitingFirstToken(false);
    setStreamingAssistant("");
    setStreamingThinking([]);
    setStreamingTools([]);
    setActivity((current) =>
      current.length === 0
        ? current
        : updateRunEvents(current, {
          id: "stream-stopped",
          kind: "status",
          label: "Stopped",
          detail: "Streaming cancelled",
          state: "error",
        }),
    );
  }, []);

  const showToast = useCallback((message: string, kind: "error" | "info" = "info") => {
    const id = Date.now();
    setToasts((current) => [...current, { id, message, kind }]);
    window.setTimeout(() => {
      setToasts((current) => current.filter((t) => t.id !== id));
    }, 4500);
  }, []);

  const dismissToast = useCallback((id: number) => {
    setToasts((current) => current.filter((t) => t.id !== id));
  }, []);

  const {
    attachments,
    setAttachments,
    attachLoading,
    isDragging,
    clearAttachments,
    handleFileSelect,
    handleDragOver,
    handleDragLeave,
    handleDrop,
  } = useAttachments({ showToast });

  const resetDraftSession = useCallback(() => {
    streamRef.current?.abort();
    streamRef.current = null;
    setSending(false);
    setSelectedSessionId(null);
    setMessages([]);
    setStreamingAssistant("");
    setStreamingThinking([]);
    setStreamingTools([]);
    setActivity([]);
    setStreamError("");
    setEditTarget(null);
    setArtifacts([]);
    setActiveArtifact(null);
    setInput("");
    clearAttachments();
    window.setTimeout(() => textareaRef.current?.focus(), 30);
  }, [clearAttachments]);

  useEffect(() => {
    function handleGlobalKey(e: KeyboardEvent) {
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        resetDraftSession();
      }
    }
    document.addEventListener("keydown", handleGlobalKey);
    return () => document.removeEventListener("keydown", handleGlobalKey);
  }, [resetDraftSession]);

  const addArtifact = useCallback((content: string, language: string) => {
    const kindMap: Record<string, Artifact["kind"]> = {
      html: "html", svg: "svg", mermaid: "mermaid",
    };
    const kind = kindMap[language] ?? "code";
    const lineCount = content.split("\n").length;
    const title = language
      ? `${language} snippet (${lineCount} lines)`
      : `Code snippet (${lineCount} lines)`;
    const artifact: Artifact = {
      id: `artifact-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      kind,
      language,
      content,
      title,
    };
    setArtifacts((current) => [artifact, ...current]);
    setActiveArtifact(artifact);
    setActivePane("artifacts");
  }, []);

  const handleRenameSession = useCallback(async (sessionId: string, draft: string) => {
    setEditingTitle(null);
    const trimmed = draft.trim();
    if (!sessionId || !trimmed) return;
    try {
      await api(`/api/sessions/${sessionId}`, {
        method: "PATCH",
        body: JSON.stringify({ title: trimmed }),
      });
      setSessions((current) =>
        current.map((s) => (s.id === sessionId ? { ...s, title: trimmed } : s)),
      );
    } catch {
      showToast("Failed to rename session.", "error");
    }
  }, [showToast]);

  useEffect(() => {
    async function bootstrap() {
      const overview = await api<OverviewResponse>("/api/overview");
      const status = await api<BackendStatus>("/api/backend/status");
      setSessions(overview.sessions);
      setBackendStatus(status);
      setTodos(overview.todos);
      setRepos(overview.repos);
      if (overview.sessions.length > 0) {
        const initialSessionId = overview.sessions[0].id;
        setSelectedSessionId(initialSessionId);
        const session = await api<{
          id: string;
          messages: ChatMessage[];
          todos: TodoItem[];
        }>(`/api/sessions/${initialSessionId}`);
        setMessages(session.messages);
        if (session.todos.length > 0) {
          setTodos(session.todos);
        }
      }
      const memory = await api<MemoryPayload>("/api/memory");
      setMemoryData(memory);
    }

    void bootstrap();
  }, []);

  useEffect(() => {
    if (selectedRepoId && !repos.some((repo) => repo.id === selectedRepoId)) {
      setSelectedRepoId("");
    }
  }, [repos, selectedRepoId]);

  useEffect(() => {
    if (!selectedRepoId) {
      setGraph({ nodes: [], edges: [] });
      return;
    }
    async function loadGraph() {
      const nextGraph = await api<GraphPayload>(`/api/repos/${selectedRepoId}/graph`);
      setGraph(nextGraph);
    }
    void loadGraph();
  }, [selectedRepoId]);

  useEffect(() => {
    async function loadMemory() {
      const params = new URLSearchParams();
      if (deferredMemoryQuery) {
        params.set("query", deferredMemoryQuery);
      }
      if (memoryType) {
        params.set("obs_type", memoryType);
      }
      const nextMemory = await api<MemoryPayload>(`/api/memory?${params.toString()}`);
      setMemoryData(nextMemory);
    }
    void loadMemory();
  }, [deferredMemoryQuery, memoryType]);

  const handleEditSubmit = useCallback(async (index: number, newContent: string) => {
    if (!selectedSessionId || !newContent.trim() || sending) return;
    const target = messages[index];
    if (!target) return;

    const truncated = messages.slice(0, index);
    setMessages(truncated);
    setEditTarget(null);

    if (target.id !== undefined) {
      try {
        await api(`/api/sessions/${selectedSessionId}/truncate`, {
          method: "POST",
          body: JSON.stringify({ at_id: target.id }),
        });
      } catch {
        showToast("Could not truncate history; regenerating from scratch.", "error");
      }
    }

    await handleSend(undefined, newContent.trim());
  }, [selectedSessionId, sending, showToast, messages]);

  const refreshSessions = useCallback(async () => {
    const next = await api<{ sessions: ChatSession[] }>("/api/sessions");
    setSessions(next.sessions);
  }, []);

  const refreshTodos = useCallback(async () => {
    const next = await api<{ todos: TodoItem[] }>("/api/todos");
    setTodos(next.todos);
  }, []);

  const recordActivity = useCallback(
    (item: RunEvent, matcher?: (event: RunEvent) => boolean) => {
      setActivity((current) => updateRunEvents(current, item, matcher));
    },
    [],
  );

  const clearStreamingState = useCallback(() => {
    setStreamingAssistant("");
    setStreamingThinking([]);
    setStreamingTools([]);
    setWaitingFirstToken(false);
  }, []);

  const finalizeFromSession = useCallback(async (sessionId: string | null, reason: string) => {
    if (!sessionId) {
      showToast(reason, "error");
      setStreamError(reason);
      setSending(false);
      return;
    }
    try {
      const detail = await api<{
        id: string;
        messages: ChatMessage[];
        todos: TodoItem[];
      }>(`/api/sessions/${sessionId}`);
      setSelectedSessionId(sessionId);
      setMessages(detail.messages);
      if (detail.todos.length > 0) {
        setTodos(detail.todos);
      }
      clearStreamingState();
      setStreamError("");
      await refreshSessions();
    } catch (error) {
      const msg = error instanceof Error ? error.message : reason;
      showToast(msg, "error");
      setStreamError(msg);
    } finally {
      setSending(false);
    }
  }, [showToast, clearStreamingState, refreshSessions]);

  const handleSessionSelect = useCallback(async (sessionId: string) => {
    streamRef.current?.abort();
    streamRef.current = null;
    setSending(false);
    startTransition(() => setSelectedSessionId(sessionId));
    setStreamingAssistant("");
    setStreamingThinking([]);
    setStreamingTools([]);
    setActivity([]);
    setStreamError("");
    setArtifacts([]);
    setActiveArtifact(null);
    const detail = await api<{
      messages: ChatMessage[];
      todos: TodoItem[];
    }>(`/api/sessions/${sessionId}`);
    setMessages(detail.messages);
    if (detail.todos.length > 0) {
      setTodos(detail.todos);
    }
    window.setTimeout(() => textareaRef.current?.focus(), 50);
  }, []);

  const handleSend = useCallback(async (event?: FormEvent, messageOverride?: string) => {
    if (event) event.preventDefault();
    const rawText = (messageOverride ?? input).trim();
    const message = messageOverride
      ? rawText
      : buildMessageWithAttachments(rawText, attachments);
    if (!message || sending) {
      return;
    }

    setSending(true);
    setWaitingFirstToken(true);
    setEditTarget(null);
    setStreamError("");
    if (!messageOverride) {
      setInput("");
      setAttachments([]);
      if (textareaRef.current) {
        textareaRef.current.style.height = "";
      }
    }
    setStreamingAssistant("");
    setStreamingThinking([]);
    setStreamingTools([]);
    setActivity([]);
    setMessages((current) => [
      ...current,
      {
        role: "user",
        content: message,
        timestamp: new Date().toISOString(),
      },
    ]);

    streamRef.current?.abort();
    const controller = new AbortController();
    streamRef.current = controller;
    let streamCompleted = false;
    let sawStreamingProgress = false;
    let streamSessionId = selectedSessionId;
    let streamedAssistantText = "";
    let streamedThinkingEntries: string[] = [];
    let streamedToolCalls: ToolCallItem[] = [];

    const markStreamingProgress = () => {
      sawStreamingProgress = true;
      setWaitingFirstToken(false);
    };

    try {
      await streamSse(
        "/api/chat/stream",
        {
          method: "POST",
          body: JSON.stringify({
            message,
            session_id: selectedSessionId,
            fresh_session: !selectedSessionId,
            repo_id: selectedRepoId || undefined,
          }),
          signal: controller.signal,
        },
        (eventName, data) => {
          if (eventName === "session") {
            const payload = data as { session_id?: string };
            const nextSessionId = String(payload.session_id || "").trim();
            if (nextSessionId) {
              streamSessionId = nextSessionId;
            }
            return;
          }

          if (eventName === "token") {
            const payload = data as { text?: string };
            markStreamingProgress();
            streamedAssistantText = String(payload.text || "");
            setStreamingAssistant(streamedAssistantText);
            recordActivity(
              {
                id: "answer-stream",
                kind: "status",
                label: "Answer",
                detail: "Streaming assistant response",
                state: "live",
              },
              (item) => item.kind === "status" && item.label === "Answer",
            );
            return;
          }

          if (eventName === "thinking") {
            const payload = data as { content?: string };
            const content = String(payload.content || "").trim();
            if (!content) {
              return;
            }
            markStreamingProgress();
            streamedThinkingEntries = [...streamedThinkingEntries, content];
            setStreamingThinking((current) => [...current, content]);
            recordActivity({
              id: `thinking-${Date.now()}-${streamedThinkingEntries.length}`,
              kind: "thinking",
              label: "Thinking",
              detail: compactText(content),
              state: "live",
            });
            return;
          }

          if (eventName === "token_reset") {
            streamedAssistantText = "";
            setStreamingAssistant("");
            return;
          }

          if (eventName === "tool") {
            const payload = data as {
              tool?: string;
              arguments?: Record<string, unknown>;
              meta?: {
                stage?: string;
                status?: "ok" | "error";
                duration_ms?: number;
                cache_hit?: boolean;
                result_preview?: string;
                result_output?: string;
                error?: string;
              };
            };
            const toolName = String(payload.tool || "").trim();
            if (!toolName) {
              return;
            }
            markStreamingProgress();
            const stage = String(payload.meta?.stage || "");
            if (stage === "end") {
              let lastIdx = -1;
              for (let i = streamedToolCalls.length - 1; i >= 0; i -= 1) {
                if (streamedToolCalls[i].tool === toolName && streamedToolCalls[i].status === "running") {
                  lastIdx = i;
                  break;
                }
              }
              const response = compactText(
                String(
                  payload.meta?.error ||
                  payload.meta?.result_output ||
                  payload.meta?.result_preview ||
                  "",
                ),
                220,
              );
              if (lastIdx >= 0) {
                streamedToolCalls = streamedToolCalls.map((item, idx) =>
                  idx === lastIdx
                    ? {
                      ...item,
                      response: response || undefined,
                      status: payload.meta?.status === "error" ? "error" : "ok",
                      duration_ms: payload.meta?.duration_ms,
                      cache_hit: payload.meta?.cache_hit,
                    }
                    : item,
                );
                setStreamingTools([...streamedToolCalls]);
              }
              recordActivity(
                {
                  id: `tool-finish-${toolName}`,
                  kind: "tool",
                  label: toolName,
                  detail: response || "Completed",
                  meta: [
                    payload.meta?.cache_hit ? "cached" : "",
                    payload.meta?.duration_ms !== undefined ? `${payload.meta.duration_ms}ms` : "",
                  ].filter(Boolean).join(" · "),
                  state: payload.meta?.status === "error" ? "error" : "done",
                },
                (item) => item.kind === "tool" && item.label === toolName && item.state === "live",
              );
              return;
            }

            const toolId = `tool-${Date.now()}-${toolName}`;
            const toolItem: ToolCallItem = {
              id: toolId,
              tool: toolName,
              args: payload.arguments,
              status: "running",
            };
            streamedToolCalls = [...streamedToolCalls, toolItem];
            setStreamingTools([...streamedToolCalls]);
            recordActivity({
              id: toolId,
              kind: "tool",
              label: toolName,
              detail: compactText(JSON.stringify(payload.arguments || {}), 120),
              state: "live",
            });
            return;
          }

          if (eventName === "graph") {
            const payload = data as GraphPayload;
            setGraph(payload);
            return;
          }

          if (eventName === "agent_error") {
            const payload = data as { message?: string };
            const errMsg = String(payload.message || "Streaming failed");
            showToast(errMsg, "error");
            setStreamError(errMsg);
            recordActivity({
              id: `error-${Date.now()}`,
              kind: "status",
              label: "Stream error",
              detail: compactText(errMsg),
              state: "error",
            });
            clearStreamingState();
            setSending(false);
            streamCompleted = true;
            return;
          }

          if (eventName === "done") {
            const payload = data as {
              session_id: string;
              assistant_message?: string;
              messages: ChatMessage[];
              todos: TodoItem[];
              thinking_log: string[];
            };
            streamCompleted = true;
            streamSessionId = payload.session_id;
            setSelectedSessionId(payload.session_id);
            setMessages(payload.messages);
            setTodos(payload.todos);
            clearStreamingState();
            setSending(false);
            recordActivity(
              {
                id: "answer-stream",
                kind: "status",
                label: "Answer",
                detail: "Completed",
                state: "done",
              },
              (item) => item.kind === "status" && item.label === "Answer",
            );
            void refreshSessions();
          }
        },
      );

      if (controller.signal.aborted) {
        return;
      }
      if (streamCompleted) {
        return;
      }
      if (sawStreamingProgress) {
        await finalizeFromSession(
          streamSessionId,
          "Stream closed before the final event landed, recovered from session state.",
        );
        return;
      }
      const errorMessage = "Streaming channel closed before any events arrived.";
      showToast(errorMessage, "error");
      setStreamError(errorMessage);
      clearStreamingState();
      setSending(false);
    } catch (error) {
      if (controller.signal.aborted) {
        return;
      }
      if (sawStreamingProgress) {
        await finalizeFromSession(streamSessionId, "Stream interrupted, recovered from session state.");
        return;
      }
      const msg = error instanceof Error ? error.message : "Streaming request failed.";
      showToast(msg, "error");
      setStreamError(msg);
      clearStreamingState();
      setSending(false);
    } finally {
      if (streamRef.current === controller) {
        streamRef.current = null;
      }
    }
  }, [
    input,
    attachments,
    sending,
    selectedSessionId,
    selectedRepoId,
    recordActivity,
    showToast,
    clearStreamingState,
    finalizeFromSession,
    refreshSessions,
  ]);

  const handleAddTodo = useCallback(async (event: FormEvent) => {
    event.preventDefault();
    if (!todoDraft.trim()) {
      return;
    }
    const payload = await api<{ todos: TodoItem[] }>("/api/todos", {
      method: "POST",
      body: JSON.stringify({
        command: "add",
        title: todoDraft.trim(),
      }),
    });
    setTodos(payload.todos);
    setTodoDraft("");
  }, [todoDraft]);

  const updateTodo = useCallback(async (id: number, status: TodoItem["status"]) => {
    const payload = await api<{ todos: TodoItem[] }>("/api/todos", {
      method: "POST",
      body: JSON.stringify({
        command: "mark",
        id,
        status,
      }),
    });
    setTodos(payload.todos);
  }, []);

  const runRagSearch = useCallback(async (query: string) => {
    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      return;
    }

    setRagLoading(true);
    setRagError("");
    setRagQuery(trimmedQuery);
    try {
      const payload = await api<{ results: RagResult[]; graph: GraphPayload }>("/api/rag/search", {
        method: "POST",
        body: JSON.stringify({
          query: trimmedQuery,
          repo_id: selectedRepoId || undefined,
          n_results: 8,
        }),
      });
      setRagResults(payload.results);
      if (payload.graph.nodes.length > 0) {
        setGraph(payload.graph);
      }
      startTransition(() => setActivePane("graph"));
    } catch (error) {
      setRagError(error instanceof Error ? error.message : "RAG search failed");
    } finally {
      setRagLoading(false);
    }
  }, [selectedRepoId]);

  return (
    <div className="workspace-shell mx-auto flex min-h-screen max-w-[1880px] flex-col gap-4 p-4 lg:p-6">
      <ToastStack toasts={toasts} onDismiss={dismissToast} />
      <AppHeader
        sessionsCount={sessions.length}
        todosCount={todos.length}
        memoryCount={memoryData?.overview.count ?? 0}
        repoName={selectedRepo?.name ?? "Global"}
        graphLabel={selectedRepo ? `${graphStats.highlighted} focused` : "inactive"}
        sessionTitle={selectedSession?.title ?? "Fresh thread"}
        backendStatus={backendStatus}
      />

      <div
        className={`grid flex-1 gap-4 ${sidebarCollapsed
            ? "xl:grid-cols-[64px_minmax(0,1fr)_390px]"
            : "xl:grid-cols-[280px_minmax(0,1fr)_390px]"
          }`}
      >
        <SessionSidebar
          sessions={sessions}
          selectedSessionId={selectedSessionId}
          repos={repos}
          selectedRepoId={selectedRepoId}
          sessionFilter={sessionFilter}
          setSessionFilter={setSessionFilter}
          onSessionSelect={handleSessionSelect}
          onRepoSelect={(id) => {
            startTransition(() => setSelectedRepoId(id));
            void refreshTodos();
          }}
          onRefresh={refreshSessions}
          sidebarCollapsed={sidebarCollapsed}
          setSidebarCollapsed={setSidebarCollapsed}
          filteredSessions={filteredSessions}
          groupedSessions={groupedSessions}
          editingTitle={editingTitle}
          titleDraft={titleDraft}
          setEditingTitle={setEditingTitle}
          titleDraftChanged={(draft) => setTitleDraft(draft)}
          handleRenameSession={handleRenameSession}
          selectedRepo={selectedRepo}
          onNewSession={resetDraftSession}
        />

        <ChatInterface
          displayMessages={displayMessages}
          streamingAssistant={streamingAssistant}
          streamingThinking={streamingThinking}
          streamingTools={streamingTools}
          activity={activity}
          artifacts={artifacts}
          activeArtifact={activeArtifact}
          atBottom={messagesAtBottom.current}
          sending={sending}
          waitingFirstToken={waitingFirstToken}
          hasLiveRunPanel={hasLiveRunPanel}
          liveTimeline={liveTimeline}
          editTarget={editTarget}
          setEditTarget={setEditTarget}
          attachments={attachments}
          setAttachments={setAttachments}
          attachLoading={attachLoading}
          isDragging={isDragging}
          input={input}
          setInput={setInput}
          scrollRef={scrollRef}
          composerFormRef={composerFormRef}
          textareaRef={textareaRef}
          fileInputRef={fileInputRef}
          handleScrollContainer={handleScrollContainer}
          scrollToBottom={scrollToBottom}
          abortStream={abortStream}
          addArtifact={addArtifact}
          handleEditSubmit={handleEditSubmit}
          handleFileSelect={handleFileSelect}
          handleDragOver={handleDragOver}
          handleDragLeave={handleDragLeave}
          handleDrop={handleDrop}
          onSendMessage={handleSend}
        />

        <UtilityRail
          activePane={activePane}
          setActivePane={setActivePane}
          repos={repos}
          selectedRepoId={selectedRepoId}
          setSelectedRepoId={setSelectedRepoId}
          graph={graph}
          ragQuery={ragQuery}
          setRagQuery={setRagQuery}
          ragLoading={ragLoading}
          ragResults={ragResults}
          ragError={ragError}
          runRagSearch={runRagSearch}
          todos={todos}
          todoDraft={todoDraft}
          setTodoDraft={setTodoDraft}
          handleAddTodo={handleAddTodo}
          updateTodo={updateTodo}
          memoryData={memoryData}
          memoryQuery={memoryQuery}
          setMemoryQuery={setMemoryQuery}
          memoryType={memoryType}
          setMemoryType={setMemoryType}
          graphStats={graphStats}
          artifacts={artifacts}
          activeArtifact={activeArtifact}
          setActiveArtifact={setActiveArtifact}
          liveTimeline={liveTimeline}
          streamingTools={streamingTools}
          streamingThinking={streamingThinking}
          sending={sending}
          waitingFirstToken={waitingFirstToken}
        />
      </div>
    </div>
  );
}
