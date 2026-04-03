import { useEffect, useState, useCallback } from "react";
import {
  type ChatMessage,
  type Artifact,
  type Attachment,
  type RunEvent,
} from "./types";
import {
  compactText,
  messageRowLayout,
  messageBubbleShape,
  roleTone,
  messageRoleLabel,
  formatTimestamp,
} from "./utils/messageFormatting";
import { MarkdownMessage } from "./components/MarkdownMessage";
import { ThinkingDisclosure } from "./components/ThinkingDisclosure";
import { ArtifactViewer } from "./components/ArtifactViewer";
import { ToolCallRenderer, type ToolCallItem } from "./components/ToolCallRenderer";

type Props = {
  displayMessages: ChatMessage[];
  streamingAssistant: string;
  streamingThinking: string[];
  streamingTools: ToolCallItem[];
  liveTimeline: RunEvent[];
  artifacts: Artifact[];
  activeArtifact: Artifact | null;
  atBottom: boolean;
  sending: boolean;
  waitingFirstToken: boolean;
  editTarget: { index: number; content: string } | null;
  setEditTarget: React.Dispatch<React.SetStateAction<{ index: number; content: string } | null>>;
  attachments: Attachment[];
  setAttachments: React.Dispatch<React.SetStateAction<Attachment[]>>;
  attachLoading: boolean;
  isDragging: boolean;
  input: string;
  setInput: (v: string) => void;
  scrollRef: React.RefObject<HTMLDivElement>;
  composerFormRef: React.RefObject<HTMLFormElement>;
  textareaRef: React.RefObject<HTMLTextAreaElement>;
  fileInputRef: React.RefObject<HTMLInputElement>;
  handleScrollContainer: (e: React.UIEvent<HTMLDivElement>) => void;
  scrollToBottom: () => void;
  abortStream: () => void;
  addArtifact: (content: string, language: string) => void;
  handleEditSubmit: (index: number, newContent: string) => Promise<void>;
  handleFileSelect: (event: React.ChangeEvent<HTMLInputElement>) => Promise<void>;
  handleDragOver: (event: React.DragEvent) => void;
  handleDragLeave: (event: React.DragEvent) => void;
  handleDrop: (event: React.DragEvent) => void;
  onSendMessage: (event?: React.FormEvent, messageOverride?: string) => Promise<void>;
};

function TypingIndicator() {
  return (
    <div className="typing-indicator">
      <span />
      <span />
      <span />
      <div className="ml-2 text-sm text-sand/70">Thinking</div>
    </div>
  );
}

function EmptyState({
  onPick,
}: {
  onPick: (prompt: string) => void;
}) {
  const suggestions = [
    {
      label: "Inspect repo",
      prompt: "Inspect this repository and summarize the current architecture and active work.",
    },
    {
      label: "Continue refactor",
      prompt: "Continue the current App.tsx refactor and suggest the next safe extraction step.",
    },
    {
      label: "Trace bug",
      prompt: "Trace the current frontend bug, identify the root cause, and propose the smallest fix.",
    },
    {
      label: "Plan next steps",
      prompt: "Look at the current work and propose the next concrete implementation steps.",
    },
  ];

  return (
    <div className="soft-panel rounded-[26px] border border-white/10 px-5 py-5">
      <div className="panel-title">Fresh Thread</div>
      <div className="mt-2 font-['Space_Grotesk'] text-2xl font-semibold tracking-[-0.04em] text-white">
        Keep the old workspace flow
      </div>
      <p className="mt-2 max-w-2xl text-sm leading-6 text-sand/55">
        Start with repo exploration, continue the refactor, or ask for a focused bug trace.
      </p>
      <div className="suggestion-grid mt-5">
        {suggestions.map((item) => (
          <button
            key={item.label}
            type="button"
            onClick={() => onPick(item.prompt)}
            className="suggestion-card"
          >
            <div className="suggestion-card-label">{item.label}</div>
            <div className="suggestion-card-prompt">{item.prompt}</div>
          </button>
        ))}
      </div>
    </div>
  );
}

function LiveStatusStrip({
  streamingAssistant,
  streamingThinking,
  streamingTools,
  liveTimeline,
  sending,
  waitingFirstToken,
}: {
  streamingAssistant: string;
  streamingThinking: string[];
  streamingTools: ToolCallItem[];
  liveTimeline: RunEvent[];
  sending: boolean;
  waitingFirstToken: boolean;
}) {
  const latestEvent = liveTimeline[liveTimeline.length - 1] ?? null;
  const runningTools = streamingTools.filter((item) => item.status === "running").length;
  const completedTools = streamingTools.filter((item) => item.status === "ok").length;
  const erroredTools = streamingTools.filter((item) => item.status === "error").length;

  const detail = waitingFirstToken
    ? "Waiting for the first token or tool event..."
    : latestEvent?.detail
      ? compactText(latestEvent.detail, 140)
      : sending
        ? "Preparing the next step..."
        : "Idle";

  const latestLabel = latestEvent?.label
    ? compactText(latestEvent.label, 28)
    : waitingFirstToken
      ? "boot"
      : sending
        ? "live"
        : "idle";

  return (
    <div className="live-status-strip">
      <div className="live-status-strip-row">
        <div className="flex items-center gap-2">
          {sending ? <span className="thinking-live-dot" aria-hidden="true" /> : null}
          <span className="live-status-strip-title">{latestLabel}</span>
        </div>
        <div className="live-status-strip-pills">
          {streamingThinking.length > 0 ? (
            <span className="live-status-pill">thinking {streamingThinking.length}</span>
          ) : null}
          {runningTools > 0 ? (
            <span className="live-status-pill live-status-pill-active">running {runningTools}</span>
          ) : null}
          {completedTools > 0 ? (
            <span className="live-status-pill">done {completedTools}</span>
          ) : null}
          {erroredTools > 0 ? (
            <span className="live-status-pill live-status-pill-error">error {erroredTools}</span>
          ) : null}
          {streamingAssistant ? (
            <span className="live-status-pill">answer</span>
          ) : null}
        </div>
      </div>
      <div className="live-status-strip-detail">{detail}</div>
    </div>
  );
}

function ComposerMetaRow({
  sending,
  waitingFirstToken,
  liveTimeline,
}: {
  sending: boolean;
  waitingFirstToken: boolean;
  liveTimeline: RunEvent[];
}) {
  const latestEvent = liveTimeline[liveTimeline.length - 1] ?? null;
  const statusText = waitingFirstToken
    ? "Waiting for first event"
    : latestEvent?.label
      ? latestEvent.detail
        ? `${latestEvent.label} · ${compactText(latestEvent.detail, 84)}`
        : latestEvent.label
      : sending
        ? "Streaming"
        : "Idle";

  return (
    <div className="composer-meta-row">
      <div className="composer-meta-hints">
        <span>Enter send</span>
        <span>Shift+Enter newline</span>
        <span>Drag files to attach</span>
      </div>
      <div className={`composer-meta-status ${sending ? "composer-meta-status-live" : ""}`}>
        {sending ? <span className="thinking-live-dot" aria-hidden="true" /> : null}
        <span>{statusText}</span>
      </div>
    </div>
  );
}

export function ChatInterface({
  displayMessages,
  streamingAssistant,
  streamingThinking,
  streamingTools,
  liveTimeline,
  artifacts,
  activeArtifact,
  atBottom,
  sending,
  waitingFirstToken,
  editTarget,
  setEditTarget,
  attachments,
  setAttachments,
  attachLoading,
  isDragging,
  input,
  setInput,
  scrollRef,
  composerFormRef,
  textareaRef,
  fileInputRef,
  handleScrollContainer,
  scrollToBottom,
  abortStream,
  addArtifact,
  handleEditSubmit,
  handleFileSelect,
  handleDragOver,
  handleDragLeave,
  handleDrop,
  onSendMessage,
}: Props) {
  const [composerHeight, setComposerHeight] = useState("auto");
  const hasStreamingBubble =
    streamingAssistant.length > 0 ||
    streamingThinking.length > 0 ||
    streamingTools.length > 0;

  useEffect(() => {
    if (!textareaRef.current) return;
    textareaRef.current.style.height = composerHeight;
    const scrollHeight = textareaRef.current.scrollHeight;
    textareaRef.current.style.height = scrollHeight + "px";
    setComposerHeight(scrollHeight + "px");
  }, [input, textareaRef, composerHeight]);

  const handleSend = useCallback(
    async (e?: React.FormEvent, messageOverride?: string) => {
      await onSendMessage(e as unknown as React.FormEvent, messageOverride);
    },
    [onSendMessage]
  );

  const handleAttachFiles = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files.length > 0) {
        await handleFileSelect(e);
      }
      e.target.value = "";
    },
    [handleFileSelect]
  );

  return (
    <div className="conversation-shell glass-card flex h-full flex-col overflow-hidden">
      {activeArtifact && (
        <div className="border-b border-white/10 bg-white/5 px-4 py-3">
          <ArtifactViewer artifact={activeArtifact} onClose={() => { }} />
        </div>
      )}

      <div className="conversation-header flex items-center justify-between gap-3 border-b border-white/10 px-5 py-4">
        <div>
          <div className="panel-title">Conversation</div>
          <div className="mt-1 text-sm text-sand/50">
            {displayMessages.length} messages in view
          </div>
        </div>
        <div className={`agent-status-chip ${sending ? "agent-status-active" : "agent-status-idle"}`}>
          {sending ? <span className="agent-status-ring" /> : null}
          <span>{sending ? "Agent active" : "Idle"}</span>
        </div>
      </div>

      <div
        ref={scrollRef}
        onScroll={handleScrollContainer}
        className="chat-scroll relative flex-1 overflow-y-auto px-4 py-6"
      >
        <div className="mx-auto max-w-3xl space-y-4">
          {displayMessages.length === 0 && !streamingAssistant && !waitingFirstToken ? (
            <EmptyState onPick={(prompt) => void onSendMessage(undefined, prompt)} />
          ) : null}

          {displayMessages.map((message, index) => (
            <div
              key={`${message.id ?? index}-${message.timestamp ?? index}`}
              className={`${messageRowLayout(message.role)}`}
            >
              <div className="w-full">
                <div
                  className={`group relative rounded-[28px] border ${roleTone(message.role)} ${messageBubbleShape(message.role)} px-6 py-5 transition hover:border-white/20`}
                >
                  <div className="flex items-start gap-4">
                    <div className="shrink-0">
                      <div
                        className={`msg-avatar ${message.role === "user" ? "msg-avatar-user" : "msg-avatar-assistant"
                          }`}
                      >
                        {messageRoleLabel(message.role).charAt(0).toUpperCase()}
                      </div>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="mb-2 flex items-center justify-between">
                        <span className="text-sm font-semibold text-sand/80">
                          {messageRoleLabel(message.role)}
                        </span>
                        <span className="text-xs text-sand/50">
                          {formatTimestamp(message.timestamp)}
                        </span>
                      </div>
                      {message.thinking_log?.length ? (
                        <ThinkingDisclosure entries={message.thinking_log} />
                      ) : null}
                      <ToolCallRenderer items={message.tool_calls ?? []} />
                      <MarkdownMessage content={message.content || ""} streaming={false} />
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ))}

          {hasStreamingBubble && (
            <div className="flex items-start justify-start">
              <div className="w-full">
                <div className={`rounded-[28px] border ${roleTone("assistant")} ${messageBubbleShape("assistant")} px-6 py-5`}>
                  <div className="flex items-start gap-4">
                    <div className="shrink-0">
                      <div className="msg-avatar msg-avatar-assistant">
                        A
                      </div>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="mb-2 flex items-center justify-between">
                        <span className="text-sm font-semibold text-sand/80">assistant</span>
                        <span className="text-xs text-sand/50">
                          {formatTimestamp(new Date().toISOString())}
                        </span>
                      </div>
                      <LiveStatusStrip
                        streamingAssistant={streamingAssistant}
                        streamingThinking={streamingThinking}
                        streamingTools={streamingTools}
                        liveTimeline={liveTimeline}
                        sending={sending}
                        waitingFirstToken={waitingFirstToken}
                      />
                      {streamingThinking.length > 0 && (
                        <ThinkingDisclosure entries={streamingThinking} streaming />
                      )}
                      <ToolCallRenderer items={streamingTools} streaming />
                      {streamingAssistant && (
                        <MarkdownMessage content={streamingAssistant} streaming />
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {waitingFirstToken && <TypingIndicator />}
        </div>
        {!atBottom && (
          <button
            type="button"
            onClick={scrollToBottom}
            className="scroll-fab"
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
          </button>
        )}
      </div>

      <div className="border-t border-white/10 bg-white/5 px-4 py-4">
        <div className="mx-auto max-w-3xl">
          <form
            ref={composerFormRef}
            onSubmit={(e) => handleSend(e)}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`composer-shell mt-4 flex flex-col gap-3 rounded-[24px] border p-3 transition ${isDragging
                ? "border-tide/50 bg-tide/10"
                : ""
              }`}
          >
            {attachments.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {attachments.map((att) => (
                  <div key={att.id} className="attachment-chip">
                    <span className="attachment-chip-icon">{att.kind}</span>
                    <span className="attachment-chip-name">{att.name}</span>
                    <button
                      type="button"
                      onClick={() => {
                        setAttachments((prev) => prev.filter((a) => a.id !== att.id));
                      }}
                      className="attachment-chip-remove"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}

            {attachLoading ? (
              <div className="text-xs uppercase tracking-[0.16em] text-sand/40">
                Processing attachments...
              </div>
            ) : null}

            <ComposerMetaRow
              sending={sending}
              waitingFirstToken={waitingFirstToken}
              liveTimeline={liveTimeline}
            />

            <div className="flex items-end gap-3">
              <label
                htmlFor="file-attach"
                className="attach-btn flex cursor-pointer"
                title="Attach files"
              >
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"
                  />
                </svg>
                <input
                  id="file-attach"
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  multiple
                  onChange={handleAttachFiles}
                />
              </label>

              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Message logician..."
                rows={1}
                className="field max-h-32 flex-1 resize-none !rounded-[22px] !bg-transparent px-2 py-2 text-[15px] leading-7"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
              />

              {sending ? (
                <button
                  type="button"
                  onClick={abortStream}
                  className="action-button !rounded-2xl"
                >
                  <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Abort
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!input.trim() && attachments.length === 0}
                  className="action-button-primary !rounded-2xl disabled:opacity-50"
                >
                  <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                  </svg>
                  Send
                </button>
              )}
            </div>

          </form>
        </div>
      </div>
    </div>
  );
}

export default ChatInterface;
