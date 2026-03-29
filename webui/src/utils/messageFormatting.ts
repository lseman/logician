import { type ChatSession } from "../types";

type SessionGroup = {
  label: string;
  sessions: ChatSession[];
};

export function normalizeThinkingLog(entries: string[]): string[] {
  return entries
    .map((entry) => String(entry || "").trim())
    .filter((entry) => entry.length > 0);
}

export function compactText(value: string, limit = 160): string {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (!text) {
    return "";
  }
  if (text.length <= limit) {
    return text;
  }
  return `${text.slice(0, limit).trimEnd()}...`;
}

function parseTimestamp(value?: string): Date | null {
  if (!value) {
    return null;
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return null;
  }
  return date;
}

export function formatTimestamp(value?: string): string | undefined {
  const date = parseTimestamp(value);
  if (!date) {
    return value;
  }
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatSessionStamp(value?: string): string {
  const date = parseTimestamp(value);
  if (!date) {
    return "";
  }
  const diffMs = Date.now() - date.getTime();
  const dayMs = 24 * 60 * 60 * 1000;
  if (diffMs < dayMs) {
    return date.toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  }
  if (diffMs < dayMs * 7) {
    return date.toLocaleDateString([], { weekday: "short" });
  }
  return date.toLocaleDateString([], {
    month: "short",
    day: "numeric",
  });
}

export function groupSessionsByAge(sessions: ChatSession[]): SessionGroup[] {
  const now = Date.now();
  const dayMs = 24 * 60 * 60 * 1000;
  const groups: SessionGroup[] = [
    { label: "Last 24 hours", sessions: [] },
    { label: "Last 7 days", sessions: [] },
    { label: "Last 30 days", sessions: [] },
    { label: "Older", sessions: [] },
  ];

  for (const session of sessions) {
    const parsed = parseTimestamp(session.last_updated);
    if (!parsed) {
      groups[3].sessions.push(session);
      continue;
    }
    const diffMs = Math.max(0, now - parsed.getTime());
    if (diffMs < dayMs) {
      groups[0].sessions.push(session);
    } else if (diffMs < dayMs * 7) {
      groups[1].sessions.push(session);
    } else if (diffMs < dayMs * 30) {
      groups[2].sessions.push(session);
    } else {
      groups[3].sessions.push(session);
    }
  }

  return groups.filter((group) => group.sessions.length > 0);
}

export function roleTone(role: string): string {
  if (role === "user") {
    return "border-amber-300/20 bg-[linear-gradient(180deg,rgba(210,163,93,0.14),rgba(210,163,93,0.05))]";
  }
  if (role === "assistant") {
    return "border-cyan-300/20 bg-[linear-gradient(180deg,rgba(103,183,200,0.14),rgba(103,183,200,0.05))]";
  }
  if (role === "tool") {
    return "border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.045),rgba(255,255,255,0.02))]";
  }
  return "border-white/10 bg-black/20";
}

export function messageRowLayout(role: string): string {
  return role === "user" ? "justify-end" : "justify-start";
}

function messageBubbleWidth(role: string): string {
  return role === "tool" ? "max-w-[88%] xl:max-w-4xl" : "max-w-[82%] xl:max-w-3xl";
}

export function messageBubbleShape(role: string): string {
  if (role === "user") {
    return "rounded-[28px_28px_10px_28px]";
  }
  return "rounded-[28px_28px_28px_10px]";
}

export function messageRoleLabel(role: string): string {
  return role === "user" ? "you" : role;
}

const TEXT_EXTENSIONS = new Set([
  "py","js","ts","tsx","jsx","java","cpp","c","h","cc","hh","cs","rs","go",
  "rb","php","sh","bash","zsh","fish","sql","md","txt","json","yaml","yml",
  "toml","ini","cfg","env","csv","html","htm","css","scss","xml","log","lock",
  "dockerfile","makefile","gitignore","editorconfig","prettierrc","eslintrc",
]);

export function langFromFilename(name: string): string {
  const ext = (name.split(".").pop() ?? "").toLowerCase();
  const map: Record<string, string> = {
    py: "python", js: "javascript", ts: "typescript", tsx: "tsx", jsx: "jsx",
    java: "java", cpp: "cpp", cc: "cpp", c: "c", h: "c", cs: "csharp",
    rs: "rust", go: "go", rb: "ruby", php: "php", sh: "bash", bash: "bash",
    zsh: "bash", fish: "bash", sql: "sql", md: "markdown", json: "json",
    yaml: "yaml", yml: "yaml", toml: "toml", html: "html", htm: "html",
    css: "css", scss: "scss", xml: "xml",
  };
  return map[ext] ?? ext;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)}MB`;
}

export function buildMessageWithAttachments(text: string, attachments: { kind: string; name: string; language?: string; content: string }[]): string {
  if (attachments.length === 0) return text;
  const parts = attachments.map((att) => {
    if (att.kind === "image") {
      return `[Image: ${att.name}]`;
    }
    const fence = att.language ? `\`\`\`${att.language}` : "```";
    return `**File: ${att.name}**\n${fence}\n${att.content}\n\`\`\``;
  });
  return [...parts, text].filter(Boolean).join("\n\n");
}
