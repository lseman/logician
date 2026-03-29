export type ChatMessage = {
  id?: number;
  role: string;
  content: string;
  name?: string;
  tool_call_id?: string;
  timestamp?: string;
  thinking_log?: string[];
  tool_calls?: import("./components/ToolCallCard").ToolCallItem[];
};

export type ChatSession = {
  id: string;
  last_updated: string;
  message_count: number;
  title: string;
  preview: string;
  last_role: string;
};

export type TodoItem = {
  id: number;
  title: string;
  status: "not-started" | "in-progress" | "completed" | "blocked";
  note?: string;
};

export type MemoryObservation = {
  id: number;
  session: string;
  type: string;
  emoji: string;
  timestamp: string;
  title: string;
  preview: string;
  content: string;
  files: string[];
  path: string;
};

export type MemoryFact = {
  name: string;
  type: string;
  description: string;
  path: string;
};

export type RepoEntry = {
  id: string;
  name: string;
  path: string;
  graph_nodes: number;
  graph_edges: number;
  chunks_added: number;
  git?: {
    branch?: string;
    commit?: string;
    remote?: string;
  };
};

export type GraphNode = {
  id: string;
  label: string;
  kind: string;
  rel_path?: string;
  symbol_kind?: string;
  line?: number;
  highlight?: boolean;
};

export type GraphEdge = {
  source: string;
  target: string;
  kind: string;
};

export type GraphPayload = {
  repo?: RepoEntry;
  focus_paths?: string[];
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type RagResult = {
  content: string;
  source: string;
  rel_path: string;
  metadata: Record<string, string | number | boolean | null>;
};

export type Attachment = {
  id: string;
  name: string;
  kind: "text" | "pdf" | "image";
  content: string;
  language: string;
  size: number;
};

export type MemoryPayload = {
  overview: {
    count: number;
    types: Record<string, number>;
    summary_markdown: string;
  };
  facts: MemoryFact[];
  observations: MemoryObservation[];
};

export type Artifact = {
  id: string;
  kind: "html" | "svg" | "mermaid" | "code";
  language: string;
  content: string;
  title: string;
};

export type RunEvent = {
  id: string;
  kind: "thinking" | "tool" | "status";
  label: string;
  detail: string;
  state?: "live" | "done" | "error";
  meta?: string;
};

export type ToolCallItem = {
  id: string;
  tool: string;
  args?: Record<string, unknown>;
  response?: string;
};

export type Pane = "chat" | "agents" | "tools" | "memory";
