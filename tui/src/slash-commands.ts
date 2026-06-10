// ── Slash command definitions ─────────────────────────────────────────────────
// ~30 commands matching Rust TUI, with usage patterns and dispatch types.

export type SlashDispatch = "local" | "bridge" | "state" | "quit";

export interface SlashCommandDef {
  command: string; // e.g. "/help"
  usage: string; // e.g. "/help [topic]"
  description: string;
  dispatch: SlashDispatch;
  acceptsArgs: boolean;
  handler?: (args: string) => string | undefined; // returns message for local-only commands
  bridgeHandler?: (args: string) => void; // sends to bridge
}

// ── Command spec factory ──────────────────────────────────────────────────────

function cmd(
  command: string,
  description: string,
  dispatch: SlashDispatch = "bridge",
  acceptsArgs = false,
  handler?: (args: string) => string | undefined,
  bridgeHandler?: (args: string) => void,
): SlashCommandDef {
  return {
    command,
    usage: command,
    description,
    dispatch,
    acceptsArgs,
    handler,
    bridgeHandler,
  };
}

// ── Full command list ─────────────────────────────────────────────────────────

export function createSlashCommands(
  _bridge: {
    sendSlash: (raw: string) => void;
    cancel: () => void;
    reset: () => void;
  },
  localHandlers: Record<string, (...args: unknown[]) => unknown>,
): SlashCommandDef[] {
  return [
    // ── Help & info ──────────────────────────────────────────────────────
    cmd("/help", "Show all available commands", "local", false, () => {
      return "Type / to see all commands. Prefix with / and Tab to autocomplete.";
    }),
    cmd("/?", "Alias for /help", "local", false, () => "/help"),

    // ── Session management ───────────────────────────────────────────────
    cmd("/new", "Start a new session", "bridge", false),
    cmd("/sessions", "List previous sessions", "bridge", true),
    cmd("/load", "Load a previous session (ID)", "bridge", true),
    cmd("/export", "Export chat history to file", "bridge", true),

    // ── Agent control ────────────────────────────────────────────────────
    cmd("/status", "Show runtime state snapshot", "state", false),
    cmd("/agents", "List loaded agents", "bridge", false),
    cmd("/agent", "Switch active agent", "bridge", true),
    cmd("/pipeline", "Set inter-agent pipeline", "bridge", true),
    cmd("/reload", "Reload config and agents", "bridge", false),

    // ── Context & memory ─────────────────────────────────────────────────
    cmd("/context", "Show session/data context", "local", false, () => {
      return (
        (localHandlers.getContext?.() as string | undefined) ||
        "No context available."
      );
    }),
    cmd("/compact", "Summarize older conversation history", "bridge", false),
    cmd("/fork", "Fork the conversation into a branch", "bridge", false),
    cmd(
      "/branch-summary",
      "Summarize the active branch back into the parent",
      "bridge",
      false,
    ),
    cmd("/discard-branch", "Discard the active branch", "bridge", false),
    cmd("/reset", "Reset runtime tool state", "bridge", false),
    cmd("/changes", "Show git status and diff preview", "bridge", false),

    // ── RAG & docs ───────────────────────────────────────────────────────
    cmd("/mount", "Mount codebase (context + RAG)", "bridge", true),
    cmd("/mount-code", "Alias for /mount", "bridge", true),
    cmd("/upload", "Ingest one document into RAG", "bridge", true),
    cmd("/upload-dir", "Bulk ingest docs into RAG", "bridge", true),
    cmd("/docs", "Fetch docs from Context7 library", "bridge", true),
    cmd("/rag", "Search RAG index", "bridge", true),

    // ── Skills ───────────────────────────────────────────────────────────
    cmd("/skills-health", "Show skill loader diagnostics", "bridge", false),
    cmd("/plugins", "Manage installed plugins", "local", true),
    cmd("/mcp", "Manage MCP servers", "local", true),

    // ── Reasoning ────────────────────────────────────────────────────────
    cmd(
      "/reasoner",
      "Select reasoning mode (none|ssr|tot|reflexion|...)",
      "local",
      true,
    ),

    // ── Display ──────────────────────────────────────────────────────────
    cmd(
      "/thinking",
      "Set thinking level (off|low|medium|high|xhigh)",
      "local",
      true,
      (args: string) => {
        const level = args.trim().toLowerCase();
        const valid = ["off", "low", "medium", "high", "xhigh"];
        if (valid.includes(level)) {
          localHandlers.setThinking?.(level);
          return `Thinking level: ${level}`;
        }
        return `Valid levels: ${valid.join(", ")}`;
      },
    ),
    cmd(
      "/mode",
      "Cycle thinking display mode (collapsed|summary|expanded)",
      "local",
      false,
      () => {
        localHandlers.cycleThinking?.();
        return "Thinking mode cycled.";
      },
    ),
    cmd(
      "/thinking-steps",
      "Set thinking display mode",
      "local",
      true,
      (args: string) => {
        const mode = args.trim().toLowerCase();
        if (["collapsed", "summary", "expanded"].includes(mode)) {
          localHandlers.setThinkingMode?.(mode);
          return `Thinking display: ${mode}`;
        }
        return "Valid modes: collapsed, summary, expanded";
      },
    ),
    cmd("/cache", "Toggle prompt caching", "local", true, (args: string) => {
      const enabled =
        args.trim().toLowerCase() !== "disable" &&
        args.trim().toLowerCase() !== "off" &&
        args.trim().toLowerCase() !== "0";
      localHandlers.setCache?.(enabled);
      return `Cache: ${enabled ? "enabled" : "disabled"}`;
    }),
    cmd("/trace", "Toggle trace messages", "local", true, (args: string) => {
      const state = args.trim().toLowerCase();
      if (state === "off" || state === "0") {
        localHandlers.setTrace?.(false);
        return "Trace: off";
      }
      localHandlers.setTrace?.(true);
      return "Trace: on";
    }),
    cmd("/clear", "Clear visible transcript only", "local", false, () => {
      localHandlers.clear?.();
      return "Transcript cleared.";
    }),

    // ── Shortcuts ────────────────────────────────────────────────────────
    cmd("/q", "Quick quit", "quit", false),
    cmd("/quit", "Exit TUI", "quit", false),
    cmd("/exit", "Alias for /quit", "quit", false),

    // ── Misc ─────────────────────────────────────────────────────────────
    cmd("/version", "Show TUI and bridge version", "local", false),
    cmd("/login", "Authenticate with provider", "bridge", true),
    cmd("/export", "Export transcript", "bridge", true),
  ];
}

// ── Fuzzy filter helper ───────────────────────────────────────────────────────

export function filterSlashCommands(
  commands: SlashCommandDef[],
  query: string,
  limit: number = 10,
): SlashCommandDef[] {
  if (!query || query.length <= 1) return commands.slice(0, limit);

  const lowerQuery = query.toLowerCase().trim();
  const scored = commands
    .map((cmd, idx) => {
      const cmdName = cmd.command.toLowerCase();
      const desc = cmd.description.toLowerCase();
      let score = -1;

      // Exact match on command name
      if (cmdName === lowerQuery) score = 3000 - idx;
      // Prefix match
      else if (cmdName.startsWith(lowerQuery))
        score = 2500 - (cmdName.length - lowerQuery.length) - idx;
      // Contains match
      else if (cmdName.includes(lowerQuery))
        score = 2000 - cmdName.indexOf(lowerQuery) * 8 - idx;
      // Subsequence match
      else if (subsequenceMatch(lowerQuery, cmdName)) score = 1500 - idx;
      // Description match
      else if (desc.includes(lowerQuery)) score = 800 - idx;
      // Word match in description
      else if (desc.split(/\s+/).some((w) => w.startsWith(lowerQuery)))
        score = 1000 - idx;

      return score >= 0 ? { cmd, score } : null;
    })
    .filter(Boolean) as { cmd: SlashCommandDef; score: number }[];

  scored.sort((a, b) => b.score - a.score);
  return scored.map((s) => s.cmd).slice(0, limit);
}

function subsequenceMatch(query: string, text: string): boolean {
  let qi = 0;
  for (let ti = 0; ti < text.length && qi < query.length; ti++) {
    if (text[ti] === query[qi]) qi++;
  }
  return qi === query.length;
}
