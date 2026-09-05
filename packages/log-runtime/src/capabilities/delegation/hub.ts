// ── Hub Message Bus ──────────────────────────────────────────────────────────
// Shared in-memory coordination layer for spawned subagents. Each spawned agent
// registers with the bus, can send messages to peers, wait for responses, and
// observe the status of its siblings.

export interface HubMessage {
  /** Unique message identifier. */
  id: string;
  /** Unix timestamp (ms). */
  timestamp: number;
  /** Sender agent ID. */
  from: string;
  /** Recipient agent ID or "*" for broadcast. */
  to: string;
  /** Message body. */
  body: string;
  /** Optional metadata. */
  meta?: Record<string, unknown>;
}

export interface HubAgentInfo {
  /** Unique agent ID (e.g. "agent_abc123"). */
  id: string;
  /** Agent definition name (e.g. "general", "explorer"). */
  agent: string;
  /** Task description. */
  task: string;
  /** Current status. */
  status: "running" | "completed" | "failed" | "cancelled";
  /** Position within a spawn_agents batch, if applicable. */
  taskIndex?: number;
  /** Arbitrary result/error string after completion. */
  result?: string;
}

export interface HubMessageBus {
  /** Register an agent with the bus. */
  register(id: string, info: HubAgentInfo): void;
  /** Remove an agent from the bus. */
  unregister(id: string): void;
  /**
   * Send a message from `from` to `to` (an agent ID or "*" for broadcast).
   * Synchronous — the message is immediately available in the recipient's inbox.
   */
  send(from: string, to: string, body: string, meta?: Record<string, unknown>): HubMessage;
  /**
   * Wait for messages from specific agent IDs. Resolves when matching messages
   * arrive or the timeout elapses. Returns all collected messages sorted by timestamp.
   */
  wait(handles: string[], timeoutMs?: number): Promise<HubMessage[]>;
  /** List all registered agents and their statuses. */
  jobs(): HubAgentInfo[];
  /**
   * Drain and return all messages addressed to `agentId` (including broadcasts).
   * Messages are removed from the inbox so subsequent calls return only newer ones.
   */
  inbox(agentId: string): HubMessage[];
  /** Signal that an agent has completed (success or failure). */
  complete(id: string, status: "completed" | "failed" | "cancelled", result?: string): void;
  /** Get all messages ever sent through this bus. */
  allMessages(): HubMessage[];
  /** Reset all state (for re-use). */
  reset(): void;
}

interface _AgentEntry {
  id: string;
  info: HubAgentInfo;
  /** Incoming messages not yet drained by inbox(). */
  inbox: HubMessage[];
  /** Resolvers waiting via hub.wait() for this agent. */
  waitResolvers: Array<{
    resolve: (messages: HubMessage[]) => void;
    timeoutId?: ReturnType<typeof setTimeout>;
  }>;
}

export function createHubMessageBus(): HubMessageBus {
  const agents = new Map<string, _AgentEntry>();
  const allMessages: HubMessage[] = [];
  let nextId = 0;

  function genId(): string {
    return String(nextId++);
  }

  return {
    register(id, info) {
      if (agents.has(id)) {
        return; // already registered
      }
      agents.set(id, { id, info, inbox: [], waitResolvers: [] });
    },

    unregister(id) {
      agents.delete(id);
    },

    send(from, to, body, meta) {
      const msg: HubMessage = {
        id: genId(),
        timestamp: Date.now(),
        from,
        to,
        body,
        meta,
      };
      allMessages.push(msg);

      // Deliver to target agent(s)
      const targets = to === "*" ? Array.from(agents.keys()) : [to];
      for (const targetId of targets) {
        const entry = agents.get(targetId);
        if (entry) {
          entry.inbox.push(msg);
          // Resolve all active waiters watching this agent
          for (const resolver of entry.waitResolvers) {
            clearTimeout(resolver.timeoutId);
            resolver.resolve([...entry.inbox]);
          }
          entry.waitResolvers = [];
        }
      }

      return msg;
    },

    wait(handles, timeoutMs) {
      return new Promise<HubMessage[]>((resolve) => {
        // Collect messages from target agents right now
        const collected: HubMessage[] = [];
        for (const handle of handles) {
          const entry = agents.get(handle);
          if (entry) {
            collected.push(...entry.inbox);
          }
        }
        if (collected.length > 0) {
          collected.sort((a, b) => a.timestamp - b.timestamp);
          resolve(collected);
          return;
        }

        // Register resolver — triggered when new messages arrive for any target
        const timeoutId = timeoutMs ? setTimeout(() => resolve([]), timeoutMs) : undefined;
        const resolver = { resolve: (_messages: HubMessage[]) => {}, timeoutId };
        const realResolve = (messages: HubMessage[]) => {
          clearTimeout(timeoutId);
          resolve([...messages].sort((a, b) => a.timestamp - b.timestamp));
        };
        resolver.resolve = realResolve;

        for (const handle of handles) {
          const entry = agents.get(handle);
          if (entry) {
            entry.waitResolvers.push(resolver as unknown as NonNullable<typeof entry.waitResolvers>[number]);
          }
        }
      });
    },

    jobs() {
      return Array.from(agents.values()).map(e => ({ ...e.info }));
    },

    inbox(agentId) {
      const entry = agents.get(agentId);
      if (!entry) return [];
      const msgs = [...entry.inbox];
      entry.inbox = [];
      return msgs;
    },

    complete(id, status, result) {
      const entry = agents.get(id);
      if (entry) {
        entry.info.status = status;
        entry.info.result = result;
        // Resolve all waiters watching this agent
        for (const resolver of entry.waitResolvers) {
          clearTimeout(resolver.timeoutId);
          resolver.resolve([...entry.inbox]);
        }
        entry.waitResolvers = [];
      }
    },

    allMessages() {
      return [...allMessages];
    },

    reset() {
      agents.clear();
      allMessages.length = 0;
      nextId = 0;
    },
  };
}
