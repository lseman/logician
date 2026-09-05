// ── Hub Tools ────────────────────────────────────────────────────────────────
// Tools that spawned subagents can use to coordinate with each other via the
// shared HubMessageBus. These tools are added to the child's tool set when
// a hub is available.

import type { Tool } from "@logician/log-core";
import type { HubMessageBus } from "./hub.ts";

// ── Hub tool dependencies ────────────────────────────────────────────────────

export interface HubToolDeps {
  /** The shared message bus instance. */
  hub: HubMessageBus;
  /** This agent's unique ID for the hub. */
  agentId: string;
}

// ── hub.send tool ────────────────────────────────────────────────────────────

export function hubSendTool(deps: HubToolDeps): Tool {
  return {
    name: "hub_send",
    description:
      "Send a message to a peer subagent or broadcast to all peers. " +
      "The message is delivered immediately and can be read by the recipient " +
      "via hub_inbox or hub_wait. Use \"*\" as the recipient to broadcast.",
    parameters: {
      type: "object",
      properties: {
        to: {
          type: "string",
          description: "Recipient agent ID, or \"*\" to broadcast to all peers.",
        },
        body: {
          type: "string",
          description: "Message content.",
        },
        meta: {
          type: "object",
          description: "Optional metadata (key-value pairs).",
        },
      },
      required: ["to", "body"],
    },
    execute: async (args) => {
      const to = typeof args.to === "string" ? args.to : "";
      const body = typeof args.body === "string" ? args.body : "";
      if (!to || !body) {
        return {
          content: "Error: hub_send requires 'to' (agent ID or \"*\") and 'body' (message text).",
          isError: true,
        };
      }
      deps.hub.send(deps.agentId, to, body, typeof args.meta === "object" && args.meta !== null && !Array.isArray(args.meta) ? args.meta as Record<string, unknown> : undefined);
      return {
        content: `Message sent to ${to}.`,
      };
    },
  };
}

// ── hub.wait tool ────────────────────────────────────────────────────────────

export function hubWaitTool(deps: HubToolDeps): Tool {
  return {
    name: "hub_wait",
    description:
      "Wait for messages from specific peer agents. Resolves immediately if " +
      "messages are already in their inboxes, otherwise waits up to the timeout. " +
      "Returns all collected messages sorted by timestamp.",
    parameters: {
      type: "object",
      properties: {
        handles: {
          type: "array",
          items: { type: "string" },
          description: "Agent IDs to wait for messages from.",
        },
        timeout_ms: {
          type: "integer",
          minimum: 100,
          description: "Maximum wait time in milliseconds (default: 30000).",
        },
      },
      required: ["handles"],
    },
    execute: async (args) => {
      const handles = args.handles;
      if (!Array.isArray(handles) || handles.length === 0) {
        return {
          content: "Error: hub_wait requires a non-empty 'handles' array of agent IDs.",
          isError: true,
        };
      }
      const timeoutMs = typeof args.timeout_ms === "number" && args.timeout_ms > 0
        ? args.timeout_ms
        : 30_000;
      const messages = await deps.hub.wait(handles, timeoutMs);
      if (messages.length === 0) {
        return {
          content: `No messages received from ${handles.join(", ")} within ${timeoutMs}ms.`,
        };
      }
      const lines = messages.map(
        m => `[${m.from}] ${m.body}${m.meta ? ` (meta: ${JSON.stringify(m.meta)})` : ""}`,
      );
      return {
        content: `Received ${messages.length} message(s):\n` + lines.join("\n"),
      };
    },
  };
}

// ── hub.jobs tool ────────────────────────────────────────────────────────────

export function hubJobsTool(deps: HubToolDeps): Tool {
  return {
    name: "hub_jobs",
    description:
      "List all registered subagents and their current status. " +
      "Shows agent IDs, names, tasks, and completion status.",
    parameters: {
      type: "object",
      properties: {},
    },
    execute: async () => {
      const jobs = deps.hub.jobs();
      if (jobs.length === 0) {
        return { content: "No active subagents." };
      }
      const lines = jobs.map(
        j => `  ${j.id} (${j.agent}) [${j.status}] task=${j.task}${j.taskIndex !== undefined ? ` index=${j.taskIndex}` : ""}`,
      );
      return {
        content: `Active subagents (${jobs.length}):\n` + lines.join("\n"),
      };
    },
  };
}

// ── hub_inbox tool ───────────────────────────────────────────────────────────

export function hubInboxTool(deps: HubToolDeps): Tool {
  return {
    name: "hub_inbox",
    description:
      "Drain and return all messages addressed to this agent (including broadcasts). " +
      "Messages are removed from the inbox so subsequent calls return only newer ones.",
    parameters: {
      type: "object",
      properties: {},
    },
    execute: async () => {
      const messages = deps.hub.inbox(deps.agentId);
      if (messages.length === 0) {
        return { content: "No messages in inbox." };
      }
      const lines = messages.map(
        m => `[${m.from}] ${m.body}${m.meta ? ` (meta: ${JSON.stringify(m.meta)})` : ""}`,
      );
      return {
        content: `Inbox (${messages.length} message(s)):\n` + lines.join("\n"),
      };
    },
  };
}
