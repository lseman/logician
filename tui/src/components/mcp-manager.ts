import { type Component, clampLineToWidth, visibleWidth } from "../tui-core.ts";

const RESET = "\x1b[0m";
const DIM = "\x1b[2m";
const BOLD = "\x1b[1m";
const HEADER = "\x1b[38;5;159m";
const SELECTED = "\x1b[38;5;111m";
const MUTED = "\x1b[38;5;245m";
const WARN = "\x1b[38;5;215m";

export interface McpServerItem {
  serverName: string;
  url: string;
  type: "stdio" | "http" | "streamable-http";
  command?: string;
  enabled: boolean;
  toolCount: number;
  configPath: string;
}

export type McpManagerAction =
  | { type: "toggle"; server: McpServerItem }
  | { type: "refresh" }
  | { type: "close" };

export class McpManagerOverlay implements Component {
  public visible = false;
  private servers: McpServerItem[] = [];
  private configPath = "";
  private selectedIndex = 0;
  private busyServerName: string | null = null;
  private message = "";
  private cachedLines: string[] | null = null;
  private cachedWidth = -1;

  setSnapshot(snapshot: {
    configPath?: string;
    servers: Array<Record<string, unknown>>;
    loadedServers?: Record<string, unknown>;
    errors?: string[];
  }): void {
    this.configPath = snapshot.configPath || "";
    this.servers = snapshot.servers.map((server) => {
      const serverName = String(server.server_name || server.name || "");
      const loadedServers = snapshot.loadedServers || {};
      const toolCount = Number(
        (loadedServers[serverName] as { toolCount?: number })?.toolCount || 0,
      );
      return {
        serverName,
        url: String(server.url || server.command || ""),
        type: (server.type || (server.url ? "http" : "stdio")) as
          | "stdio"
          | "http"
          | "streamable-http",
        command: String(server.command || ""),
        enabled: server.enabled !== false,
        toolCount,
        configPath: snapshot.configPath || "",
      };
    });
    if (this.selectedIndex >= this.servers.length) {
      this.selectedIndex = Math.max(0, this.servers.length - 1);
    }
    this.invalidate();
  }

  setBusy(serverName: string | null): void {
    this.busyServerName = serverName;
    this.invalidate();
  }

  setMessage(message: string): void {
    this.message = message;
    this.invalidate();
  }

  show(): void {
    this.visible = true;
    this.invalidate();
  }

  hide(): void {
    this.visible = false;
    this.busyServerName = null;
    this.invalidate();
  }

  isVisibleOverlay(): boolean {
    return this.visible;
  }

  handleInput(data: string): McpManagerAction | null {
    if (!this.visible) return null;

    if (data === "\x1b" || data === "\x03" || data.toLowerCase() === "q") {
      return { type: "close" };
    }
    if (data === "\r" || data === "\n") {
      return { type: "close" };
    }
    if (data === "r" || data === "R") {
      return { type: "refresh" };
    }
    if (data === "\x1b[A" || data === "\x1bOA" || data === "k") {
      this.moveSelection(-1);
      return null;
    }
    if (data === "\x1b[B" || data === "\x1bOB" || data === "j") {
      this.moveSelection(1);
      return null;
    }
    if (data === "\x1b[5~") {
      this.moveSelection(-8);
      return null;
    }
    if (data === "\x1b[6~") {
      this.moveSelection(8);
      return null;
    }
    if (data === " ") {
      const server = this.servers[this.selectedIndex];
      return server ? { type: "toggle", server } : null;
    }
    return null;
  }

  invalidate(): void {
    this.cachedLines = null;
  }

  render(width: number): string[] {
    if (width === this.cachedWidth && this.cachedLines !== null) {
      return this.cachedLines;
    }
    this.cachedWidth = width;

    if (!this.visible) return [];

    const overlayWidth = Math.max(48, Math.min(width, 110));
    const innerWidth = Math.max(1, overlayWidth - 4);
    const lines: string[] = [];

    lines.push(`${HEADER}┌${"─".repeat(overlayWidth - 2)}┐${RESET}`);
    lines.push(
      boxLine(
        `${BOLD}MCP Servers${RESET}${DIM} (${this.servers.length})${RESET}`,
        "space toggle · r refresh · enter/esc close",
        innerWidth,
      ),
    );
    if (this.configPath) {
      lines.push(
        boxLine(`${DIM}Config: ${this.configPath}${RESET}`, "", innerWidth),
      );
    }
    lines.push(`${HEADER}├${"─".repeat(overlayWidth - 2)}┤${RESET}`);

    if (!this.servers.length) {
      lines.push(
        boxLine(`${MUTED}No MCP servers configured.${RESET}`, "", innerWidth),
      );
    } else {
      const maxRows = 10;
      const start = Math.max(
        0,
        Math.min(
          this.selectedIndex - Math.floor(maxRows / 2),
          Math.max(0, this.servers.length - maxRows),
        ),
      );
      const end = Math.min(this.servers.length, start + maxRows);
      if (start > 0) {
        lines.push(boxLine(`${MUTED}↑ ${start} more${RESET}`, "", innerWidth));
      }
      for (let i = start; i < end; i++) {
        const server = this.servers[i];
        const selected = i === this.selectedIndex;
        const checkbox = server.enabled ? "[x]" : "[ ]";
        const cursor = selected ? "▸" : " ";
        const typeIcon =
          server.type === "http" || server.type === "streamable-http"
            ? "http"
            : "cmd";
        const typeStr = `${DIM}(${typeIcon})${RESET}`;
        const toolText =
          server.toolCount > 0
            ? `${DIM}${server.toolCount} tool(s)${RESET}`
            : `${DIM}0 tools${RESET}`;
        const urlText = server.url
          ? server.url.slice(0, 50)
          : server.command
            ? server.command.split(" ").slice(0, 3).join(" ") + "..."
            : "-";
        const busy =
          this.busyServerName === server.serverName
            ? ` ${DIM}updating...${RESET}`
            : "";
        const name = selected
          ? `${SELECTED}${BOLD}${server.serverName}${RESET}`
          : server.serverName;
        lines.push(
          boxLine(
            `${cursor} ${checkbox} ${name} ${typeStr}`,
            `${toolText} · ${urlText}${busy}`,
            innerWidth,
          ),
        );
      }
      if (end < this.servers.length) {
        lines.push(
          boxLine(
            `${MUTED}↓ ${this.servers.length - end} more${RESET}`,
            "",
            innerWidth,
          ),
        );
      }
    }

    lines.push(`${HEADER}├${"─".repeat(overlayWidth - 2)}┤${RESET}`);
    lines.push(
      boxLine(
        this.message
          ? `${DIM}${this.message}${RESET}`
          : `${MUTED}Toggle enables/disables MCP servers in config. Changes apply on next reconnect.${RESET}`,
        "",
        innerWidth,
      ),
    );
    lines.push(`${HEADER}└${"─".repeat(overlayWidth - 2)}┘${RESET}`);

    this.cachedLines = lines.map((line) => clampLineToWidth(line, width));
    return this.cachedLines;
  }

  private moveSelection(delta: number): void {
    const n = this.servers.length;
    if (!n) return;
    this.selectedIndex = (this.selectedIndex + delta + n) % n;
    this.invalidate();
  }
}

function boxLine(left: string, right: string, width: number): string {
  const leftWidth = visibleWidth(left);
  const rightWidth = visibleWidth(right);
  const gap = Math.max(1, width - leftWidth - rightWidth);
  const content = right ? `${left}${" ".repeat(gap)}${right}` : left;
  const pad = Math.max(0, width - visibleWidth(content));
  return `${HEADER}│${RESET} ${content}${" ".repeat(pad)} ${HEADER}│${RESET}`;
}
