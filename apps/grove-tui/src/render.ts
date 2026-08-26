import type { SessionEntry } from "@logician/log-core/runtime";
import stringWidth from "string-width";
import type { GroveSession, GroveState } from "./model.ts";

const RESET = "\u001b[0m";
const DIM = "\u001b[2m";
const BOLD = "\u001b[1m";
const GREEN = "\u001b[38;5;114m";
const TEAL = "\u001b[38;5;80m";
const YELLOW = "\u001b[38;5;221m";

function clip(value: string, width: number): string {
	if (width <= 0) return "";
	if (stringWidth(value) <= width) return value;
	let result = "";
	for (const character of value) {
		if (stringWidth(`${result}${character}…`) > width) break;
		result += character;
	}
	return `${result}…`;
}

function age(timestamp: number, now = Date.now()): string {
	const minutes = Math.max(0, Math.floor((now - timestamp) / 60_000));
	if (minutes < 1) return "now";
	if (minutes < 60) return `${minutes}m`;
	const hours = Math.floor(minutes / 60);
	if (hours < 24) return `${hours}h`;
	return `${Math.floor(hours / 24)}d`;
}

function textOf(entry: SessionEntry): string {
	if (entry.type === "message") {
		const content = entry.message.content;
		return typeof content === "string"
			? content.replace(/\s+/g, " ").trim()
			: "";
	}
	if (entry.type === "compaction") return `summary: ${entry.summary}`;
	if (entry.type === "branch_summary") return `branch: ${entry.summary}`;
	if (entry.type === "label")
		return entry.label ? `label: ${entry.label}` : "label removed";
	return entry.type.replaceAll("_", " ");
}

function roleOf(entry: SessionEntry): string {
	if (entry.type !== "message") return "◇";
	if (entry.message.role === "user") return `${TEAL}◆${RESET}`;
	if (entry.message.role === "assistant") return `${GREEN}●${RESET}`;
	return "⚙";
}

function isConversationEntry(entry: SessionEntry): boolean {
	return (
		entry.type === "message" ||
		entry.type === "compaction" ||
		entry.type === "branch_summary" ||
		entry.type === "label"
	);
}

/** Hide configuration events and reconnect their children to the visible tree. */
export function conversationParents(
	entries: readonly SessionEntry[],
): ReadonlyMap<string | undefined, readonly SessionEntry[]> {
	const byId = new Map(entries.map(entry => [entry.id, entry]));
	const visibleIds = new Set(
		entries.filter(isConversationEntry).map(entry => entry.id),
	);
	const children = new Map<string | undefined, SessionEntry[]>();
	for (const entry of entries) {
		if (!visibleIds.has(entry.id)) continue;
		let parentId = entry.parentId;
		const visited = new Set<string>();
		while (parentId && !visibleIds.has(parentId) && !visited.has(parentId)) {
			visited.add(parentId);
			parentId = byId.get(parentId)?.parentId;
		}
		const siblings = children.get(parentId) ?? [];
		siblings.push(entry);
		children.set(parentId, siblings);
	}
	return children;
}

export function filterSessions(
	sessions: readonly GroveSession[],
	query: string,
): readonly GroveSession[] {
	const needle = query.trim().toLowerCase();
	if (!needle) return sessions;
	return sessions.filter(session =>
		`${session.name}\n${session.preview}`.toLowerCase().includes(needle),
	);
}

export function renderForest(
	sessions: readonly GroveSession[],
	state: GroveState,
	width: number,
	height: number,
): string[] {
	const filtered = filterSessions(sessions, state.query);
	const lines = [
		`${GREEN}${BOLD}  LOGICIAN GROVE${RESET}  ${DIM}${filtered.length} conversation${filtered.length === 1 ? "" : "s"}${RESET}`,
		`${DIM}  tree-first session navigator · ${clip(process.cwd(), Math.max(10, width - 40))}${RESET}`,
		"",
	];
	const available = Math.max(1, height - 7);
	const start = Math.min(
		state.scroll,
		Math.max(0, filtered.length - available),
	);
	for (
		let index = start;
		index < Math.min(filtered.length, start + available);
		index++
	) {
		const session = filtered[index];
		if (!session) continue;
		const selected = index === state.selection;
		const marker = selected ? `${YELLOW}▶${RESET}` : " ";
		const tree = session.branchCount > 0 ? "♜" : "♟";
		const stats = `${session.messageCount} nodes${session.branchCount ? ` · ${session.branchCount} forks` : ""} · ${age(session.lastActivity)}`;
		const titleWidth = Math.max(8, width - stringWidth(stats) - 10);
		lines.push(
			`${marker} ${GREEN}${tree}${RESET} ${selected ? BOLD : ""}${clip(session.name, titleWidth)}${RESET} ${DIM}${stats}${RESET}`,
		);
		lines.push(
			`    ${DIM}${clip(session.preview || "(empty conversation)", Math.max(10, width - 6))}${RESET}`,
		);
	}
	if (filtered.length === 0)
		lines.push(`  ${DIM}No conversations match this grove.${RESET}`);
	while (lines.length < height - 2) lines.push("");
	if (state.query) lines.push(`${TEAL}  / ${state.query}${RESET}`);
	else lines.push("");
	lines.push(
		`${DIM}  ↑↓ select   →/t tree   Enter/a open Logician   / search   r refresh   q quit${RESET}`,
	);
	return lines.slice(0, height);
}

export function renderTree(
	session: GroveSession,
	width: number,
	height: number,
): string[] {
	const entries = session.entries;
	const children = conversationParents(entries);
	const visibleEntryCount = [...children.values()].reduce(
		(count, nodes) => count + nodes.length,
		0,
	);
	const lines = [
		`${GREEN}${BOLD}  ${clip(session.name, Math.max(10, width - 4))}${RESET}`,
		`${DIM}  ${visibleEntryCount} conversation entries · ${session.branchCount} forks · ${session.id}${RESET}`,
		"",
	];
	const walk = (parentId: string | undefined, prefix: string): void => {
		const nodes = children.get(parentId) ?? [];
		for (let index = 0; index < nodes.length; index++) {
			const node = nodes[index];
			if (!node || lines.length >= height - 2) return;
			const last = index === nodes.length - 1;
			const connector = last ? "└─" : "├─";
			lines.push(
				`  ${DIM}${prefix}${connector}${RESET} ${roleOf(node)} ${clip(textOf(node) || "(empty)", Math.max(8, width - stringWidth(prefix) - 9))}`,
			);
			walk(node.id, `${prefix}${last ? "  " : "│ "}`);
		}
	};
	walk(undefined, "");
	if (visibleEntryCount === 0)
		lines.push(`  ${DIM}(empty conversation)${RESET}`);
	if (lines.length >= height - 2)
		lines[height - 3] = `  ${DIM}… tree clipped to terminal height${RESET}`;
	while (lines.length < height - 1) lines.push("");
	lines.push(
		`${DIM}  ←/Esc forest   Enter/a open Logician   r refresh   q quit${RESET}`,
	);
	return lines.slice(0, height);
}

export function render(
	sessions: readonly GroveSession[],
	state: GroveState,
	width: number,
	height: number,
): string {
	const safeWidth = Math.max(40, width);
	const safeHeight = Math.max(10, height);
	const screen = state.screen;
	const session =
		screen.kind === "tree"
			? sessions.find(item => item.id === screen.sessionId)
			: undefined;
	const lines = session
		? renderTree(session, safeWidth, safeHeight)
		: renderForest(sessions, state, safeWidth, safeHeight);
	return lines.map(line => clip(line, safeWidth + 64)).join("\n");
}
