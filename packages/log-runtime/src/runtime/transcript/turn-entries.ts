// ── Turn ↔ session entry projection ──────────────────────────────────────
// The TUI's live/rendered unit is a Turn (interleaved AssistantChunk[] with
// full tool/thinking detail — see transcript.ts). The persisted unit is a
// CustomSessionEntry<Turn> in agent-core's JSONL tree: the whole Turn is
// stored verbatim as opaque app data, so nothing about it (thinking chunks,
// tool args/results/durations, subagent chunks) is lost on save or resume —
// unlike the old SQLite path, which discarded thinking/tool chunks entirely.

import type {
	CustomSessionEntry,
	SessionEntry,
	SessionStore,
} from "@logician/log-core/runtime";
import type { Turn } from "./transcript.ts";

const TURN_CUSTOM_TYPE = "tui_turn";

type TurnSessionEntry = CustomSessionEntry<Turn> & { customType: "tui_turn" };

function isTurnEntry(entry: SessionEntry): entry is TurnSessionEntry {
	return entry.type === "custom" && entry.customType === TURN_CUSTOM_TYPE;
}

/** Persist a completed turn as one entry. Only call once `turn.isComplete`. */
export function saveTurn(session: SessionStore, turn: Turn): void {
	session.appendCustom(TURN_CUSTOM_TYPE, turn);
}

/** Reconstruct the turn list for a session from its entry tree (root→leaf path). */
export function loadTurns(session: SessionStore): Turn[] {
	return session
		.getPathToRootEntries()
		.filter(isTurnEntry)
		.map(entry => entry.data);
}
