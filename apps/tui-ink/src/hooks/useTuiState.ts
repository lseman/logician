// ── Ink TUI — React ↔ TuiState bridge ─────────────────────────────────────────
// TuiState is a plain EventEmitter that mutates in place. This hook subscribes
// the React tree to it via useSyncExternalStore so any state mutation triggers
// a re-render. The store returns the numeric `version` counter as its snapshot;
// components read fields straight off the (identity-stable) `state` object.

import { useSyncExternalStore } from "react";
import type { TuiState } from "../state.ts";

export function useTuiState(state: TuiState): TuiState {
	useSyncExternalStore(state.subscribe, state.getSnapshot, state.getSnapshot);
	return state;
}
