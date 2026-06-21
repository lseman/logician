// ── Extension context ────────────────────────────────────────────────────
// Shared mutable context passed to all extension event handlers within a
// single agent run. Extensions can use it to store state that persists
// across events (e.g., counters, feature flags, collected diagnostics).
//
// The context is isolated per harness instance — no cross-session leakage.

export interface ExtensionContextState {
	/** Extension-managed counters, tracked across events */
	counters: Record<string, number>;
	/** Feature flags set by extensions */
	features: Set<string>;
	/** Extension-set labels on session entries */
	labels: Record<string, string>;
	/** Arbitrary extension data store (keyed by extension name) */
	data: Record<string, unknown>;
	/** Turn-level diagnostics collected by extensions */
	diagnostics: Array<{ source: string; message: string; severity: "info" | "warning" | "error" }>;
}

export interface ExtensionContextActions {
	/** Increment an extension counter */
	incrementCounter: (name: string) => number;
	/** Set a feature flag */
	setFeature: (name: string, value: boolean) => void;
	/** Set or clear a label on the current session entry */
	setLabel: (label: string) => void;
	/** Store data keyed by extension name */
	storeData: (extension: string, key: string, value: unknown) => void;
	/** Retrieve data keyed by extension name */
	fetchData: (extension: string, key: string) => unknown;
	/** Record a diagnostic */
	addDiagnostic: (source: string, message: string, severity?: "info" | "warning" | "error") => void;
}

export interface ExtensionContext extends ExtensionContextState, ExtensionContextActions {
	/** Current turn index */
	turnIndex: number;
	/** Current agent iteration */
	iteration: number;
	/** Whether the agent is in a tool loop */
	inToolLoop: boolean;
	/** Whether the agent is in a thinking loop */
	inThinkingLoop: boolean;
}

export function createExtensionContext(): ExtensionContext {
	const state: ExtensionContextState = {
		counters: {},
		features: new Set(),
		labels: {},
		data: {},
		diagnostics: [],
	};

	const context: ExtensionContext = {
		...state,
		turnIndex: 0,
		iteration: 0,
		inToolLoop: false,
		inThinkingLoop: false,

		incrementCounter(name: string): number {
			state.counters[name] = (state.counters[name] ?? 0) + 1;
			return state.counters[name];
		},

		setFeature(name: string, value: boolean): void {
			if (value) state.features.add(name);
			else state.features.delete(name);
		},

		setLabel(label: string): void {
			state.labels[label] = label;
		},

		storeData(extension: string, key: string, value: unknown): void {
			if (!state.data[extension]) state.data[extension] = {};
			(state.data[extension] as Record<string, unknown>)[key] = value;
		},

		fetchData(extension: string, key: string): unknown {
			const extData = state.data[extension];
			if (!extData) return undefined;
			return (extData as Record<string, unknown>)[key];
		},

		addDiagnostic(source: string, message: string, severity = "info"): void {
			state.diagnostics.push({ source, message, severity });
		},
	};

	return context;
}
