import type { SessionEntry } from "@logician/log-core/runtime";

export interface GroveSession {
	readonly id: string;
	readonly name: string;
	readonly preview: string;
	readonly cwd: string;
	readonly lastActivity: number;
	readonly messageCount: number;
	readonly branchCount: number;
	readonly entries: readonly SessionEntry[];
}

export interface GroveRepository {
	list(cwd: string): readonly GroveSession[];
}

type GroveScreen =
	| { readonly kind: "forest" }
	| { readonly kind: "tree"; readonly sessionId: string };

export interface GroveState {
	readonly screen: GroveScreen;
	readonly selection: number;
	readonly scroll: number;
	readonly query: string;
}
