// ── graphician worker ─────────────────────────────────────────────────────────
// Persistent JSONL client for the Graphician worker process
// (`graphician worker`). Queries run in-process on the Python side (no
// per-query interpreter/DB open) and index refreshes run as background
// child processes, so a query never blocks on a rebuild.
// Protocol: same JSONL contract as Legroom's sdk_worker (id + method in,
// id + ok out), consumed through JsonlWorker.

import { JsonlWorker } from "../capabilities/sdk/jsonl-worker.ts";

/** Wire state of the background index refresh (snake_case on the wire). */
interface WireBuildState {
	state: string;
	root?: string | null;
	last_exit?: number | null;
	last_output?: string | null;
}

/** State of the background index refresh. */
export interface GraphicianBuildState {
	state: "idle" | "running";
	root: string | null;
	lastExit: number | null;
	lastOutput: string | null;
}

export interface GraphicianWorkerOptions {
	/** Path to the graphician CLI executable (invoked with `worker`). */
	binary: string;
	/** Interpreter used to launch the worker (defaults to python3). */
	python?: string;
	/** Explicit launch arguments (defaults to [binary, "worker"]). */
	args?: string[];
	/** Per-request timeout in ms. */
	timeoutMs?: number;
}

const IDLE_BUILD: GraphicianBuildState = {
	state: "idle",
	root: null,
	lastExit: null,
	lastOutput: null,
};

function decodeBuild(raw: unknown): GraphicianBuildState {
	if (!raw || typeof raw !== "object") return IDLE_BUILD;
	const wire = raw as WireBuildState;
	return {
		state: wire.state === "running" ? "running" : "idle",
		root: typeof wire.root === "string" ? wire.root : null,
		lastExit: typeof wire.last_exit === "number" ? wire.last_exit : null,
		lastOutput: typeof wire.last_output === "string" ? wire.last_output : null,
	};
}

/** A lazy, persistent JSONL client for the Graphician worker process. */
export class GraphicianWorker {
	private readonly transport: JsonlWorker;
	constructor(options: GraphicianWorkerOptions) {
		this.transport = new JsonlWorker({
			name: "Graphician",
			python: options.python,
			args: options.args ?? [options.binary, "worker"],
			timeoutMs: options.timeoutMs ?? 35_000,
		});
	}

	/** Query the code graph in-process. Resolves to the response result and
	 *  the current background refresh state. */
	async query(
		db: string,
		operation: string,
		params: Record<string, unknown>,
	): Promise<{ result: Record<string, unknown>; build: GraphicianBuildState }> {
		const response = await this.transport.request({
			method: "query",
			db,
			operation,
			params,
		});
		const result = response.result;
		if (!result || typeof result !== "object" || Array.isArray(result))
			throw new Error("Graphician worker returned no result");
		return {
			result: result as Record<string, unknown>,
			build: decodeBuild(response.build),
		};
	}

	/** Start a background index refresh (smart build: full on an empty DB,
	 *  incremental otherwise). Returns immediately; at most one refresh
	 *  runs at a time. */
	async refresh(
		db: string,
		root: string,
	): Promise<{ started: boolean; build: GraphicianBuildState }> {
		const response = await this.transport.request({
			method: "refresh",
			db,
			root,
		});
		return {
			started: response.started === true,
			build: decodeBuild(response.build),
		};
	}

	/** Poll the state of the background refresh. */
	async buildStatus(): Promise<GraphicianBuildState> {
		const response = await this.transport.request({ method: "build_status" });
		return decodeBuild(response.build);
	}

	close(): void {
		this.transport.close();
	}
}
