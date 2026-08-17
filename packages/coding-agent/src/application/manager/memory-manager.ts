// ── MemoryManager ─────────────────────────────────────────────────────────
// Owns the bridge's memory store, viewer dashboard, and semantic-extraction
// lifecycle. Extracted out of AgentCoreBridge — the store/hooks/viewer fields
// only ever interact with each other and the session id they're keyed to;
// the bridge's job is just to forward session-id transitions and call
// createHooks() once at construction.

import type { OpenAIBackend } from "@logician/agent-core/agent/core/backend.ts";
import type { AgentConfig } from "@logician/agent-core";
import {
	createMemoryHooks,
	createMemoryStore,
	getBoundViewerPort,
	LocalMemoryEmbedder,
	setSessionId,
	startViewerServer,
} from "@logician/memory";
import type { RuntimeEvent } from "../../runtime/events.ts";

export interface MemoryManagerOptions {
	memoryEnabled?: boolean;
	memoryDbPath?: string;
	memoryExtractorModel?: string;
	memoryExtractorBaseUrl?: string;
	memoryCaptureTools?: boolean;
	memoryInjectContext?: boolean;
	memoryContextBudget?: number;
	memoryViewerEnabled?: boolean;
	memoryViewerPort?: number;
	memoryEmbeddingsEnabled?: boolean;
	memoryEmbeddingModel?: string;
	model: string;
}

/** Bridge collaborators the memory hooks/extractor need at call time. */
export interface MemoryManagerRuntime {
	isRunning: () => boolean;
	getBackend: () => OpenAIBackend;
	emit: (event: RuntimeEvent) => void;
}

type Store = ReturnType<typeof createMemoryStore>;

export class MemoryManager {
	private store: Store | null = null;
	private captureTools: boolean;
	private injectContext: boolean;
	private contextBudget: number;
	private dbPath: string;
	private extractorModel: string;
	private extractorBaseUrl?: string;
	private backgroundTasks = new Set<Promise<void>>();
	private extractorRequests = new Set<AbortController>();
	private shutdownController = new AbortController();
	private viewerServer: ReturnType<typeof startViewerServer> | null = null;
	private viewerPort = 3200;
	private viewerPortConfig = 3200;
	private viewerEnabled = true;
	private embedder?: LocalMemoryEmbedder;
	private readonly cwd: string;

	constructor(cwd: string, sessionId: string, opts: MemoryManagerOptions) {
		this.cwd = cwd;
		this.dbPath = opts.memoryDbPath || `${cwd}/.logician/memory.db`;
		this.extractorModel = opts.memoryExtractorModel || opts.model;
		this.extractorBaseUrl = opts.memoryExtractorBaseUrl;

		if (opts.memoryEnabled === false) {
			this.captureTools = true;
			this.injectContext = true;
			this.contextBudget = 4000;
			return;
		}

		this.captureTools = opts.memoryCaptureTools ?? true;
		this.injectContext = opts.memoryInjectContext ?? true;
		this.contextBudget = opts.memoryContextBudget ?? 4000;
		if (opts.memoryEmbeddingsEnabled) {
			this.embedder = new LocalMemoryEmbedder(opts.memoryEmbeddingModel);
		}
		this.store = createMemoryStore(this.dbPath);
		const workspace = cwd || "";
		this.store.setCurrentWorkspace(workspace);
		setSessionId(this.store, sessionId);
		this.store.createSession(sessionId, { project: "", cwd, workspace });

		if (opts.memoryViewerEnabled !== false) {
			this.viewerEnabled = true;
			this.viewerPortConfig = opts.memoryViewerPort || 3200;
			this.viewerPort = this.viewerPortConfig;
			this.startViewer();
		}
	}

	private startViewer(): void {
		if (!this.store) return;
		try {
			this.viewerServer = startViewerServer({
				port: this.viewerPort,
				host: "0.0.0.0",
				store: this.store,
			});
			const bound = getBoundViewerPort();
			if (bound) this.viewerPort = bound;
		} catch {
			this.viewerServer = null;
		}
	}

	/** Lazy accessor for tool factories (createMemorySearchTool/createMemoryGetTool)
	 * that must resolve the store at call time, not at tool-registration time. */
	getStoreRef = (): Store | null => this.store;

	getStore(): Store | null {
		return this.store;
	}

	getViewerPort(): number | undefined {
		return this.viewerPort;
	}

	/**
	 * Merge memory hooks with existing hooks. Memory hooks capture observations
	 * and inject context. Returns the combined hooks object. Hooks close over
	 * the store instance live at construction time — toggling memory on/off
	 * later via setEnabled() does not rebuild this chain.
	 */
	createHooks(
		existingHooks: AgentConfig["hooks"],
		runtime: MemoryManagerRuntime,
	): AgentConfig["hooks"] {
		if (!this.store) return existingHooks;

		const memoryHooks = createMemoryHooks(this.store, {
			captureTools: this.captureTools,
			injectContext: this.injectContext,
			contextBudget: this.contextBudget,
			embedder: this.embedder,
			shutdownSignal: this.shutdownController.signal,
			semanticExtractor: async ({ systemPrompt, userPrompt }) => {
				// The stop hook fires before runMessage has transitioned to idle. Give
				// the UI a short quiet window, and never compete with an active turn.
				while (runtime.isRunning())
					await new Promise(resolve => setTimeout(resolve, 25));
				await new Promise(resolve => setTimeout(resolve, 500));
				if (runtime.isRunning())
					throw new DOMException(
						"Extractor deferred by active turn",
						"AbortError",
					);
				const controller = new AbortController();
				this.extractorRequests.add(controller);
				const backend = runtime.getBackend();
				const extractorModel = this.extractorModel || backend.model;
				const extractorBackend =
					this.extractorBaseUrl && backend.withEndpoint
						? backend.withEndpoint(extractorModel, this.extractorBaseUrl)
						: backend.withModel(extractorModel);
				try {
					const response = await extractorBackend.generate(
						[
							{ role: "system", content: systemPrompt },
							{ role: "user", content: userPrompt },
						],
						{
							temperature: 0.1,
							maxTokens: 1000,
							thinkingLevel: "off",
							timeoutMs: 30_000,
							maxRetries: 1,
							signal: controller.signal,
						},
					);
					return response.content || "";
				} finally {
					this.extractorRequests.delete(controller);
				}
			},
			onBackgroundTask: task => {
				this.backgroundTasks.add(task);
				void task.finally(() => this.backgroundTasks.delete(task));
			},
			onMemoriesSaved: memories => {
				const added = memories.filter(memory => memory.version === 1);
				const evolved = memories.filter(memory => memory.version > 1);
				if (added.length) {
					runtime.emit({
						type: "memory_update",
						kind: "reflections_added",
						count: added.length,
						items: added.map(memory => ({
							id: memory.id,
							content: memory.title,
						})),
					});
				}
				if (evolved.length) {
					runtime.emit({
						type: "memory_update",
						kind: "reflections_evolved",
						count: evolved.length,
						items: evolved.map(memory => ({
							id: memory.id,
							content: memory.title,
						})),
					});
				}
			},
		});

		// Merge hooks: existing hooks run first, then memory hooks
		const merged: Record<string, any> = {};

		for (const [key, value] of Object.entries(existingHooks || {})) {
			merged[key as keyof AgentConfig["hooks"]] = value;
		}

		for (const [key, value] of Object.entries(memoryHooks || {})) {
			const existing = merged[key as keyof AgentConfig["hooks"]];
			if (existing) {
				// Chain: existing hook runs first, then memory hook
				merged[key as keyof AgentConfig["hooks"]] = async (
					ctx: any,
					signal: any,
				) => {
					const existingResult = await existing(ctx, signal);
					const memoryCtx =
						key === "transformContext" && existingResult?.messages
							? { ...ctx, messages: existingResult.messages }
							: ctx;
					const memoryResult = await (value as Function)(memoryCtx, signal);
					// Return whichever has a non-undefined result
					if (memoryResult !== undefined) return memoryResult;
					return existingResult;
				};
			} else {
				merged[key as keyof AgentConfig["hooks"]] = value;
			}
		}

		return merged as AgentConfig["hooks"];
	}

	/** Enable/disable memory on the fly (the "memoryEnabled" runtime toggle). */
	setEnabled(enabled: boolean, sessionId: string): void {
		if (enabled && !this.store) {
			this.store = createMemoryStore(this.dbPath);
			setSessionId(this.store, sessionId);
			const workspace = this.cwd || "";
			this.store.setCurrentWorkspace(workspace);
			this.store.createSession(sessionId, {
				project: "",
				cwd: this.cwd,
				workspace,
			});
			if (this.viewerEnabled) {
				this.viewerPort = this.viewerPortConfig;
				this.startViewer();
			}
		} else if (!enabled) {
			if (this.store) this.store.close();
			this.store = null;
			if (this.viewerServer) {
				this.viewerServer.stop();
				this.viewerServer = null;
			}
		}
	}

	/** Sync the memory session to a new conversation session id, discarding an
	 * empty provisional session left behind by the previous id. */
	onSessionChanged(newSessionId: string, previousSessionId: string): void {
		if (!this.store) return;
		if (previousSessionId !== newSessionId) {
			this.store.discardEmptySession(previousSessionId);
		}
		setSessionId(this.store, newSessionId);
		this.store.createSession(newSessionId, {
			project: "",
			cwd: this.cwd,
			workspace: this.cwd || "",
		});
	}

	/** Re-point the memory session without discarding anything (used by reset()). */
	resetSession(sessionId: string): void {
		if (!this.store) return;
		setSessionId(this.store, sessionId);
		this.store.createSession(sessionId, {
			project: "",
			cwd: this.cwd,
			workspace: this.cwd || "",
		});
	}

	renameSession(sessionId: string, name: string): void {
		this.store?.updateSession(sessionId, { name: name.trim() });
	}

	getContextForInspection(
		messages: Array<{ role: string; content?: string }>,
	): string {
		if (!this.store || !this.injectContext) return "";
		const sessionId = this.store.getCurrentSessionId();
		if (!sessionId) return "";
		const latestPrompt =
			[...messages]
				.reverse()
				.find(message => message.role === "user" && message.content?.trim())
				?.content?.trim() || "";
		const retrieval = { objective: latestPrompt };
		return this.store.getContext(sessionId, this.contextBudget, retrieval);
	}

	getStats(sessionId: string): {
		memoryEnabled: boolean;
		memoryCount: number;
		sessionCount: number;
		observationCount: number;
		viewerPort?: number;
	} {
		if (!this.store) {
			return {
				memoryEnabled: false,
				memoryCount: 0,
				sessionCount: 0,
				observationCount: 0,
			};
		}
		const memories = this.store.list({ limit: 1000 });
		const sessions = this.store.listSessions();
		const observations = this.store.listObservations(sessionId, 1000);
		return {
			memoryEnabled: true,
			memoryCount: memories.length,
			sessionCount: sessions.length,
			observationCount: observations.length,
			viewerPort: this.viewerPort,
		};
	}

	/** Abort in-flight extraction requests. Called on stop(). */
	abortExtractors(): void {
		for (const controller of this.extractorRequests) controller.abort();
		this.shutdownController.abort();
	}

	waitForBackgroundTasks(): Promise<PromiseSettledResult<void>[]> {
		return Promise.allSettled([...this.backgroundTasks]);
	}
}
