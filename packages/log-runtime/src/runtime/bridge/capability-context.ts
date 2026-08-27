/**
 * Shared registration point for AgentRuntime's independent capability
 * managers (RepositoryMap, LspClientPool, ExtensionRegistry, InteractionGateway,
 * MemoryHost). Each is self-contained at construction time — its
 * constructor takes only cwd/sessionId/options/emit, never another manager
 * — so they mount into typed slots on the exported RuntimeContext in any
 * order, instead of being `new`'d as inline constructor statements with no
 * shared registration point. AgentRuntime keeps a private field per slot
 * delegating to this context, so existing call sites (`this.lspManager.foo()`)
 * are unaffected; only construction changes shape. File name is
 * capability-context.ts (not runtime-context.ts) to stay distinct from
 * ../runtime-context.ts, the unrelated ToolRegistry-composition helper.
 *
 * What deliberately stays OUT of this context: ToolRouter (its `extraTools`
 * closure reads memory lazily, but its own construction has no cross-manager
 * input worth registering for), the AgentConfig object, and
 * ModelSelector/SettingsGateway/AgentCoordinator. Those are
 * consumers that aggregate context slots' output (hooks from lsp+memory,
 * permissions from interactions, tools from toolRouter) — they are built
 * from the context, not entries in it, and forcing them into slots would
 * hide real build-order dependencies behind fake independence.
 */

import type { RuntimeEvent } from "@logician/log-core/events";
import { ExtensionRegistry } from "../../capabilities/extensions/extensions.ts";
import { InteractionGateway } from "../../capabilities/interactions/interaction-gateway.ts";
import { LspClientPool } from "../../capabilities/lsp/lsp-client-pool.ts";
import { MemoryHost } from "../../capabilities/memory/memory.ts";
import { RepositoryMap } from "../../capabilities/repository-map/repository-map.ts";
import type { AgentBridgeOptions } from "./types.ts";

interface RuntimeContextSlots {
	repositoryMap: RepositoryMap | undefined;
	lsp: LspClientPool;
	extensions: ExtensionRegistry;
	interactions: InteractionGateway;
	memory: MemoryHost;
}

type SlotKey = keyof RuntimeContextSlots;

/** A capability entry mounted into RuntimeContext during construction. */
export interface CapabilityEntry<K extends SlotKey = SlotKey> {
	id: K;
	register: (ctx: RuntimeContext) => RuntimeContextSlots[K];
}

/**
 * Typed registration slots for AgentRuntime's independent managers. Each
 * slot is set exactly once (register throws on a second call for the same
 * id — a repeat registration is a construction bug, not a valid override)
 * and read through a getter that throws if read before registration, so an
 * ordering mistake fails loudly at the read site instead of surfacing as a
 * silent `undefined`.
 */
export class RuntimeContext {
	private readonly values = new Map<SlotKey, unknown>();

	register<K extends SlotKey>(id: K, value: RuntimeContextSlots[K]): void {
		if (this.values.has(id)) {
			throw new Error(`RuntimeContext: "${id}" is already registered`);
		}
		this.values.set(id, value);
	}

	private read<K extends SlotKey>(id: K): RuntimeContextSlots[K] {
		if (!this.values.has(id)) {
			throw new Error(
				`RuntimeContext: "${id}" was read before it was registered`,
			);
		}
		return this.values.get(id) as RuntimeContextSlots[K];
	}

	get repositoryMap(): RepositoryMap | undefined {
		return this.read("repositoryMap");
	}

	get lsp(): LspClientPool {
		return this.read("lsp");
	}

	get extensions(): ExtensionRegistry {
		return this.read("extensions");
	}

	get interactions(): InteractionGateway {
		return this.read("interactions");
	}

	get memory(): MemoryHost {
		return this.read("memory");
	}

	/** Mount every entry in order. Order is irrelevant today (no entry's
	 * register() reads another slot) but is preserved as the extension
	 * point: a future capability that needs an earlier one just reads
	 * `ctx.<slot>` inside its own register(). */
	static mount(entries: CapabilityEntry[]): RuntimeContext {
		const ctx = new RuntimeContext();
		for (const entry of entries) {
			ctx.register(entry.id, entry.register(ctx));
		}
		return ctx;
	}
}

interface RuntimeContextOptions {
	opts: AgentBridgeOptions;
	cwd: string;
	sessionId: string;
	projectTrusted: boolean;
	emit: (event: RuntimeEvent) => void;
}

/** Build the independent capability layer consumed by AgentRuntime. */
export function createRuntimeContext({
	opts,
	cwd,
	sessionId,
	projectTrusted,
	emit,
}: RuntimeContextOptions): RuntimeContext {
	return RuntimeContext.mount([
		{
			id: "repositoryMap",
			register: () =>
				opts.repositoryMap?.enabled !== false
					? new RepositoryMap(cwd, {
							maxTokens: opts.repositoryMap?.maxTokens,
						})
					: undefined,
		},
		{
			id: "lsp",
			register: () => {
				const serverOverrides = opts.lsp?.serverOverrides;
				return new LspClientPool(cwd, {
					timeoutMs: opts.lsp?.timeoutMs ?? 2_000,
					servers:
						serverOverrides && Object.keys(serverOverrides).length > 0
							? serverOverrides
							: undefined,
				});
			},
		},
		{
			id: "extensions",
			register: () =>
				new ExtensionRegistry({
					sessionId,
					cwd,
					extensionDirs: opts.extensions?.dirs,
					projectTrusted,
				}),
		},
		{
			id: "interactions",
			register: () =>
				new InteractionGateway({
					mode: opts.permissions?.mode ?? "acceptEdits",
					rules: opts.permissions?.rules,
					emit,
				}),
		},
		{
			id: "memory",
			register: () =>
				new MemoryHost(cwd, sessionId, {
					memoryEnabled: opts.memory?.enabled,
					memoryDbPath: opts.memory?.dbPath,
					memoryExtractorModel: opts.memory?.extractorModel,
					memoryExtractorBaseUrl: opts.memory?.extractorBaseUrl,
					memoryCaptureTools: opts.memory?.captureTools,
					memoryInjectContext: opts.memory?.injectContext,
					memoryContextBudget: opts.memory?.contextBudget,
					memoryViewerEnabled: opts.memory?.viewerEnabled,
					memoryViewerPort: opts.memory?.viewerPort,
					memoryEmbeddingsEnabled: opts.memory?.embeddingsEnabled,
					memoryEmbeddingModel: opts.memory?.embeddingModel,
					model: opts.model,
				}),
		},
	]);
}
