import type {
	CompactionMode,
	CompactionSettings,
} from "../../compaction/engine.ts";
import { toMessages } from "../../compaction/engine.ts";
import {
	runCompaction,
	shouldAutoCompact,
} from "../../compaction/orchestration.ts";
import type { ExtensionRunner } from "../../extensions/runner.ts";
import type { LLMBackend } from "../../provider/backend.ts";
import { resolveTokenEncoding } from "../../provider/messages.ts";
import type { AgentConfig } from "../../types/config.ts";
import type {
	BeforeCompactContext,
	BeforeCompactResult,
	Message,
} from "../../types/messages.ts";
export type CompactionReason = "auto" | "manual";

export interface SessionCompactorDependencies {
	backend: () => LLMBackend;
	history: () => Message[];
	commitHistory: (expected: Message[], replacement: Message[]) => boolean;
	config: () => Readonly<AgentConfig>;
	identity: () => { sessionId: string; cwd: string };
	extensionRunner: () => ExtensionRunner | undefined;
	beforeCompact: (
		context: BeforeCompactContext,
	) => Promise<BeforeCompactResult | undefined>;
	afterCompact: () => Promise<void>;
	persistCompaction: (
		summary: string,
		tokensBefore: number,
		firstKeptEntryId?: string,
	) => void;
	estimateTokens: () => Promise<number>;
	emit: (event: {
		type: "compaction";
		reason: CompactionReason;
		tokensBefore?: number | undefined;
		tokensAfter?: number | undefined;
	}) => void;
}

const DEFAULT_SETTINGS: CompactionSettings = {
	enabled: false,
	reserveTokens: 16_384,
	keepRecentTokens: 20_000,
	contextWindow: 128_000,
};

/**
 * Owns the complete session-level compaction transaction: policy, hooks,
 * extensions, optimistic history commit, persistence, and outcome events.
 */
export class SessionCompactor {
	private settings: CompactionSettings = { ...DEFAULT_SETTINGS };

	constructor(private readonly dependencies: SessionCompactorDependencies) {}

	get enabled(): boolean {
		return this.settings.enabled;
	}

	get contextWindow(): number {
		return this.settings.contextWindow ?? 128_000;
	}

	configure(settings: Partial<CompactionSettings>): void {
		this.settings = { ...this.settings, ...settings };
	}

	enable(enabled: boolean): void {
		this.settings = { ...this.settings, enabled };
	}

	shouldCompact(
		messages: Message[] = this.dependencies.history(),
	): Promise<boolean> {
		return shouldAutoCompact(
			this.settings,
			messages,
			resolveTokenEncoding(this.dependencies.backend().model),
		);
	}

	async recordCompaction(
		messages: Message[],
		tokensBefore: Promise<number>,
	): Promise<void> {
		const summary = messages.find(
			message => String(message.role) === "compactionSummary",
		)?.content;
		if (typeof summary !== "string" || !summary.trim()) return;
		const firstKeptEntryId = messages
			.map(message => message as Message & { entryId?: string })
			.find(
				message =>
					String(message.role) !== "compactionSummary" && message.entryId,
			)?.entryId;
		this.dependencies.persistCompaction(
			summary,
			await tokensBefore,
			firstKeptEntryId,
		);
	}

	async compact(
		reason: CompactionReason,
		force: boolean,
		mode?: CompactionMode,
	): Promise<number> {
		const messages = this.dependencies.history();
		this.dependencies.emit({ type: "compaction", reason });
		let postCompactEmitted = false;
		const emitPostCompact = async (): Promise<void> => {
			if (postCompactEmitted) return;
			postCompactEmitted = true;
			await this.dependencies.afterCompact();
		};
		const finishUnchanged = async (tokensBefore: number): Promise<number> => {
			await emitPostCompact();
			this.dependencies.emit({
				type: "compaction",
				reason,
				tokensBefore,
				tokensAfter: tokensBefore,
			});
			return 0;
		};

		try {
			const before = await this.dependencies.estimateTokens();
			if (!force && !(await this.shouldCompact(messages))) {
				return await finishUnchanged(before);
			}

			const preResult = await this.dependencies.beforeCompact({
				messages,
				tokensBefore: before,
				reason,
			});
			if (preResult?.cancel) return await finishUnchanged(before);

			const identity = this.dependencies.identity();
			await this.dependencies.extensionRunner()?.emit({
				type: "session_before_compact",
				context: {
					...identity,
					reason,
					tokensBefore: before,
					messages: [...messages],
				},
			});

			const config = this.dependencies.config();
			// A caller-supplied mode wins; otherwise fall back to the configured
			// default mode — only the "snapcompact" case (CompactionSettings.mode
			// also allows "remote"/"auto"/"shake", a pre-existing, differently
			// shaped union not unified here).
			const effectiveMode: CompactionMode | undefined =
				mode ??
				(this.settings.mode === "snapcompact" ? "snapcompact" : undefined);
			// Auto-derive the vision-model provider hint from the live model
			// unless explicitly configured — otherwise PROVIDER_COLS tuning is
			// unreachable.
			const frameOptions =
				effectiveMode === "snapcompact"
					? {
							...this.settings.frameOptions,
							provider: this.settings.frameOptions?.provider ?? config.model,
						}
					: undefined;
			const result = await runCompaction(
				this.dependencies.backend(),
				messages,
				before,
				{
					reason,
					presetSummary: preResult?.summary,
					temperature: config.temperature,
					maxTokens: config.maxTokens,
					mode: effectiveMode,
					...(frameOptions ? { frameOptions } : {}),
				},
			);

			if (
				!result.changed ||
				result.tokensAfter >= before ||
				!this.dependencies.commitHistory(messages, toMessages(result.messages))
			) {
				return await finishUnchanged(before);
			}

			await this.recordCompaction(
				toMessages(result.messages),
				Promise.resolve(before),
			);
			await emitPostCompact();
			await this.dependencies.extensionRunner()?.emit({
				type: "session_compact",
				context: {
					...identity,
					reason,
					tokensBefore: before,
					tokensAfter: result.tokensAfter,
					changed: true,
					messages: [...result.messages],
				},
			});
			this.dependencies.emit({
				type: "compaction",
				reason,
				tokensBefore: before,
				tokensAfter: result.tokensAfter,
			});
			return before - result.tokensAfter;
		} finally {
			// Hook cleanup is guaranteed even when the backend or an extension fails.
			await emitPostCompact();
		}
	}
}
