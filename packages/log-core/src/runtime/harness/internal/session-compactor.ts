import type { LLMBackend } from "../../../capabilities/provider/backend.ts";
import type { ExtensionRunner } from "../../../system/extension/runner.ts";
import type { AgentConfig } from "../../../system/types/types-config.ts";
import type {
	BeforeCompactContext,
	BeforeCompactResult,
	Message,
} from "../../../system/types/types-messages.ts";
import type { CompactionSettings } from "../../compaction/engine.ts";
import {
	runCompaction,
	shouldAutoCompact,
} from "../../compaction/orchestration.ts";

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
	estimateTokens: () => number;
	emit: (event: {
		type: "compaction";
		reason: CompactionReason;
		tokensBefore?: number;
		tokensAfter?: number;
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

	shouldCompact(messages: Message[] = this.dependencies.history()): boolean {
		return shouldAutoCompact(this.settings, messages);
	}

	recordCompaction(messages: Message[], tokensBefore: number): void {
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
			tokensBefore,
			firstKeptEntryId,
		);
	}

	async compact(reason: CompactionReason, force: boolean): Promise<number> {
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
			const before = this.dependencies.estimateTokens();
			if (!force && !this.shouldCompact(messages)) {
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
			const result = await runCompaction(
				this.dependencies.backend(),
				messages,
				before,
				{
					reason,
					presetSummary: preResult?.summary,
					temperature: config.temperature,
					maxTokens: config.maxTokens,
					thinkingLevel: config.thinkingLevel,
				},
			);

			if (
				!result.changed ||
				result.tokensAfter >= before ||
				!this.dependencies.commitHistory(messages, result.messages)
			) {
				return await finishUnchanged(before);
			}

			this.recordCompaction(result.messages, before);
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
