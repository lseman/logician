// ── InteractionCoordinator ────────────────────────────────────────────────────
// Owns the two "pending resolver" maps AgentCoreBridge uses to turn async
// tool-driven permission/question requests into UI round-trips: register a
// resolver keyed by id, emit a *_request event, and let respondToX() resolve
// it later from the UI. Extracted from agent-bridge.ts.

import type { AgentConfig } from "@logician/agent-core";
import type { AskUserContext } from "@logician/agent-core/agent/types/types-tools.ts";
import {
	PermissionManager,
	type PermissionMode,
	type PermissionRules,
} from "@logician/agent-core/tools/shared/permissions.ts";
import type { ParsedBridgeEvent } from "../runtime/events.ts";

export interface InteractionCoordinatorDeps {
	emit: (event: ParsedBridgeEvent) => void;
	permissionMode: PermissionMode;
	permissionRules?: PermissionRules;
}

export class InteractionCoordinator {
	private readonly permissionManager: PermissionManager;
	private readonly emit: (event: ParsedBridgeEvent) => void;
	private readonly permissionResolvers = new Map<
		string,
		(decision: "allow" | "deny" | "always") => void
	>();
	private readonly questionResolvers = new Map<
		string,
		{ allow: (answer: string) => void; deny: () => void }
	>();

	constructor(deps: InteractionCoordinatorDeps) {
		this.emit = deps.emit;
		this.permissionManager = new PermissionManager({
			mode: deps.permissionMode,
			rules: deps.permissionRules,
		});
	}

	getPermissionManager(): PermissionManager {
		return this.permissionManager;
	}

	/** Build the AgentConfig callbacks that route tool-driven requests here. */
	buildConfigCallbacks(): Pick<
		AgentConfig,
		"onPermissionRequest" | "onQuestionRequest"
	> {
		return {
			onPermissionRequest: ctx =>
				new Promise(resolve => {
					this.permissionResolvers.set(ctx.toolCallId, resolve);
					this.emit({
						type: "permission_request",
						tool_name: ctx.toolName,
						tool_call_id: ctx.toolCallId,
						args: ctx.args,
					});
				}),
			onQuestionRequest: (ctx: AskUserContext) =>
				new Promise<string>(resolve => {
					const questionId = `q_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
					this.questionResolvers.set(questionId, {
						allow: resolve,
						deny: () => resolve("__dismissed__"),
					});
					this.emit({
						type: "question_request",
						question_id: questionId,
						questions: ctx.questions,
					});
				}),
		};
	}

	// ── Permissions ────────────────────────────────────────────────────────

	/** Answer a pending permission_request. Returns false for unknown ids. */
	respondToPermission(
		toolCallId: string,
		decision: "allow" | "deny" | "always",
	): boolean {
		const resolve = this.permissionResolvers.get(toolCallId);
		if (!resolve) return false;
		this.permissionResolvers.delete(toolCallId);
		resolve(decision);
		return true;
	}

	/** True while a permission_request awaits a decision. */
	hasPendingPermission(): boolean {
		return this.permissionResolvers.size > 0;
	}

	/** Deny every pending permission request (abort / shutdown). */
	denyPendingPermissions(): void {
		for (const [id, resolve] of this.permissionResolvers) {
			this.permissionResolvers.delete(id);
			resolve("deny");
		}
	}

	setPermissionMode(mode: PermissionMode): void {
		this.permissionManager.setMode(mode);
		this.emit({
			type: "notice",
			level: "info",
			label: "Permissions",
			text: `mode: ${mode}`,
		});
	}

	getPermissionMode(): PermissionMode {
		return this.permissionManager.getMode();
	}

	// ── Interactive questions ────────────────────────────────────────────

	/**
	 * Register a pending question and emit it to the UI. Returns the question id
	 * so the agent can track which question it asked. Call respondToQuestion() to
	 * resolve it.
	 */
	askQuestion(
		question: string,
		choices: Array<{ value: string; label: string }>,
	): string {
		const questionId = `q_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
		this.questionResolvers.set(questionId, {
			allow: (_ans: string) => {},
			deny: () => {},
		});
		this.emit({
			type: "question_request",
			question_id: questionId,
			questions: [{ id: "answer", question, choices }],
		});
		return questionId;
	}

	/**
	 * Answer a pending question by id. The answer is forwarded to the agent's
	 * resolver. Returns false if the question id is unknown.
	 */
	respondToQuestion(questionId: string, answer: string): boolean {
		const resolver = this.questionResolvers.get(questionId);
		if (!resolver) return false;
		this.questionResolvers.delete(questionId);
		resolver.allow(answer);
		return true;
	}

	/** True while a question_request awaits an answer. */
	hasPendingQuestion(): boolean {
		return this.questionResolvers.size > 0;
	}
}
