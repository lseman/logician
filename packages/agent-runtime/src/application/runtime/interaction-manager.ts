import type { AskUserContext } from "@logician/agent-core";
import {
	PermissionManager,
	type PermissionMode,
	type PermissionRules,
} from "@logician/agent-core/permissions";
import type { RuntimeEvent } from "@logician/agent-protocol";

export type PermissionDecision = "allow" | "deny" | "always";

export interface InteractionManagerOptions {
	mode: PermissionMode;
	rules?: PermissionRules;
	emit(event: RuntimeEvent): void;
}

/** Owns interactive requests and their pending resolver lifecycle. */
export class InteractionManager {
	readonly permissions: PermissionManager;
	private readonly permissionResolvers = new Map<
		string,
		(decision: PermissionDecision) => void
	>();
	private readonly questionResolvers = new Map<
		string,
		(answer: string) => void
	>();
	private questionSequence = 0;

	constructor(private readonly options: InteractionManagerOptions) {
		this.permissions = new PermissionManager({
			mode: options.mode,
			rules: options.rules,
		});
	}

	requestPermission(context: {
		toolName: string;
		toolCallId: string;
		args: Record<string, unknown>;
	}): Promise<PermissionDecision> {
		return new Promise(resolve => {
			this.permissionResolvers.set(context.toolCallId, resolve);
			this.options.emit({
				type: "permission_request",
				toolName: context.toolName,
				toolCallId: context.toolCallId,
				args: context.args,
			});
		});
	}

	requestQuestion(context: AskUserContext): Promise<string> {
		return new Promise(resolve => {
			const questionId = `question-${++this.questionSequence}`;
			this.questionResolvers.set(questionId, resolve);
			this.options.emit({
				type: "question_request",
				questionId,
				questions: context.questions,
			});
		});
	}

	respondToPermission(id: string, decision: PermissionDecision): boolean {
		const resolve = this.permissionResolvers.get(id);
		if (!resolve) return false;
		this.permissionResolvers.delete(id);
		resolve(decision);
		return true;
	}

	respondToQuestion(id: string, answer: string): boolean {
		const resolve = this.questionResolvers.get(id);
		if (!resolve) return false;
		this.questionResolvers.delete(id);
		resolve(answer);
		return true;
	}

	denyPending(): void {
		for (const [id, resolve] of this.permissionResolvers) {
			this.permissionResolvers.delete(id);
			resolve("deny");
		}
		for (const [id, resolve] of this.questionResolvers) {
			this.questionResolvers.delete(id);
			resolve("__dismissed__");
		}
	}

	setMode(mode: PermissionMode): void {
		this.permissions.setMode(mode);
		this.options.emit({
			type: "notice",
			level: "info",
			label: "Permissions",
			text: `mode: ${mode}`,
		});
	}

	get mode(): PermissionMode {
		return this.permissions.getMode();
	}
}
