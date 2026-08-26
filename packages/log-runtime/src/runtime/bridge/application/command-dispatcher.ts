import type { RuntimeEvent } from "@logician/log-core/events";
import type { AgentSession } from "@logician/log-core/session";
import {
	findPromptByName,
	type Prompt,
} from "../../../capabilities/prompts/loader.ts";
import {
	findSkillByName,
	formatSkillInvocation,
	type Skill,
} from "../../../capabilities/skills/loader.ts";
import { runQueueCommand } from "./queue-command.ts";

export interface CommandDispatcherDependencies {
	session: () => AgentSession | null;
	skills: () => Skill[];
	prompts: () => Prompt[];
	sendMessage: (message: string) => Promise<void>;
	reload: () => Promise<void>;
	emit: (event: RuntimeEvent) => void;
	reportError: (error: unknown) => void;
}

/** Interprets user-facing commands and converts them into runtime actions. */
export class CommandDispatcher {
	private readonly dependencies: CommandDispatcherDependencies;

	constructor(dependencies: CommandDispatcherDependencies) {
		this.dependencies = dependencies;
	}

	dispatchSlash(raw: string): void {
		const trimmed = raw.trim();
		const result = runQueueCommand(this.dependencies.session(), trimmed);
		if (result) {
			this.dependencies.emit({
				type: "notice",
				level: result.level,
				label: "Queue",
				text: result.text,
			});
			return;
		}
		if (trimmed === "/reload") {
			this.settle(this.dependencies.reload());
			return;
		}
		this.settle(this.dependencies.sendMessage(raw));
	}

	invokeSkill(name: string, args: string): boolean {
		const skill = findSkillByName(this.dependencies.skills(), name);
		if (!skill) return false;

		const trimmedArgs = args.trim();
		const substitutes = skill.content.includes("$ARGUMENTS");
		const effective = substitutes
			? {
					...skill,
					content: skill.content.replaceAll("$ARGUMENTS", trimmedArgs),
				}
			: skill;
		const message = formatSkillInvocation(
			effective,
			trimmedArgs && !substitutes
				? `User arguments for this skill invocation: ${trimmedArgs}`
				: undefined,
		);
		this.settle(this.dependencies.sendMessage(message));
		return true;
	}

	invokePrompt(name: string, args: string): boolean {
		const prompt = findPromptByName(this.dependencies.prompts(), name);
		if (!prompt) return false;

		const trimmedArgs = args.trim();
		const message = prompt.content.includes("$ARGUMENTS")
			? prompt.content.replaceAll("$ARGUMENTS", trimmedArgs)
			: trimmedArgs
				? `${prompt.content}\n\n${trimmedArgs}`
				: prompt.content;
		this.settle(this.dependencies.sendMessage(message));
		return true;
	}

	private settle(operation: Promise<void>): void {
		operation.catch(error => this.dependencies.reportError(error));
	}
}
