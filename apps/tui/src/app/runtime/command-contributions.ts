import type { AgentRuntime } from "@logician/log-runtime/application";
import type { SlashCommandDef } from "@logician/log-runtime/commands";
import type { Transcript } from "@logician/log-runtime/sessions";
import type { SlashPopup } from "../../overlays/slash-popup.ts";
import type { TranscriptDisplay } from "../../rendering/transcript/display.ts";
import type { TuiHandle } from "../../terminal/core.ts";

export interface CommandContributionCtx {
	bridge: AgentRuntime;
	slashPopup: SlashPopup;
	transcript: Transcript;
	transcriptDisplay: TranscriptDisplay;
	tui: TuiHandle;
}

/** Register runtime-discovered commands behind one contribution seam. */
export function registerRuntimeCommandContributions(
	ctx: CommandContributionCtx,
): void {
	const existing = ctx.slashPopup.getCommands() as SlashCommandDef[];
	const taken = new Set(existing.map(command => command.command));
	const additions: SlashCommandDef[] = [];

	for (const command of ctx.bridge.getExtensionCommands()) {
		const name = `/${command.name}`;
		if (taken.has(name)) continue;
		taken.add(name);
		additions.push({
			command: name,
			usage: command.usage ?? name,
			description: command.description,
			dispatch: "local",
			acceptsArgs: command.acceptsArgs ?? true,
			source: "extension",
			bridgeHandler: args => {
				void ctx.bridge
					.invokeExtensionCommand(command.name, args)
					.then(result => {
						if (result) ctx.transcript.addSystemMessage(result);
						ctx.transcriptDisplay.setTurns(ctx.transcript.getTurns());
						ctx.tui.requestRender();
					});
			},
		});
	}

	for (const skill of ctx.bridge.getSkills()) {
		const name = `/${skill.slashName}`;
		if (taken.has(name)) continue;
		taken.add(name);
		additions.push({
			command: name,
			usage: `${name}${skill.argumentHint ? ` ${skill.argumentHint}` : ""}`,
			description: `Skill: ${skill.description.slice(0, 80)}`,
			dispatch: "local",
			acceptsArgs: true,
			bridgeHandler: args => {
				ctx.bridge.invokeSkill(skill.name, args);
			},
		});
	}

	for (const prompt of ctx.bridge.getPrompts()) {
		const name = `/${prompt.slashName}`;
		if (taken.has(name)) continue;
		taken.add(name);
		additions.push({
			command: name,
			usage: `${name}${prompt.argumentHint ? ` ${prompt.argumentHint}` : ""}`,
			description: `Prompt: ${prompt.description.slice(0, 80)}`,
			dispatch: "local",
			acceptsArgs: true,
			bridgeHandler: args => {
				ctx.bridge.invokePrompt(prompt.name, args);
			},
		});
	}

	if (additions.length) {
		ctx.slashPopup.setCommands([...existing, ...additions]);
	}
}
