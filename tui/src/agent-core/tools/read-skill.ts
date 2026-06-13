// ── read_skill tool ───────────────────────────────────────────────────────────
// Loads a named skill's full instructions on demand. The system prompt only
// advertises a compact catalog (name + description); the model calls this tool
// to pull the full SKILL.md body when it decides to use a skill.

import { formatSkillInvocation, type Skill } from "../skills.ts";
import type { Tool } from "../types.ts";

/**
 * Build a read_skill tool bound to the given skills. Pass the skills loaded at
 * startup; the tool resolves a name to its full body. Returns null when there
 * are no skills so the caller can skip registering it.
 */
export function createReadSkillTool(skills: Skill[]): Tool | null {
	if (!skills.length) return null;
	const byName = new Map(skills.map((s) => [s.name, s]));

	return {
		name: "read_skill",
		readOnly: true,
		executionMode: "parallel",
		description:
			"Load a skill's full instructions by name. Call this when a skill from " +
			"the <available-skills> catalog applies to the task, then follow the " +
			"returned instructions.",
		parameters: {
			type: "object",
			properties: {
				name: {
					type: "string",
					description: "The skill name, exactly as listed in the catalog.",
				},
			},
			required: ["name"],
		},
		prepareArguments: (raw): Record<string, unknown> => {
			if (typeof raw === "string") return { name: raw };
			if (!raw || typeof raw !== "object") return {};
			return raw as Record<string, unknown>;
		},
		execute: async (args) => {
			const name = typeof args.name === "string" ? args.name : "";
			const skill = byName.get(name);
			if (!skill) {
				const available = [...byName.keys()].join(", ") || "(none)";
				return `Error: Unknown skill "${name}". Available: ${available}`;
			}
			return formatSkillInvocation(skill);
		},
	};
}
