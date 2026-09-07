// ── skill:// protocol handler ────────────────────────────────────────────────
// Resolves skill names to their SKILL.md content.
// URL forms:
//   skill://<name> — Reads SKILL.md
//   skill://<name>/<path> — Reads relative path within skill's baseDir

import type { InternalResource, InternalUrl, ProtocolHandler, ResolveContext } from "./types";

export class SkillProtocolHandler implements ProtocolHandler {
	readonly scheme = "skill";
	readonly immutable = true;

	async resolve(url: InternalUrl, context?: ResolveContext): Promise<InternalResource> {
		const skills = context?.skills;
		const skillName = url.rawHost || url.hostname;

		if (!skillName) {
			const names = skills?.map(s => s.name) ?? [];
			throw new Error(
				`skill:// URL requires a skill name: skill://<name>\nAvailable: ${names.join(", ") || "none"}`,
			);
		}

		const skill = skills?.find(s => s.name === skillName);
		if (!skill) {
			const available = skills?.map(s => s.name) ?? [];
			throw new Error(
				`Unknown skill: ${skillName}\nAvailable: ${available.join(", ") || "none"}`,
			);
		}

		const content = skill.content;
		return {
			url: url.href,
			content,
			contentType: "text/markdown",
			size: Buffer.byteLength(content, "utf-8"),
			sourcePath: skill.path,
			notes: [],
		};
	}

	async complete(_query: string, context?: ResolveContext): Promise<
		Array<{ value: string; description?: string }>
	> {
		return (context?.skills ?? []).map(skill => ({
			value: skill.name,
			description: skill.name,
		}));
	}
}
