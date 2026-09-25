// ── skill:// protocol handler ────────────────────────────────────────────────
// Resolves skill names to their SKILL.md content.
// URL forms:
//   skill://<name> — Reads SKILL.md

import type {
	InternalResource,
	InternalUrl,
	ProtocolHandler,
	ResolveContext,
} from "./types";

/** Edit distance for short strings (skill names). */
function editDistance(a: string, b: string): number {
	const dp = Array.from({ length: b.length + 1 }, (_, j) => j);
	for (let i = 1; i <= a.length; i++) {
		let prev = dp[0];
		dp[0] = i;
		for (let j = 1; j <= b.length; j++) {
			const tmp = dp[j];
			dp[j] = Math.min(
				dp[j] + 1,
				dp[j - 1] + 1,
				prev + (a[i - 1] === b[j - 1] ? 0 : 1),
			);
			prev = tmp;
		}
	}
	return dp[b.length];
}

/** Closest available skill name, or null when nothing is close enough. */
function closestSkillName(name: string, available: string[]): string | null {
	let best: string | null = null;
	let bestDist = Math.max(2, Math.floor(name.length / 3));
	for (const candidate of available) {
		// Suggestions should survive a stray case slip; resolution stays exact.
		const dist = Math.min(
			editDistance(name, candidate),
			editDistance(name.toLowerCase(), candidate.toLowerCase()),
		);
		if (dist < bestDist) {
			best = candidate;
			bestDist = dist;
		}
	}
	return best;
}

export class SkillProtocolHandler implements ProtocolHandler {
	readonly scheme = "skill";
	readonly immutable = true;

	async resolve(
		url: InternalUrl,
		context?: ResolveContext,
	): Promise<InternalResource> {
		const skills = context?.skills;
		const skillName = url.host;

		if (!skillName) {
			const names = skills?.map(s => s.name) ?? [];
			throw new Error(
				`skill:// URL requires a skill name: skill://<name>\nAvailable: ${names.join(", ") || "none"}`,
			);
		}

		if (url.target !== url.host && url.target !== `${url.host}/`) {
			throw new Error(
				"skill:// supports an exact name only; paths, queries and fragments are not supported.",
			);
		}

		const skill = skills?.find(s => s.name === skillName);
		if (!skill) {
			const available = skills?.map(s => s.name) ?? [];
			const suggestion = closestSkillName(skillName, available);
			throw new Error(
				`Unknown skill: ${skillName}. The skill name must be an exact match from the available list. Do not guess or infer alternative names.${suggestion ? ` Did you mean: ${suggestion}?` : ""} Available: ${available.join(", ") || "none"}`,
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

	async complete(
		_query: string,
		context?: ResolveContext,
	): Promise<Array<{ value: string; description?: string }>> {
		return (context?.skills ?? []).map(skill => ({
			value: skill.name,
			description: skill.name,
		}));
	}
}
