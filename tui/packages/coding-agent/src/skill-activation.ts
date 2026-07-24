import {
	formatSkillInvocation,
	type Skill,
} from "./skills.ts";

export interface SkillActivation {
	skill: Skill;
	score: number;
	reason: string;
}

export interface SkillActivationOptions {
	maxSkills?: number;
}

/**
 * Owns cross-turn activation state. User turns are freshly matched; only an
 * explicitly scheduled internal continuation inherits the prior skill set.
 */
export class SkillActivationSession {
	private continuation: SkillActivation[] = [];
	private reuseOnNextTurn = false;

	select(skills: Skill[], prompt: string): SkillActivation[] {
		if (this.reuseOnNextTurn) {
			this.reuseOnNextTurn = false;
			const inherited = this.continuation;
			this.continuation = [];
			return inherited;
		}
		this.continuation = [];
		return selectSkillsForPrompt(skills, prompt);
	}

	continueWith(activations: SkillActivation[]): void {
		this.continuation = [...activations];
		this.reuseOnNextTurn = activations.length > 0;
	}

	reset(): void {
		this.continuation = [];
		this.reuseOnNextTurn = false;
	}
}

const STOP_WORDS = new Set([
	"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
	"is", "it", "of", "on", "or", "the", "this", "to", "use", "when", "with",
]);

/**
 * Select the smallest strongly-relevant skill set for a user prompt.
 *
 * Metadata authored specifically for routing (names, aliases, triggers, and
 * examples) dominates description overlap. Skills hidden from model invocation
 * are deliberately excluded; they remain available through slash commands.
 */
export function selectSkillsForPrompt(
	skills: Skill[],
	prompt: string,
	options: SkillActivationOptions = {},
): SkillActivation[] {
	const normalizedPrompt = normalize(prompt);
	const explicitPrompt = prompt.normalize("NFKC").toLowerCase();
	if (!normalizedPrompt) return [];
	const promptTokens = new Set(tokens(normalizedPrompt));
	const matches = skills
		.filter((skill) => !skill.disableModelInvocation)
		.map((skill) =>
			scoreSkill(skill, normalizedPrompt, explicitPrompt, promptTokens),
		)
		.filter((match): match is SkillActivation => match !== null)
		.sort((a, b) =>
			b.score - a.score ||
			a.skill.name.localeCompare(b.skill.name),
		);
	if (!matches.length) return [];

	const explicit = matches.filter((match) => match.score >= 100);
	if (explicit.length) return explicit.slice(0, options.maxSkills ?? 3);

	const floor = Math.max(10, matches[0].score * 0.75);
	const selected: SkillActivation[] = [];
	for (const match of matches) {
		if (match.score < floor) break;
		if (selected.some((current) => sameSkillFamily(current.skill, match.skill))) {
			continue;
		}
		selected.push(match);
		if (selected.length >= (options.maxSkills ?? 1)) break;
	}
	return selected;
}

export function formatActivatedSkills(activations: SkillActivation[]): string {
	if (!activations.length) return "";
	return [
		"<activated-skills>",
		"These skills were selected for the current user request. Follow their instructions for this turn.",
		...activations.map(({ skill }) => formatSkillInvocation(skill)),
		"</activated-skills>",
	].join("\n\n");
}

function scoreSkill(
	skill: Skill,
	prompt: string,
	explicitPrompt: string,
	promptTokens: Set<string>,
): SkillActivation | null {
	for (const excluded of skill.whenNotToUse ?? []) {
		if (phraseMatches(prompt, promptTokens, excluded, 1)) return null;
	}

	const lookupNames = [
		skill.name,
		skill.slashName,
		skill.displayName,
		...(skill.aliases ?? []),
	];
	for (const name of lookupNames) {
		const normalizedName = name.normalize("NFKC").toLowerCase();
		if (
			explicitPrompt.includes(`$${normalizedName}`) ||
			explicitPrompt.includes(`/${normalizedName}`)
		) {
			return { skill, score: 100, reason: `explicitly named ${name}` };
		}
	}

	let best: SkillActivation | null = null;
	const consider = (score: number, reason: string): void => {
		if (!best || score > best.score) best = { skill, score, reason };
	};

	for (const name of lookupNames) {
		if (phraseMatches(prompt, promptTokens, name, 0.66)) {
			consider(24, `name or alias matched "${name}"`);
		}
	}
	for (const trigger of skill.triggers ?? []) {
		if (phraseMatches(prompt, promptTokens, trigger, 0.66)) {
			consider(20, `trigger matched "${trigger}"`);
		}
	}
	for (const example of skill.exampleQueries ?? []) {
		if (phraseMatches(prompt, promptTokens, example, 0.7)) {
			consider(16, `example matched "${example}"`);
		}
	}

	const descriptionTokens = tokens(skill.description);
	const overlap = descriptionTokens.filter((token) => promptTokens.has(token)).length;
	if (overlap >= 3) consider(Math.min(15, 6 + overlap * 2), "description matched");
	const family = tokens(skill.name)[0];
	const matched = best as SkillActivation | null;
	if (matched && family && promptTokens.has(family)) {
		matched.score += 12;
		matched.reason += `; ${family} task`;
	}
	return matched;
}

function phraseMatches(
	prompt: string,
	promptTokens: Set<string>,
	value: string,
	minCoverage: number,
): boolean {
	const phrase = normalize(value);
	if (!phrase) return false;
	if (prompt.includes(phrase)) return true;
	const meaningful = tokens(phrase);
	if (!meaningful.length) return false;
	const overlap = meaningful.filter((token) => promptTokens.has(token)).length;
	return overlap >= 2 && overlap / meaningful.length >= minCoverage;
}

function tokens(value: string): string[] {
	return normalize(value)
		.split(" ")
		.filter((token) => token.length >= 2 && !STOP_WORDS.has(token));
}

function normalize(value: string): string {
	return value
		.normalize("NFKC")
		.toLowerCase()
		.replace(/[_/:.-]+/g, " ")
		.replace(/[^\p{L}\p{N}+$ ]+/gu, " ")
		.replace(/\s+/g, " ")
		.trim();
}

function sameSkillFamily(a: Skill, b: Skill): boolean {
	const family = (skill: Skill): string => skill.name.split(/[-/:]/, 1)[0];
	return family(a) === family(b);
}
