// ── Trust checker — determines which directories require trust decisions ─────
// Walks up the directory tree from cwd and checks for trust-requiring resources.
// Resources: settings.json, extensions/, skills/, prompts/, themes/, SYSTEM.md,
//            APPEND_SYSTEM.md, .agents/skills/

import { existsSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { homedir } from "node:os";

const TRUST_RESOURCES = [
	"settings.json",
	"extensions",
	"skills",
	"prompts",
	"themes",
	"SYSTEM.md",
	"APPEND_SYSTEM.md",
];

const CONFIG_DIR = ".logician";

const TRUST_REQUIRING_NAMES = new Set([
	...TRUST_RESOURCES,
	...CONFIG_DIR,
	".agents",
]);

function hasTrustRequiringResourcesInDir(dir: string): boolean {
	try {
		const entries = readdirSync(dir, { withFileTypes: true });
		for (const entry of entries) {
			if (!TRUST_REQUIRING_NAMES.has(entry.name)) continue;
			if (entry.isDirectory()) return true;
			if (entry.isFile() && TRUST_RESOURCES.includes(entry.name)) return true;
		}
	} catch (e: unknown) {
		// directory unreadable — skip
	}
	return false;
}

function hasAgentsSkills(cwd: string): boolean {
	let currentDir = cwd.replace(/\/+$/, "");
	const home = homedir();

	while (true) {
		const agentsSkills = join(currentDir, ".agents", "skills");
		if (existsSync(agentsSkills)) {
			// Don't flag the user's global .agents/skills
			if (agentsSkills.startsWith(home)) continue;
			return true;
		}
		const parent = dirname(currentDir);
		if (parent === currentDir) break;
		currentDir = parent;
	}
	return false;
}

/**
 * Returns true when cwd has project-local resources that require trust.
 */
export function hasTrustRequiringProjectResources(cwd: string): boolean {
	let currentDir = cwd.replace(/\/+$/, "");

	// Check for .logician directory with trust-requiring resources
	while (true) {
		const configDir = join(currentDir, CONFIG_DIR);
		if (existsSync(configDir) && hasTrustRequiringResourcesInDir(configDir)) {
			return true;
		}
		const parent = dirname(currentDir);
		if (parent === currentDir) break;
		currentDir = parent;
	}

	return hasAgentsSkills(cwd);
}

/**
 * Returns the list of trust-requiring paths found under the given cwd.
 */
export function getTrustRequiringPaths(cwd: string): string[] {
	const paths: string[] = [];
	let currentDir = cwd.replace(/\/+$/, "");

	while (true) {
		const configDir = join(currentDir, CONFIG_DIR);
		if (existsSync(configDir)) {
			for (const resource of TRUST_RESOURCES) {
				const resourcePath = join(configDir, resource);
				if (existsSync(resourcePath)) {
					paths.push(resourcePath);
				}
			}
		}
		const parent = dirname(currentDir);
		if (parent === currentDir) break;
		currentDir = parent;
	}

	// Check for .agents/skills
	const agentsSkills = join(cwd, ".agents", "skills");
	if (existsSync(agentsSkills)) {
		paths.push(agentsSkills);
	}

	return paths;
}

/**
 * Formats a trust prompt for display to the user.
 */
export function formatTrustPrompt(cwd: string, paths: string[]): string {
	const pathsSection = paths.length > 0
		? `\n\nTrust-requiring resources found:\n  ${paths.join("\n  ")}`
		: "";

	return `Trust project folder?\n${cwd}${pathsSection}\n\nThis allows Logician to load local settings, extensions, skills, and execute project resources.`;
}
