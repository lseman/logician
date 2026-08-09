// ── GSD Bridge — STATE.md management ────────────────────────────────────────
// Manages .planning/STATE.md — GSD's project memory document.
// Provides field extraction, field replacement, status transitions, and
// section-level operations (addDecision, addBlocker, addLearning, etc.)

import fs from "node:fs";
import path from "node:path";

const STATE_FILE = ".planning/STATE.md";

interface StateFrontmatter {
	status?: string;
	currentPhase?: string;
	currentPlan?: string;
	quickTasksCompleted?: number;
	lastUpdated?: string;
	[key: string]: unknown;
}

interface StateBody {
	sections: Array<{ heading: string; content: string }>;
}

export function readState(cwd: string): {
	frontmatter: StateFrontmatter;
	body: string;
} {
	const filePath = path.join(cwd, STATE_FILE);
	if (!fs.existsSync(filePath)) {
		throw new Error(`STATE.md not found. Run /gsd:new-project first.`);
	}
	const content = fs.readFileSync(filePath, "utf-8");
	return parseState(content);
}

function parseState(content: string): {
	frontmatter: StateFrontmatter;
	body: string;
} {
	const fmMatch = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
	if (!fmMatch) {
		return { frontmatter: {}, body: content };
	}
	const fm = fmMatch[1];
	const body = fmMatch[2];
	const frontmatter: StateFrontmatter = {};
	for (const line of fm.split("\n")) {
		const colonIdx = line.indexOf(":");
		if (colonIdx > -1) {
			const key = line.slice(0, colonIdx).trim();
			const val = line.slice(colonIdx + 1).trim();
			frontmatter[key] = val || undefined;
		}
	}
	return { frontmatter, body };
}

export function writeState(cwd: string, content: string): void {
	const filePath = path.join(cwd, STATE_FILE);
	const dir = path.dirname(filePath);
	if (!fs.existsSync(dir)) {
		fs.mkdirSync(dir, { recursive: true });
	}
	fs.writeFileSync(filePath, content);
}

export function getStateField(cwd: string, field: string): string | undefined {
	const { frontmatter } = readState(cwd);
	return String(frontmatter[field] ?? "");
}

export function setStateField(cwd: string, field: string, value: string): void {
	const { frontmatter, body } = readState(cwd);
	frontmatter[field] = value;
	const fmLines = Object.entries(frontmatter)
		.filter(([, v]) => v !== undefined)
		.map(([k, v]) => `${k}: ${v}`);
	writeState(cwd, `---\n${fmLines.join("\n")}\n---\n${body}`);
}

export function beginPhase(cwd: string, phaseId: string): void {
	setStateField(cwd, "currentPhase", phaseId);
	setStateField(cwd, "status", `Planning phase ${phaseId}`);
}

export function advancePlan(
	cwd: string,
	phaseId: string,
	planId: string,
): void {
	setStateField(cwd, "currentPhase", phaseId);
	setStateField(cwd, "currentPlan", planId);
	setStateField(cwd, "status", `Executing plan ${planId}`);
}

export function completePhase(cwd: string, phaseId: string): void {
	setStateField(cwd, "status", `Phase ${phaseId} complete`);
}

export function addDecision(
	cwd: string,
	decision: string,
	context: string,
): void {
	const { frontmatter, body } = readState(cwd);
	const section = `## Decisions\n\n### ${new Date().toISOString()}\n\n- **${context}**: ${decision}\n`;
	const decisionsIdx = body.indexOf("## Decisions");
	if (decisionsIdx > -1) {
		// Append to existing decisions section
		const afterDecisions = body.indexOf("\n##", decisionsIdx + 1);
		const insertAt = afterDecisions > -1 ? afterDecisions : body.length;
		writeState(cwd, body.slice(0, insertAt) + section + body.slice(insertAt));
	} else {
		writeState(cwd, `${body}\n${section}`);
	}
}

export function addBlocker(cwd: string, blocker: string): void {
	const { frontmatter, body } = readState(cwd);
	const section = `## Blockers\n\n### ${new Date().toISOString()}\n\n- ${blocker}\n`;
	const blockersIdx = body.indexOf("## Blockers");
	if (blockersIdx > -1) {
		const afterBlockers = body.indexOf("\n##", blockersIdx + 1);
		const insertAt = afterBlockers > -1 ? afterBlockers : body.length;
		writeState(cwd, body.slice(0, insertAt) + section + body.slice(insertAt));
	} else {
		writeState(cwd, `${body}\n${section}`);
	}
}

export function addLearning(cwd: string, learning: string): void {
	const { frontmatter, body } = readState(cwd);
	const section = `## Learnings\n\n### ${new Date().toISOString()}\n\n- ${learning}\n`;
	const learningsIdx = body.indexOf("## Learnings");
	if (learningsIdx > -1) {
		const afterLearnings = body.indexOf("\n##", learningsIdx + 1);
		const insertAt = afterLearnings > -1 ? afterLearnings : body.length;
		writeState(cwd, body.slice(0, insertAt) + section + body.slice(insertAt));
	} else {
		writeState(cwd, `${body}\n${section}`);
	}
}
