import { test } from "bun:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	findSkillByName,
	formatSkillCatalog,
	formatSkillInvocation,
	loadSkills,
} from "../features/skills/index.ts";

void test("frontmatter extensions are parsed (allowed-tools, argument-hint, model)", async () => {
	const root = mkdtempSync(join(tmpdir(), "skills-"));
	const dir = join(root, "deploy-check");
	mkdirSync(dir);
	writeFileSync(
		join(dir, "SKILL.md"),
		[
			"---",
			"name: deploy-check",
			"description: Verify a deployment",
			"allowed-tools: bash, read_file",
			"argument-hint: <environment>",
			"model: small-fast",
			"---",
			"Step 1: check the thing.",
		].join("\n"),
		"utf8",
	);

	const { skills, diagnostics } = await loadSkills(root);
	assert.equal(diagnostics.length, 0);
	assert.equal(skills.length, 1);
	const skill = skills[0];
	assert.equal(skill.name, "deploy-check");
	assert.equal(skill.displayName, "deploy-check");
	assert.equal(skill.slashName, "deploy-check");
	assert.deepEqual(skill.allowedTools, ["bash", "read_file"]);
	assert.equal(skill.argumentHint, "<environment>");
	assert.equal(skill.model, "small-fast");

	const invocation = formatSkillInvocation(skill, "User arguments: staging");
	assert.match(invocation, /Step 1: check the thing\./);
	assert.match(invocation, /Preferred tools.*bash, read_file/);
	assert.match(invocation, /User arguments: staging/);
});

void test("openclaude-style nested skills keep path ids and display names", async () => {
	const root = mkdtempSync(join(tmpdir(), "skills-"));
	const parent = join(root, "academic");
	const child = join(parent, "semantic_scholar");
	mkdirSync(child, { recursive: true });
	writeFileSync(
		join(parent, "SKILL.md"),
		[
			"---",
			"name: Academic",
			"description: Use for academic literature discovery.",
			"aliases:",
			"  - literature discovery",
			"triggers:",
			"  - find academic papers",
			"preferred_tools:",
			"  - s2_search",
			"next_skills:",
			"  - academic/semantic_scholar",
			"---",
			"Parent skill body.",
		].join("\n"),
		"utf8",
	);
	writeFileSync(
		join(child, "SKILL.md"),
		[
			"---",
			"name: Semantic Scholar",
			"description: Search Semantic Scholar papers.",
			"---",
			"Child skill body.",
		].join("\n"),
		"utf8",
	);

	const { skills, diagnostics } = await loadSkills(root);
	assert.equal(diagnostics.length, 0);
	assert.deepEqual(skills.map(s => s.name).sort(), [
		"academic",
		"academic/semantic_scholar",
	]);

	const academic = findSkillByName(skills, "Academic");
	assert.ok(academic);
	assert.equal(academic.name, "academic");
	assert.equal(academic.displayName, "Academic");
	assert.equal(academic.slashName, "academic");
	assert.deepEqual(academic.aliases, ["literature discovery"]);
	assert.deepEqual(academic.triggers, ["find academic papers"]);
	assert.deepEqual(academic.allowedTools, ["s2_search"]);
	assert.deepEqual(academic.nextSkills, ["academic/semantic_scholar"]);

	const childSkill = findSkillByName(skills, "academic/semantic_scholar");
	assert.ok(childSkill);
	assert.equal(childSkill.displayName, "Semantic Scholar");
	assert.equal(childSkill.slashName, "semantic_scholar");
});

void test("skill catalog and invocation render openclaude metadata and resources", async () => {
	const root = mkdtempSync(join(tmpdir(), "skills-"));
	const skillDir = join(root, "coding", "file_ops");
	mkdirSync(join(skillDir, "scripts"), { recursive: true });
	mkdirSync(join(skillDir, "references"), { recursive: true });
	writeFileSync(
		join(skillDir, "scripts", "file_ops.py"),
		"print('ok')",
		"utf8",
	);
	writeFileSync(
		join(skillDir, "references", "workflow.md"),
		"# workflow",
		"utf8",
	);
	writeFileSync(
		join(skillDir, "SKILL.md"),
		[
			"---",
			"name: File Ops",
			"description: Use for local filesystem operations.",
			"preferred_tools: read_file, edit_file",
			"example_queries:",
			"  - inspect this file",
			"when_not_to_use:",
			"  - remote-only task",
			"---",
			"See `scripts/file_ops.py` for implementation details.",
		].join("\n"),
		"utf8",
	);

	const { skills } = await loadSkills(root);
	const skill = findSkillByName(skills, "File Ops");
	assert.ok(skill);
	assert.equal(skill.name, "coding/file_ops");
	assert.equal(skill.slashName, "file_ops");

	const catalog = formatSkillCatalog(skills);
	assert.match(catalog, /name="coding\/file_ops"/);
	assert.match(catalog, /display_name="File Ops"/);
	assert.match(catalog, /preferred_tools="read_file, edit_file"/);

	const invocation = formatSkillInvocation(skill);
	assert.match(invocation, /<metadata>/);
	assert.match(invocation, /<preferred_tools>/);
	assert.match(invocation, /scripts\/file_ops\.py|file_ops\.py/);
	assert.match(invocation, /workflow\.md/);
});

void test("skill invocation bounds resource traversal while scanning", async () => {
	const root = mkdtempSync(join(tmpdir(), "skills-"));
	const skillDir = join(root, "bounded-resources");
	const references = join(skillDir, "references");
	mkdirSync(references, { recursive: true });
	writeFileSync(
		join(skillDir, "SKILL.md"),
		"---\nname: bounded-resources\ndescription: Test bounded resources.\n---\nBody.",
		"utf8",
	);
	for (let index = 0; index < 50; index++) {
		writeFileSync(
			join(references, `${String(index).padStart(2, "0")}.md`),
			"x",
		);
	}

	const { skills } = await loadSkills(root);
	const skill = findSkillByName(skills, "bounded-resources");
	assert.ok(skill);
	const invocation = formatSkillInvocation(skill);
	assert.equal(invocation.match(/<file>/g)?.length, 40);
	assert.doesNotMatch(invocation, /49\.md/);
});

void test("metadata-only and lenient frontmatter skills are accepted", async () => {
	const root = mkdtempSync(join(tmpdir(), "skills-"));
	const dir = join(root, "coding", "web");
	mkdirSync(dir, { recursive: true });
	writeFileSync(
		join(dir, "SKILL.md"),
		[
			"---",
			"name: Web",
			"description: Use for web-related tasks: fetching documentation, inspecting REST APIs.",
			"preferred_tools:",
			"  - fetch_url",
		].join("\n"),
		"utf8",
	);

	const { skills, diagnostics } = await loadSkills(root);
	assert.equal(diagnostics.length, 0);
	const skill = findSkillByName(skills, "Web");
	assert.ok(skill);
	assert.equal(skill.name, "coding/web");
	assert.equal(
		skill.description,
		"Use for web-related tasks: fetching documentation, inspecting REST APIs.",
	);
	assert.deepEqual(skill.allowedTools, ["fetch_url"]);
});

void test("skill without a description is rejected with a diagnostic", async () => {
	const root = mkdtempSync(join(tmpdir(), "skills-"));
	const dir = join(root, "nameless");
	mkdirSync(dir);
	writeFileSync(
		join(dir, "SKILL.md"),
		"---\nname: nameless\n---\nBody.",
		"utf8",
	);
	const { skills, diagnostics } = await loadSkills(root);
	assert.equal(skills.length, 0);
	assert.ok(diagnostics.some(d => d.code === "invalid_metadata"));
});

void test("skill catalog is bounded while retaining every skill name", () => {
	const skills = Array.from({ length: 40 }, (_, index) => ({
		name: `skill-${index}`,
		displayName: `Skill ${index}`,
		description: `A deliberately long description for skill ${index} `.repeat(
			8,
		),
		content: "body",
		filePath: `/skills/${index}/SKILL.md`,
		baseDir: `/skills/${index}`,
		slashName: `skill-${index}`,
		disableModelInvocation: false,
		source: "user" as const,
	}));
	const catalog = formatSkillCatalog(skills, { maxChars: 4_000 });
	assert.ok(catalog.length <= 4_000);
	for (const skill of skills)
		assert.match(catalog, new RegExp(`name="${skill.name}"`));
});
