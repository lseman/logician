import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { formatSkillInvocation, loadSkills } from "../tools/shared/skills.ts";

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
	assert.deepEqual(skill.allowedTools, ["bash", "read_file"]);
	assert.equal(skill.argumentHint, "<environment>");
	assert.equal(skill.model, "small-fast");

	const invocation = formatSkillInvocation(skill, "User arguments: staging");
	assert.match(invocation, /Step 1: check the thing\./);
	assert.match(invocation, /Preferred tools.*bash, read_file/);
	assert.match(invocation, /User arguments: staging/);
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
	assert.ok(diagnostics.some((d) => d.code === "invalid_metadata"));
});
