import {
	EVAL_SCHEMA_VERSION,
	type EvalCorpus,
	type EvalTask,
} from "./types.ts";

function object(value: unknown, label: string): Record<string, unknown> {
	if (!value || typeof value !== "object" || Array.isArray(value))
		throw new Error(`${label} must be an object`);
	return value as Record<string, unknown>;
}

function nonEmpty(value: unknown, label: string): string {
	if (typeof value !== "string" || value.trim() === "")
		throw new Error(`${label} must be a non-empty string`);
	return value;
}

export function validateTask(value: unknown, label = "task"): EvalTask {
	const task = object(value, label);
	if (task.schemaVersion !== EVAL_SCHEMA_VERSION)
		throw new Error(`${label}.schemaVersion must be ${EVAL_SCHEMA_VERSION}`);
	nonEmpty(task.id, `${label}.id`);
	nonEmpty(task.title, `${label}.title`);
	nonEmpty(task.prompt, `${label}.prompt`);
	if (
		!["bugfix", "feature", "refactor", "docs", "investigation"].includes(
			String(task.kind),
		)
	)
		throw new Error(`${label}.kind is invalid`);
	const fixture = object(task.fixture, `${label}.fixture`);
	nonEmpty(fixture.repository, `${label}.fixture.repository`);
	nonEmpty(fixture.revision, `${label}.fixture.revision`);
	const agent = object(task.agent, `${label}.agent`);
	nonEmpty(agent.command, `${label}.agent.command`);
	if (!Array.isArray(task.graders) || task.graders.length === 0)
		throw new Error(`${label}.graders must contain at least one grader`);
	for (const [index, raw] of task.graders.entries()) {
		const grader = object(raw, `${label}.graders[${index}]`);
		nonEmpty(grader.id, `${label}.graders[${index}].id`);
		if (
			!["command", "file_contains", "file_absent", "diff_scope"].includes(
				String(grader.type),
			)
		)
			throw new Error(`${label}.graders[${index}].type is invalid`);
		if (grader.type === "command")
			nonEmpty(grader.command, `${label}.graders[${index}].command`);
		if (grader.type === "file_contains") {
			nonEmpty(grader.path, `${label}.graders[${index}].path`);
			nonEmpty(grader.pattern, `${label}.graders[${index}].pattern`);
		}
		if (grader.type === "file_absent")
			nonEmpty(grader.path, `${label}.graders[${index}].path`);
		if (grader.type === "diff_scope") {
			nonEmpty(grader.baseRef, `${label}.graders[${index}].baseRef`);
			if (
				!Array.isArray(grader.allowedPaths) ||
				grader.allowedPaths.length === 0
			)
				throw new Error(
					`${label}.graders[${index}].allowedPaths must not be empty`,
				);
		}
	}
	return value as EvalTask;
}

export function validateCorpus(value: unknown): EvalCorpus {
	const corpus = object(value, "corpus");
	if (corpus.schemaVersion !== EVAL_SCHEMA_VERSION)
		throw new Error(`corpus.schemaVersion must be ${EVAL_SCHEMA_VERSION}`);
	nonEmpty(corpus.name, "corpus.name");
	if (!Array.isArray(corpus.tasks) || corpus.tasks.length === 0)
		throw new Error("corpus.tasks must contain at least one task");
	const ids = new Set<string>();
	for (const [index, raw] of corpus.tasks.entries()) {
		const task = validateTask(raw, `corpus.tasks[${index}]`);
		if (ids.has(task.id)) throw new Error(`duplicate task id: ${task.id}`);
		ids.add(task.id);
	}
	return value as EvalCorpus;
}
