---
name: skill-design
description: Design and implement new Logician skills with proper structure, metadata, and integration.
aliases:
  - create skill
  - design skill
  - new skill
  - skill creation
  - skill development
  - add skill
triggers:
  - create a skill
  - design a skill
  - new skill
  - add a skill
  - skill development
  - implement skill
  - skill creation
  - make a skill
  - build a skill
  - skill authoring
preferred_tools:
  - write_file
  - read_file
  - edit_file
  - list_files
  - bash
example_queries:
  - create a skill for reviewing Python code
  - design a new skill for database migrations
  - add a skill that wraps the docker CLI
  - implement a skill for GraphQL API testing
when_not_to_use:
  - the user already has a skill file that just needs content editing (use edit_file directly)
  - the task is about using an existing skill rather than creating a new one
  - the skill already exists and only needs minor fixes
next_skills:
  - skill-review
entry_criteria:
  - the user wants to create a new skill from scratch
  - the user needs help structuring a SKILL.md file
  - the user wants to add a new capability to the agent
decision_rules:
  - start with a clear name (lowercase, hyphen-separated, max 64 chars)
  - write a concise description (one sentence, present tense)
  - include at least 3-5 trigger phrases the user would naturally say
  - list preferred_tools that match the skill's workflow
  - include example_queries that show real usage patterns
  - specify when_not_to_use to prevent misuse
  - add next_skills if the workflow chains to other skills
  - include implementation notes if the skill wraps external scripts/tools
anti_patterns:
  - creating a skill with too broad a description (e.g., "do everything")
  - using triggers that overlap with existing skills without clear differentiation
  - forgetting to list preferred_tools for tool-dependent skills
  - writing implementation that assumes tools the agent doesn't have
---

# Skill Design Guide

Use this skill when creating new Logician skills. Follow the conventions below to ensure skills integrate cleanly with the agent framework.

## Skill Structure

Every skill is a `SKILL.md` file with YAML frontmatter followed by body content.

### Frontmatter Fields

| Field | Required | Format | Notes |
|---|---|---|---|
| `name` | Yes | `lowercase-hyphenated` | Max 64 chars, matches directory name |
| `description` | Yes | One sentence, present tense | What the skill does, not how |
| `aliases` | Optional | List of strings | Alternative names for lookup |
| `triggers` | Yes | List of phrases | Natural-language phrases that activate this skill |
| `preferred_tools` | Optional | List of tool names | Tools the skill needs |
| `example_queries` | Optional | List of examples | Real user queries that would use this skill |
| `when_not_to_use` | Optional | List of conditions | Prevent misuse |
| `next_skills` | Optional | List of skill names | Skills to chain to next |
| `preferred_sequence` | Optional | Ordered list | Execution order for next_skills |
| `entry_criteria` | Optional | List | Conditions to enter this skill |
| `decision_rules` | Optional | List | How to choose between options |
| `failure_recovery` | Optional | List | What to do on failure |
| `exit_criteria` | Optional | List | When the skill is complete |
| `anti_patterns` | Optional | List | Common mistakes to avoid |
| `implementation` | Optional | List | References to scripts/tools |

### Body Content

The body is Markdown with sections for:
- **Overview** — What the skill does and when to use it
- **Workflow** — Step-by-step process
- **Rules** — Constraints and conventions
- **Examples** — Usage patterns
- **Related** — Links to related skills

## Naming Conventions

- **Directory name**: `lowercase-with-hyphens` (e.g., `code-review`, `db-migrations`)
- **Skill name**: Same as directory name, used for `read_skill` invocation
- **Display name**: Human-readable, CamelCase (e.g., "Code Review", "DB Migrations")

## Trigger Design

Triggers should match natural user language, not technical jargon.

**Good triggers:**
- "review the Python code"
- "create a docker container"
- "migrate the database"

**Bad triggers:**
- "execute python review task"
- "invoke docker skill"
- "run migration workflow"

Include at least 3-5 triggers per skill to cover common phrasings.

## Tool Selection

List only tools the skill actually needs. Be specific:

```yaml
preferred_tools:
  - read_file
  - grep
  - bash
  - edit_file
```

If the skill wraps an external CLI tool, document it in the body.

## Implementation Pattern

For skills that wrap external tools:

```yaml
implementation:
  - The executable code lives in `skills/<name>/scripts/<tool>.py`.
```

In the body, document:
1. How to run the script
2. What arguments it accepts
3. What output format to expect
4. How to parse the output

## Directory Structure

```
skills/<skill-name>/
├── SKILL.md              # Skill definition (always present)
├── scripts/              # External tool scripts (optional)
│   └── <tool>.py
└── references/           # Reference docs (optional)
    └── api-reference.md
```

## Example: Creating a Skill

Here's a complete example for a "code-review" skill:

```yaml
---
name: code-review
description: Review code changes for correctness, risks, and missing tests.
aliases:
  - code review
  - review code
  - PR review
triggers:
  - review the code
  - review changes
  - check this PR
  - review the diff
  - code review
preferred_tools:
  - read_file
  - grep
  - git
  - file_diff
example_queries:
  - review the latest changes
  - check this PR for bugs
  - review the diff
when_not_to_use:
  - the user wants a style/formatting review only
  - the task is about writing new code rather than reviewing existing code
next_skills:
  - cpp-code-review
  - python-code-review
  - javascript-code-review
entry_criteria:
  - the user has code to review
  - there is a diff or changed files to inspect
exit_criteria:
  - all changed files have been reviewed
  - bugs, risks, and suggestions are documented
anti_patterns:
  - reviewing only style without checking logic
  - suggesting changes without explaining the why
---

# Code Review

Inspect changed files for bugs, regressions, and quality issues.

## Workflow

1. Identify changed files using `git diff` or `file_diff`
2. Read each changed file to understand context
3. Check for:
   - Logic errors and edge cases
   - Missing error handling
   - Security vulnerabilities
   - Performance regressions
   - Test coverage gaps
4. Report findings prioritized by severity
```

## Common Patterns

### Router Skills

For skills that delegate to specialists:

```yaml
---
name: my-router
description: Route tasks to the appropriate specialist skills.
triggers:
  - my task
  - route my task
next_skills:
  - specialist-a
  - specialist-b
  - specialist-c
---

# Router

Choose the right specialist for the task:

- **specialist-a**: Use when X
- **specialist-b**: Use when Y
- **specialist-c**: Use when Z
```

### Tool-Wrapper Skills

For skills that wrap CLI tools:

```yaml
---
name: docker-compose
description: Manage Docker Compose services and containers.
triggers:
  - docker compose
  - docker-compose
  - manage containers
  - docker services
preferred_tools:
  - bash
implementation:
  - The executable code lives in `skills/docker-compose/scripts/docker-compose.sh`.
---

# Docker Compose

Manage Docker Compose services.

## Usage

Run from the project root:

```bash
skills/docker-compose/scripts/docker-compose.sh <command> [args]
```

## Commands

- `up` - Start services
- `down` - Stop services
- `logs` - View logs
- `ps` - List services
```

## Validation Checklist

Before finalizing a skill:

- [ ] Name is lowercase-hyphenated and under 64 chars
- [ ] Description is one sentence, present tense
- [ ] At least 3 triggers that match natural language
- [ ] preferred_tools listed if the skill needs specific tools
- [ ] example_queries included (2-5 real examples)
- [ ] when_not_to_use prevents common misuse
- [ ] Body has clear workflow and rules
- [ ] No overlap with existing skill triggers
- [ ] File is saved as `SKILL.md` in the skill directory
