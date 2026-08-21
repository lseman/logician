---
title: Skills API
description: Skill data structures, loading, diagnostics, and frontmatter.
---

# Skills API

Import the loader from `@logician/log-core/skills`.

## Skill shape

```typescript
interface Skill {
  name: string                  // stable path ID, e.g. coding/file_ops
  displayName: string           // frontmatter name or directory name
  description: string
  content: string
  filePath: string
  baseDir: string
  slashName: string             // final path segment, e.g. file_ops
  disableModelInvocation: boolean
  allowedTools?: string[]
  aliases?: string[]
  triggers?: string[]
  exampleQueries?: string[]
  whenNotToUse?: string[]
  nextSkills?: string[]
  preferredSequence?: string[]
  entryCriteria?: string[]
  decisionRules?: string[]
  failureRecovery?: string[]
  exitCriteria?: string[]
  antiPatterns?: string[]
  argumentHint?: string
  model?: string
  source: 'user' | 'project' | 'path'
}
```

## Load skills

`loadSkills` accepts one directory or a list of directories/source descriptors and returns both skills and diagnostics:

```typescript
import { loadSkills } from '@logician/log-core/skills'

const { skills, diagnostics } = await loadSkills([
  { dir: '/home/me/.agents/skills', source: 'user' },
  { dir: '/workspace/project/skills', source: 'project' },
])
```

Missing roots are skipped. Discovery is recursive, honors `.gitignore`, `.ignore`, and `.fdignore`, and deduplicates resolved file paths.

## Find and format skills

Use `findSkillByName(skills, query)` for exact IDs, display names, slash names, and aliases. `formatSkillCatalog` produces bounded system-prompt metadata; `formatSkillInvocation` renders one skill's full instructions and discovered resources.

## Diagnostics

```typescript
type SkillDiagnosticCode =
  | 'file_info_failed'
  | 'list_failed'
  | 'read_failed'
  | 'parse_failed'
  | 'invalid_metadata'
  | 'collision'
```

Diagnostics include `type`, `code`, `message`, and `path`; collisions also include winner and loser paths.

## Frontmatter keys

The required `description` and optional `name` use standard YAML frontmatter. List fields accept YAML arrays or comma-separated strings. Supported aliases include `allowed-tools`/`allowed_tools`, `argument-hint`/`argument_hint`, and `preferred_tools`.

See the [Skills guide](/guides/skills) for a complete example and naming behavior.
