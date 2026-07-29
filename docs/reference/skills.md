---
title: Skills API
description: Programmatic API for loading and managing skills.
---

# Skills API

Programmatic access to the skills system.

## Skill interface

```typescript
interface Skill {
  name: string                    // Stable id: coding/file_ops
  displayName: string             // Human name: "File Operations"
  description: string
  content: string                 // SKILL.md content
  filePath: string                // Path to SKILL.md
  baseDir: string                 // Directory containing SKILL.md
  slashName: string               // Slash-safe: coding-file_ops
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

## Loading skills

```typescript
import { loadSkills, findSkills } from '@logician/coding-agent/skills'

// Load all skills from default locations
const skills = await loadSkills()

// Find skills matching triggers
const matching = findSkills(skills, ['read', 'write', 'edit'])

// Load from specific path
const custom = await loadSkills({ paths: ['./my-skills'] })
```

## Diagnostics

```typescript
interface SkillDiagnostic {
  type: 'warning' | 'collision'
  code: SkillDiagnosticCode
  message: string
  path: string
  winnerPath?: string
  loserPath?: string
}

type SkillDiagnosticCode =
  | 'file_info_failed'
  | 'list_failed'
  | 'read_failed'
  | 'parse_failed'
  | 'invalid_metadata'
  | 'collision'
```

## SKILL.md frontmatter

```yaml
name: my-skill
displayName: My Skill
description: What this skill does
triggers:
  - trigger1
  - trigger2
allowedTools:
  - read_file
  - grep
whenNotToUse:
  - when this skill is not appropriate
```
