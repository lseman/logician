---
description: Audit changelog entries before release
---
Audit changelog entries for all commits since the last release.

## Process

1. **Find the last release tag:**
   ```bash
   git tag --sort=-version:refname | head -1
   ```

2. **List all commits since that tag:**
   ```bash
   git log <tag>..HEAD --oneline
   ```

3. **Read each package's [Unreleased] section:**
   - tui/packages/agent-core/CHANGELOG.md
   - tui/packages/agent-capabilities/CHANGELOG.md
   - tui/packages/coding-agent/CHANGELOG.md
   - tui/packages/legacy-observational-memory/CHANGELOG.md
   - tui/packages/tui/CHANGELOG.md

4. **For each commit, check:**
   - Skip: changelog updates, doc-only changes, release housekeeping
   - Skip: changes to generated files
   - Determine which package(s) the commit affects (use `git show <hash> --stat`)
   - Verify a changelog entry exists in the affected package(s)
   - For external contributions (PRs), verify format: `Description ([#N](url) by [@user](url))`

5. **Add New Features section after changelog fixes:**
   - Insert a `### New Features` section at the start of `## [Unreleased]` in the appropriate package changelog.
   - Propose the top new features to the user for confirmation before writing them.

6. **Report:**
   - List commits with missing entries
   - Add any missing entries directly

## Changelog Format Reference

Sections (in order):
- `### Breaking Changes` - API changes requiring migration
- `### Added` - New features
- `### Changed` - Changes to existing functionality
- `### Fixed` - Bug fixes
- `### Removed` - Removed features

Attribution:
- Internal: `Fixed foo ([#123](https://github.com/<owner>/<repo>/issues/123))`
- External: `Added bar ([#456](https://github.com/<owner>/<repo>/pull/456) by [@user](https://github.com/user))`
