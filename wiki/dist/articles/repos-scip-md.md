# scip Repository

> Compiled article maintained by the wiki builder.

- ID: `repos-scip-md`
- Source: [source note](../../source/repos/scip.md)
- Source path: `repos/scip.md`
- Updated: `2026-04-08T16:11:07Z`
- Words: `1330`
- Summary: > Derived from local repository `/data/dev/logician/wiki/raw/repos/scip/checkout` via raw snapshot `repos/scip/snapshot.md`.

## Concepts

- [[concepts/folder-repos|Repos]]
- [[concepts/keyword-files|Files]]
- [[concepts/keyword-readme|Readme]]
- [[concepts/keyword-repository|Repository]]
- [[concepts/keyword-snapshot|Snapshot]]

## Backlinks

- No explicit backlinks yet.

## Related Notes

- [[articles/repos-highs-md|highs Repository]]: shared section `repos`; shared keywords: files, readme, repository, snapshot

## Headings

- scip Repository
- Summary
- Repository Metadata
- Search Graph Artifacts
- Repository Structure
- Candidate files for AST inspection
- `.gitignore`
- `Makefile`
- `README.md`
- `examples/GMI/src/Makefile`
- `examples/LOP/src/Makefile`
- `src/amplmp/README`
- `src/dejavu/README.md`
- `tests/Makefile`
- Ingest Focus
- Related Existing Notes
- Suggested Follow-Up Pages
- Snapshot Excerpt
- checkout Repository Snapshot
- Summary
- Snapshot Metadata
- Top-Level Areas
- Dominant File Types
- Key Files
- Representative Excerpts
- .gitignore
- git ls-files --others --exclude-from=.git/info/exclude
- Lines that start with '#' are comments.
- For a project mostly in C, the following would be a good set of
- exclude patterns (uncomment them if you want to use them):
- compiled object files
- created by make compilation
- common cmake build folder names
- check folder, files created by or for `make test`
- vim swap files
- directories created by file managers (KDE Dolphin, macOS Finder)
- lint and pclint configurations files
- doc files
- editor cache files
- Makefile
- README.md
- SCIP: Solving Constraint Integer Programs
- examples/GMI/src/Makefile
- examples/LOP/src/Makefile
- src/amplmp/README
- src/dejavu/README.md
- Compilation, Library, Tests

## Source Material

# scip Repository

> Derived from local repository `/data/dev/logician/wiki/raw/repos/scip/checkout` via raw snapshot `repos/scip/snapshot.md`.

## Summary

Repository snapshot for `checkout` with 1985 text/code files across 10 top-level areas.

## Repository Metadata

- Repository path: `/data/dev/logician/wiki/raw/repos/scip/checkout`
- Raw snapshot: `repos/scip/snapshot.md`
- Raw manifest: `repos/scip/manifest.json`
- Files scanned: `2785`
- Text/code files included: `1985`
- Test-like files detected: `162`

## Search Graph Artifacts

- Search graph JSON: `/data/dev/logician/wiki/raw/repos/scip/search_graph.json`
- Interactive graph viewer: `/data/dev/logician/wiki/raw/repos/scip/search_graph.html`
- Audit report: `/data/dev/logician/wiki/raw/repos/scip/search_graph_audit.md`
- Top-level areas: `src`, `examples`, `applications`, `tests`, `check`, `scripts`
## Repository Structure

### Candidate files for AST inspection

### `.gitignore`
- Language: `text`

### `Makefile`
- Language: `text`
- Rationale comments: 2
  - `the testgams target need to come after make/local/make.targets has been included (if any), because the latter may assign a value to CLIENTTMPDIR`
  - `do not attempt to include .d files if there will definitely be any (empty DFLAGS), because it slows down the build on Windows considerably`

### `README.md`
- Language: `markdown`

### `examples/GMI/src/Makefile`
- Language: `text`

### `examples/LOP/src/Makefile`
- Language: `text`

### `src/amplmp/README`
- Language: `text`

### `src/dejavu/README.md`
- Language: `markdown`

### `tests/Makefile`
- Language: `text`
- Rationale comments: 4
  - `TODO: use the $ORIGIN variable`
  - `NOTE: currently, compilation with SHARED=false is not supported.`
  - `TODO: in newer version, check if we still need this`


## Ingest Focus

- Capture the architecture, entrypoints, and core module boundaries.
- Note how the repository is operated, tested, and configured.
- Link this repo note to existing workflows, concepts, or decision pages.
- Prefer follow-up pages for durable topics rather than stuffing everything into one note.

## Related Existing Notes

- [[repos/checkout|checkout Repository]]
- [[repos/highs|highs Repository]]

## Suggested Follow-Up Pages

- `update` `repos/checkout.md`: This existing note appears related to the new source and may need fresh evidence, links, or contradiction checks.
- `update` `repos/highs.md`: This existing note appears related to the new source and may need fresh evidence, links, or contradiction checks.
- `create` `concepts/files-checkout.md`: This source appears to introduce or strengthen a durable topic that is not yet represented as its own wiki note.
- `create` `concepts/scip-architecture.md`: Capture the repository's major modules, boundaries, and how the pieces fit together.
- `create` `concepts/scip-entrypoints.md`: Document the main commands, services, scripts, or runtime entrypoints exposed by this codebase.
- `create` `concepts/scip-testing.md`: Summarize the test strategy, important fixtures, and what parts of the codebase are exercised.

## Snapshot Excerpt

```text
# checkout Repository Snapshot

> Generated from local repository `/data/dev/logician/wiki/raw/repos/scip/checkout` for wiki ingest.

## Summary

Repository snapshot for `checkout` with 1985 text/code files across 10 top-level areas.

## Snapshot Metadata

- Captured at: `2026-04-08T16:11:06Z`
- Repository path: `/data/dev/logician/wiki/raw/repos/scip/checkout`
- Files scanned: `2785`
- Text/code files included: `1985`
- Test-like files detected: `162`

## Top-Level Areas

- `src`: 1421 files
- `examples`: 150 files
- `applications`: 147 files
- `tests`: 145 files
- `check`: 47 files
- `scripts`: 46 files
- `doc`: 17 files
- `<root>`: 8 files
- `lint`: 3 files
- `pclint`: 1 files

## Dominant File Types

- `.h`: 760 files
- `.c`: 681 files
- `.hpp`: 238 files
- `.sh`: 76 files
- `.cpp`: 70 files
- `.txt`: 52 files
- `.md`: 30 files
- `<none>`: 28 files
- `.py`: 24 files
- `.html`: 9 files
- `.xml`: 9 files
- `.java`: 3 files

## Key Files

- `.gitignore`
- `Makefile`
- `README.md`
- `examples/GMI/src/Makefile`
- `examples/LOP/src/Makefile`
- `src/amplmp/README`
- `src/dejavu/README.md`
- `tests/Makefile`

## Representative Excerpts

### .gitignore

- Size: `2593` bytes

```text
# git ls-files --others --exclude-from=.git/info/exclude
# Lines that start with '#' are comments.
# For a project mostly in C, the following would be a good set of
# exclude patterns (uncomment them if you want to use them):

\#*
backup/
cplex.log

settings/

# compiled object files
*.[oa]
*.pyc
*.so
*~

# created by make compilation
bin/
lib/
obj/
TAGS

# common cmake build folder names
build/
build*
debug/
release/
cmake-build-debug/
cmake-build-release/

# check folder, files created by or for `make test`
check/results/
check/results*/
check/locks
check/MINLP
check/IP
check/LP
check/instancedata
check/solchecker/doc/doc/

# vim swap files
.*.swp

# directories created by file managers (KDE Dolphin, macOS Finder)
.directory
.DS_Store

# lint and pclint configurations files
lint.out
lint/gcc-include-path.lnt
lint/lint_cmac.h
lint/lint_cppmac.h
lint/size-options.lnt
pclint/co-gcc.h
pclint/co-gcc.lnt
pclint/co-clang.h
pclint/co-clang.lnt

# doc files
doc/html/
doc/html_devel/
doc/doxygen
doc/scip.tag
doc/inc/faq/faqdata.php
doc/inc/faq/faq.inc
doc/inc/parameters.set
doc/**/*.tmp
doc/*.log
doc/inc/simpleinstance/output.log

# editor cache files
.cproject
.project
.vscode/
.idea/

[truncated]
```

### Makefile

- Size: `74140` bytes

```text
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *#
#*                                                                           *#
#*                  This file is part of the program and library             *#
#*         SCIP --- Solving Constraint Integer Programs                      *#
#*                                                                           *#
#*  Copyright (c) 2002-2026 Zuse Institute Berlin (ZIB)                      *#
#*                                                                           *#
#*  Licensed under the Apache License, Version 2.0 (the "License");          *#
#*  you may not use this file except in compliance with the License.         *#
#*  You may obtain a copy of the License at                                  *#
#*                                                                           *#
#*      http://www.apache.org/licenses/LICENSE-2.0                           *#
#*                                                                           *#
#*  Unless required by applicable law or agreed to in writing, software      *#

[truncated]
```

### README.md

- Size: `1353` bytes

```markdown
# SCIP: Solving Constraint Integer Programs

Welcome to what is currently one of the **fastest academically developed solvers**
for mixed integer programming (**MIP**) and mixed integer nonlinear programming
(**MINLP**). In addition, SCIP provides a highly flexible framework for **constraint
integer programming**, **branch-cut-and-price**, and can optionally be configured to
solve mixed-integer linear programs in a **numerically exact solving mode**.
SCIP allows for total control of the solution process and the access of detailed
information down to the guts of the solver.

The original instance of this repository is hosted at
[git.zib.de](https://git.zib.de) and a read-only
mirror is available at
[github.com/scipopt/scip](https://github.com/scipopt/scip).

Further information and resources are available through the official website at
[scipopt.org](https://scipopt.org):

- [online documentation](https://scipopt.org/doc) of the code with information how to get started,
- downloads of official releases and binaries for various platforms,
- release reports and scientific articles describing the algorithms behind SCIP,

[truncated]
```

### examples/GMI/src/Makefile

- Size: `1896` bytes

```text
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*                                                                           *
#*                  This file is part of the program and library             *
#*         SCIP --- Solving Constraint Integer Programs                      *
#*                                                                           *
#*  Copyright (c) 2002-2026 Zuse Institute Berlin (ZIB)                      *
#*                                                                           *
#*  Licensed under the Apache License, Version 2.0 (the "License");          *
#*  you may not use this file except in compliance with the License.         *
#*  You may obtain a copy of the License at                                  *
#*                                                                           *
#*      http://www.apache.org/licenses/LICENSE-2.0                           *
#*                                                                           *
#*  Unless required by applicable law or agreed to in writing, software      *
#*  distributed under the License is distributed on an "AS IS" BASIS,        *

[truncated]
```

### examples/LOP/src/Makefile

- Size: `1896` bytes

```text
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*                                                                           *
#*                  This file is part of the program and library             *
#*         SCIP --- Solving Constraint Integer Programs                      *
#*                                                                           *
#*  Copyright (c) 2002-2026 Zuse Institute Berlin (ZIB)                      *
#*                                                                           *
#*  Licensed under the Apache License, Version 2.0 (the "License");          *
#*  you may not use this file except in compliance with the License.         *
#*  You may obtain a copy of the License at                                  *
#*                                                                           *
#*      http://www.apache.org/licenses/LICENSE-2.0                           *
#*                                                                           *
#*  Unless required by applicable law or agreed to in writing, software      *
#*  distributed under the License is distributed on an "AS IS" BASIS,        *

[truncated]
```

### src/amplmp/README

- Size: `335` bytes

```text
This directory contains some of the source of AMPL/MP taken from https://github.com/ampl/mp.
AMPL/MP is used by SCIP to read .nl files.
See LICENSE.rst for the license that applies to the files below this directory.

If updating source here, also update the version number (githash or tag)
src/scip/reader_nl.c::SCIPincludeReaderNl().
```

### src/dejavu/README.md

- Size: `2040` bytes

```markdown
# Compilation, Library, Tests
dejavu is a solver and C++ library for the fast detection and manipulation of [combinatorial symmetry](https://automorphisms.org/quick_start/symmetry/). 
Below, you can find some information on

[truncated]
```
