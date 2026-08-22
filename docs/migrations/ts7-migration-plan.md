# TypeScript 7 Migration Plan

## Executive Summary

- **Files affected**: 1 file (`packages/log-runtime/src/capabilities/lsp/post-edit-diagnostics.ts`)
- **TypeScript version**: `6.0.3` → `7.0.2`
- **Breaking change**: `import ts from "typescript"` no longer returns the compiler API — only a version string
- **Risk**: Medium — requires replacing 20+ compiler API calls
- **Estimated effort**: 4-8 hours (depending on chosen approach)

---

## What Changed in TypeScript 7

### Package Structure

| | TypeScript 6.x | TypeScript 7.x |
|---|---|---|
| `import ts from "typescript"` | Full compiler API | **Version string only** |
| `ts.transpileModule()` | ✅ | ❌ **gone** |
| `ts.createProgram()` | ✅ | ❌ **gone** |
| `ts.ScriptTarget` | ✅ | Available via `typescript/unstable/ast` |
| `ts.createSourceFile()` | ✅ | ❌ **gone** |

### New API Surface (unstable)

TypeScript 7 introduces a socket-based client/server architecture:

```ts
// Old (TS 6):
import ts from "typescript";
ts.transpileModule(source, { ... });
ts.createProgram(files, options);

// New (TS 7) — unstable, requires server process:
import { API } from "typescript/unstable/sync";
const api = new API(); // spawns tsc server
const snapshot = api.updateSnapshot({ ... });
// ... API calls over socket
api.close();
```

### Enums (Available ✅)

The following enums are available via `typescript/unstable/ast`:
- `ScriptTarget` (ES2015-ES2025, Latest/ESNext)
- `ScriptKind` (JS, JSX, TS, TSX)
- `ModuleKind` (CommonJS, ESNext, etc.)
- `JsxEmit` (Preserve, React, etc.)
- `SyntaxKind`, `DiagnosticCategory`, etc.

### What's Missing ❌

The following compiler APIs have **no replacement** in the unstable API:
- `transpileModule()` — no transpilation function
- `createSourceFile()` — no text-to-AST parser
- `createProgram()` — no program creation
- `sys.*` — no built-in file system abstraction
- `findConfigFile()` — no config file search
- `readConfigFile()` — no config reading
- `parseJsonConfigFileContent()` — no config parsing
- `flattenDiagnosticMessageText()` — no message flattening

---

## Affected Code Analysis

### File: `packages/log-runtime/src/capabilities/lsp/post-edit-diagnostics.ts` (248 lines)

**Only 1 file imports from "typescript"**, with 2 distinct code paths:

#### Path 1: `collectTypeScriptDiagnostics()` (lines 61-91)
**Purpose**: Quick parse-level diagnostics for a single file

| TS 6 API | Usage | Replacement Options |
|---|---|---|
| `ts.transpileModule()` | Transpile with diagnostics | `tsc --transpileOnly` CLI, or skip |
| `ts.createSourceFile()` | Parse source for position mapping | `tsc` AST, or custom scanner |
| `ts.ScriptTarget.Latest` | Target ESNext | `typescript/unstable/ast` |
| `ts.ScriptKind` | Extension → kind enum | `typescript/unstable/ast` |
| `ts.ModuleKind.ESNext` | Module system | `typescript/unstable/ast` |
| `ts.JsxEmit.Preserve` | JSX handling | `typescript/unstable/ast` |
| `diagnostic.start` | Error position | Same in new API |
| `diagnostic.messageText` | Error message | Same in new API |
| `diagnostic.code` | Error code | Same in new API |

#### Path 2: `collectProjectDiagnostics()` (lines 94-137)
**Purpose**: Full project-level type checking

| TS 6 API | Usage | Replacement Options |
|---|---|---|
| `ts.findConfigFile()` | Find tsconfig.json | CLI `tsc --showConfig`, or manual search |
| `ts.sys.fileExists` | File existence check | `fs.existsSync()` |
| `ts.sys.readFile` | File reading | `fs.readFileSync()` |
| `ts.readConfigFile()` | Parse tsconfig | `JSON.parse(fs.readFileSync(...))` |
| `ts.parseJsonConfigFileContent()` | Resolve config → file list | `tsc --listFiles` CLI |
| `ts.createProgram()` | Create compilation unit | `tsc` process, or socket API |
| `program.getSourceFile()` | Get AST node | Socket API only |
| `program.getSyntacticDiagnostics()` | Parse errors | Socket API only |
| `program.getSemanticDiagnostics()` | Type errors | Socket API only |
| `ts.flattenDiagnosticMessageText()` | Format messages | Simple string conversion |

---

## Migration Approaches

### Option A: CLI Subprocess (Recommended for Stability)

Spawn the `tsc` binary as a subprocess. Simple, reliable, no new dependencies.

**Pros:**
- No breaking changes to behavior
- No external dependencies
- Simple implementation (~50 lines new code)
- Works today with TS 7

**Cons:**
- Slower (process spawn ~50-100ms per call)
- Less control over diagnostics
- Harder to get position mappings

**Implementation:**
```ts
import { execFileSync } from "node:child_process";
import { join, dirname } from "node:path";

const TS_BIN = join(
  dirname(require.resolve("typescript")),
  "bin",
  "tsc",
);

function collectTypeScriptDiagnostics(filePath, source, extension) {
  // Write temp file, run tsc, parse output
  const tempDir = fs.mkdtempSync(...);
  fs.writeFileSync(tempFile, source);
  const result = execFileSync(TS_BIN, [
    "--noEmit",
    "--target", "ESNext",
    "--module", "ESNext",
    "--jsx", "preserve",
    "--allowJs",
    tempFile,
  ], { cwd: dirname(filePath), timeout: 5000 });
  
  // Parse tsc stdout for errors
  return parseTscOutput(result.stdout, filePath);
}
```

### Option B: Socket-Based API (Unstable)

Use the new `typescript/unstable/sync` API directly. Full control but requires server lifecycle management.

**Pros:**
- Full control over diagnostics
- No file I/O overhead
- Modern API surface
- Better for future TS features

**Cons:**
- API is **unstable** — may change without notice
- Requires spawning and managing a server process
- Complex error handling
- No `transpileModule` equivalent anyway
- More code to maintain (~200+ lines)

**Implementation outline:**
```ts
import { API, type Snapshot } from "typescript/unstable/sync";

let api: API | null = null;

function getAPI(): API {
  if (!api) {
    api = new API({
      spawnOptions: {
        command: "node",
        args: [join(__dirname, "node_modules/typescript/bin/tsc")],
      },
    });
  }
  return api;
}

function collectProjectDiagnostics(cwd, filePath) {
  const api = getAPI();
  const snapshot = api.updateSnapshot({
    files: [{ path: filePath, content: source }],
  });
  const project = snapshot.getProject(cwd + "/tsconfig.json");
  const diagnostics = project.program.getSyntacticDiagnostics();
  // ...
}
```

### Option C: Hybrid (Best of Both)

Use CLI for quick transpile diagnostics, socket API for project diagnostics.

**Pros:**
- Best performance for each use case
- CLI is fine for single-file checks
- Socket API for full project checks

**Cons:**
- Two different code paths to maintain
- More complex

---

## Recommended Approach: Option A (CLI Subprocess)

### Why CLI Subprocess?

1. **Simplicity**: ~50 lines vs ~200+ lines for socket API
2. **Stability**: `tsc` CLI is stable, won't change
3. **No new deps**: Uses existing TS installation
4. **Acceptable perf**: 50-100ms is fine for post-edit diagnostics (not called in hot path)
5. **Fallback**: If TS 7 removes `tsc`, we can add a runtime check

### Migration Steps

#### Step 1: Extract TypeScript types (Day 1)

```ts
// New file: packages/log-runtime/src/capabilities/lsp/ts-types.ts
// Re-exports from typescript/unstable/ast (safe, will work in TS 7)
export { ScriptKind, ScriptTarget, ModuleKind, JsxEmit } from "typescript/unstable/ast";
export type { Diagnostic, SourceFile } from "typescript/unstable/sync";
```

#### Step 2: Replace `collectTypeScriptDiagnostics()` (Day 1-2)

```ts
// NEW: CLI-based transpile diagnostics
function collectTypeScriptDiagnostics(
  filePath: string,
  source: string,
  extension: string,
): PostEditDiagnostic[] {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "tsc-"));
  const tmpFile = path.join(tmpDir, path.basename(filePath));
  fs.writeFileSync(tmpFile, source);
  
  try {
    const result = execFileSync(
      getTSBin(),
      ["--noEmit", "--target", "ESNext", "--module", "ESNext", "--jsx", "preserve", "--allowJs", tmpFile],
      { cwd: path.dirname(filePath), timeout: 5000, stdio: ["pipe", "pipe", "pipe"] }
    );
    
    return parseTscStdout(result.stdout?.toString() ?? "", filePath);
  } finally {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
}
```

#### Step 3: Replace `collectProjectDiagnostics()` (Day 2-3)

```ts
// NEW: CLI-based project diagnostics
function collectProjectDiagnostics(
  cwd: string,
  filePath: string,
): PostEditDiagnostic[] {
  const tsConfig = findTsConfig(dirname(filePath));
  if (!tsConfig) return [];
  
  const result = execFileSync(
    getTSBin(),
    ["--noEmit", "--project", tsConfig, "--showConfig"],
    { cwd, timeout: 10000 }
  );
  
  // Parse tsc output for diagnostics
  return parseTscStdout(result.stdout?.toString() ?? "", filePath);
}
```

#### Step 4: Update imports (Day 3)

```ts
// OLD:
import ts from "typescript";

// NEW:
import { ScriptKind, ScriptTarget, ModuleKind, JsxEmit } from "typescript/unstable/ast";

// Replace all ts.ScriptKind.X → ScriptKind.X, etc.
```

#### Step 5: Add runtime check (Day 3)

```ts
// packages/log-runtime/src/capabilities/lsp/ts-version-check.ts
import { existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { resolve } from "node:path";

let TS_BIN: string | null = null;

export function getTSBin(): string {
  if (TS_BIN) return TS_BIN;
  
  try {
    const tsDir = dirname(resolve(require.resolve("typescript/package.json")));
    const bin = join(tsDir, "bin", "tsc");
    if (!existsSync(bin)) throw new Error(`tsc binary not found at ${bin}`);
    TS_BIN = bin;
    return TS_BIN;
  } catch (e) {
    console.warn("[post-edit-diagnostics] TypeScript compiler not available, skipping diagnostics", e);
    return "";
  }
}
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| TS 7 removes `tsc` CLI | Low | High | Runtime check, fallback to no-op |
| Socket API changes before stable | High | Medium | We don't use it (Option A) |
| Performance regression | Medium | Low | 50-100ms is acceptable for post-edit |
| New diagnostics format from tsc | Low | Medium | Parse only standard error output |
| Temp file cleanup issues | Low | Low | try/finally blocks |

---

## Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| Phase 1: Type exports + CLI wrapper | 2 hours | `ts-types.ts`, `ts-version-check.ts`, CLI-based `collectTypeScriptDiagnostics` |
| Phase 2: Project diagnostics | 3 hours | CLI-based `collectProjectDiagnostics`, config file search |
| Phase 3: Cleanup + testing | 2 hours | Remove old imports, add tests, verify behavior |
| **Total** | **7 hours** | **All typechecks pass, behavior unchanged** |

---

## Dependencies

- None — uses existing TypeScript installation
- `tsc` binary must exist (standard for all TS projects)

---

## Testing

### Unit Tests
1. `collectTypeScriptDiagnostics()` with valid TS → no errors
2. `collectTypeScriptDiagnostics()` with syntax error → error reported
3. `collectProjectDiagnostics()` with tsconfig → full project errors
4. `collectProjectDiagnostics()` without tsconfig → empty array

### Integration Tests
1. Edit a TS file with syntax error → diagnostic shown
2. Edit a TS file with type error → diagnostic shown
3. Edit a JS file → still works with `--allowJs`

### Performance Tests
1. Single file check < 200ms
2. Project check < 2s (acceptable for post-edit)

---

## Rollback Plan

If migration fails:
1. Keep TS 6 in `package.json`
2. Revert to original file
3. No breaking changes to dependents

---

## Future Considerations

### If TS 7 stabilizes the socket API
- Migrate from CLI to socket API (better perf, more control)
- CLI approach serves as a bridge

### If TS 8 removes the `tsc` CLI
- This would be a breaking change announced in release notes
- Migration window would be 6+ months
- Could use `tsx` or `esbuild` as fallback

### `noUncheckedIndexedAccess` (separate PR)
- Do after TS 7 migration
- Would catch bugs like `diagnostic.start ?? 0` (already handled)
