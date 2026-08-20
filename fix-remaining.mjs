import fs from 'fs';
import path from 'path';

const srcRoot = './packages/agent-core/src';

// Create a mapping of "alias" names to actual file paths
// These are files that the new code references at the top level but actually live in subdirs
const aliasMap = {
  // Top-level aliases that map to subdirectories
  'controller.ts': 'application/eoh/controller.ts',
  'event-mapping.ts': 'runtime/event-mapping.ts',
  'events.ts': 'runtime/events.ts',
  'plugin-result-formatter.ts': 'runtime/plugin-result-formatter.ts',
  'registry.ts': 'infrastructure/tools/registry.ts',
  'backend.ts': 'core/backend.ts',
  'messages.ts': 'core/messages.ts',
  'agent-settings.ts': 'core/agent-settings.ts',
  'execution-policy.ts': 'core/execution-policy.ts',
  'intervention-controller.ts': 'core/intervention-controller.ts',
  'output-guard.ts': 'infrastructure/guards/output-guard.ts',
  'callbacks.ts': 'engine/loop/callbacks.ts',
  'provider-response.ts': 'engine/loop/provider-response.ts',
  'provider-turn.ts': 'engine/loop/provider-turn.ts',
  'event-bus.ts': 'extension/event-bus.ts',
  'types.ts': 'types/index.ts',
  'hook-bus.ts': 'hooks/hook-bus.ts',
  'builtin/budget.ts': 'hooks/builtin/budget.ts',
  'builtin/builtin-hooks.ts': 'hooks/builtin/builtin-hooks.ts',
  'skills/index.ts': 'features/skills/index.ts',
};

function resolveAlias(subpath) {
  if (subpath.startsWith('../')) {
    // Handle @logician/agent-core/../path (go up from srcRoot)
    const actualPath = subpath.replace('@logician/agent-core/', '');
    // Resolve from the file's directory
    return actualPath;
  }
  return aliasMap[subpath] || subpath;
}

function findFile(fromFile, subpath) {
  const fromDir = path.dirname(fromFile);
  const resolvedPath = resolveAlias(subpath);
  
  // Try relative from fromDir
  let resolved = path.resolve(fromDir, resolvedPath);
  if (fs.existsSync(resolved) && fs.statSync(resolved).isFile()) return resolved;
  if (fs.existsSync(resolved + '.ts')) return resolved + '.ts';
  const indexFile = path.join(resolved, 'index.ts');
  if (fs.existsSync(indexFile)) return indexFile;
  
  // Try relative to srcRoot
  const srcResolved = path.join(srcRoot, resolvedPath);
  if (fs.existsSync(srcResolved) && fs.statSync(srcResolved).isFile()) return srcResolved;
  if (fs.existsSync(srcResolved + '.ts')) return srcResolved + '.ts';
  const srcIndexFile = path.join(srcResolved, 'index.ts');
  if (fs.existsSync(srcIndexFile)) return srcIndexFile;
  
  return null;
}

function walk(dir, fileFilter) {
  const result = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      result.push(...walk(fullPath, fileFilter));
    } else if (fileFilter(fullPath)) {
      result.push(fullPath);
    }
  }
  return result;
}

const untrackedDirs = ['application', 'engine', 'features', 'infrastructure', 'runtime'];
const allFiles = [];
for (const dir of untrackedDirs) {
  const dirPath = path.join(srcRoot, dir);
  if (fs.existsSync(dirPath)) {
    allFiles.push(...walk(dirPath, f => f.endsWith('.ts') && !f.endsWith('.test.ts')));
  }
}

let fixedCount = 0;
let unresolvedCount = 0;
let totalChecked = 0;

for (const file of allFiles) {
  const content = fs.readFileSync(file, 'utf8');
  if (!content.includes('@logician/agent-core/')) continue;
  totalChecked++;
  
  let fixed = content.replace(/from\s+"@logician\/agent-core\/([^"]+)"/g, (match, subpath) => {
    const resolved = findFile(file, subpath);
    if (resolved) {
      const relPath = path.relative(path.dirname(file), resolved).replace(/\\/g, '/');
      return `from "${relPath}"`;
    }
    unresolvedCount++;
    return match;
  });
  
  if (fixed !== content) {
    fs.writeFileSync(file, fixed, 'utf8');
    fixedCount++;
  }
}

console.log(`Checked ${totalChecked} files, fixed ${fixedCount}, unresolved ${unresolvedCount}`);
