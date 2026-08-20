import fs from 'fs';
import path from 'path';

const srcRoot = './packages/agent-core/src';

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
  // Additional aliases for remaining unresolved
  'adapters/claude-code/hook-layer.ts': 'extension/adapters/claude-code/hook-layer.ts',
  'harness-queue-hooks.ts': 'core/harness-queue-hooks.ts',
  'agent-loop-runner.ts': 'core/agent-loop-runner.ts',
  'continuation-tracker.ts': 'core/continuation-tracker.ts',
  'runtime-state.ts': 'core/runtime-state.ts',
  'session.ts': 'core/session.ts',
  'file-checkpoints.ts': 'core/file-checkpoints.ts',
  'configuration/config.ts': 'infrastructure/configuration/config.ts',
  'core/types/index.ts': 'engine/core/types/index.ts',
  '../infrastructure/tools/registry.ts': '../infrastructure/tools/registry.ts',
  '../infrastructure/tools/permissions.ts': '../infrastructure/tools/permissions.ts',
  '../infrastructure/guards/acceptance-contract.ts': '../infrastructure/guards/acceptance-contract.ts',
  '../infrastructure/guards/response-patterns.ts': '../infrastructure/guards/response-patterns.ts',
  '../../infrastructure/tools/json-utils.ts': '../../infrastructure/tools/json-utils.ts',
  '../../infrastructure/tools/plugins.ts': '../../infrastructure/tools/plugins.ts',
  '../core/messages.ts': '../core/messages.ts',
  '../core/types/index.ts': '../core/types/index.ts',
  // engine/harness imports
  'harness/model.ts': 'harness/model.ts',
};

function resolveAlias(subpath) {
  // Handle @logician/agent-core/../X patterns (already relative)
  if (subpath.startsWith('../')) {
    return subpath;  // Already relative, don't modify
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
