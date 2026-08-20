import fs from 'fs';
import path from 'path';

const srcRoot = './packages/agent-core/src';

function resolveRelative(fromFile, toImport) {
  const fromDir = path.dirname(fromFile);
  const resolved = path.resolve(fromDir, toImport);
  // Check if file exists
  if (fs.existsSync(resolved)) return path.relative(srcRoot, resolved).replace(/\\/g, '/').replace(/\.tsx?$/, '');
  if (fs.existsSync(resolved + '.ts')) return path.relative(srcRoot, resolved + '.ts').replace(/\\/g, '/');
  if (fs.existsSync(path.join(resolved, 'index.ts'))) return path.relative(srcRoot, path.join(resolved, 'index.ts')).replace(/\\/g, '/');
  return null;
}

function fixFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  let fixed = content;
  
  // Replace @logician/agent-core/... imports with relative paths
  fixed = fixed.replace(/from\s+"@logician\/agent-core\/([^"]+)"/g, (match, subpath) => {
    const resolved = resolveRelative(filePath, subpath);
    if (resolved) {
      return `from "./${resolved}"`;
    }
    // Try with trailing .ts
    const resolvedWithTs = resolveRelative(filePath, subpath + '.ts');
    if (resolvedWithTs) {
      return `from "./${resolvedWithTs}"`;
    }
    console.log(`COULD NOT RESOLVE: ${subpath} in ${filePath}`);
    return match;
  });
  
  if (fixed !== content) {
    fs.writeFileSync(filePath, fixed, 'utf8');
    console.log(`Fixed: ${filePath}`);
    return true;
  }
  return false;
}

const files = [];
function walk(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(fullPath);
    } else if (entry.name.endsWith('.ts') && !entry.name.endsWith('.test.ts')) {
      files.push(fullPath);
    }
  }
}

walk(srcRoot);

let count = 0;
for (const file of files) {
  if (fs.readFileSync(file, 'utf8').includes('@logician/agent-core/')) {
    if (fixFile(file)) count++;
  }
}

console.log(`Fixed ${count} files`);
