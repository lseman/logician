import fs from 'fs';
import path from 'path';

const srcRoot = './packages/agent-core/src';

function findImport(fromFile, importPath) {
  const fromDir = path.dirname(fromFile);
  
  // Try: fromDir -> importPath (direct relative)
  let resolved = path.resolve(fromDir, importPath);
  if (fs.existsSync(resolved) && fs.statSync(resolved).isFile()) return resolved;
  if (fs.existsSync(resolved + '.ts')) return resolved + '.ts';
  
  // Try: fromDir -> importPath/index.ts
  const indexFile = path.join(resolved, 'index.ts');
  if (fs.existsSync(indexFile)) return indexFile;
  
  // Try: srcRoot -> importPath (the @logician/agent-core/X pattern)
  const srcResolved = path.join(srcRoot, importPath);
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

// Process all new untracked files - use srcRoot + dir
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
    const resolved = findImport(file, subpath);
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
