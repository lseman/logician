import fs from 'fs';
import path from 'path';

const srcRoot = './packages/agent-core/src';

function findFile(fromDir, importPath) {
  // First try relative from fromDir
  let resolved = path.resolve(fromDir, importPath);
  if (fs.existsSync(resolved + '.ts')) return resolved + '.ts';
  if (fs.existsSync(resolved)) return resolved;
  
  // Try with index.ts
  resolved = path.join(path.resolve(fromDir, importPath), 'index.ts');
  if (fs.existsSync(resolved)) return resolved;
  
  // Try relative to srcRoot (prepend ../)
  const relFromSrc = path.relative(srcRoot, fromDir);
  if (!relFromSrc.startsWith('.') && relFromSrc !== '') {
    // fromDir is inside srcRoot, go up to srcRoot first
    const upPath = path.join('..', importPath);
    resolved = path.resolve(fromDir, upPath);
    if (fs.existsSync(resolved + '.ts')) return resolved + '.ts';
    if (fs.existsSync(resolved)) return resolved;
    resolved = path.join(path.resolve(fromDir, upPath), 'index.ts');
    if (fs.existsSync(resolved)) return resolved;
  }
  
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

// Only process untracked directories
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
for (const file of allFiles) {
  const content = fs.readFileSync(file, 'utf8');
  if (!content.includes('@logician/agent-core/')) continue;
  
  const fromDir = path.dirname(file);
  let fixed = content.replace(/from\s+"@logician\/agent-core\/([^"]+)"/g, (match, subpath) => {
    const resolved = findFile(fromDir, subpath);
    if (resolved) {
      const relPath = path.relative(fromDir, resolved).replace(/\\/g, '/');
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

console.log(`Fixed ${fixedCount} files, ${unresolvedCount} unresolved`);
