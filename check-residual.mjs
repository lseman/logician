import fs from 'fs';
import path from 'path';

const srcRoot = './packages/agent-core/src';

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
for (const dir of untrackedDirs) {
  const dirPath = path.join(srcRoot, dir);
  if (fs.existsSync(dirPath)) {
    const files = walk(dirPath, f => f.endsWith('.ts') && !f.endsWith('.test.ts'));
    for (const file of files) {
      const content = fs.readFileSync(file, 'utf8');
      if (content.includes('@logician/agent-core/')) {
        console.log(`${file}:`);
        const matches = content.match(/from\s+"@logician\/agent-core\/([^"]+)"/g);
        if (matches) {
          for (const m of matches) {
            console.log(`  ${m}`);
          }
        }
      }
    }
  }
}
