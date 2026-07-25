const fs = require('fs');
const js = fs.readFileSync('logician-explorer.js','utf8');

// Check for balanced braces, parens, and backticks
let braceDepth = 0, parenDepth = 0, templateDepth = 0;
let inString = false, stringChar = '';
let inLineComment = false, inBlockComment = false;

for (let i = 0; i < js.length; i++) {
  const c = js[i];
  const next = js[i+1];
  
  if (inLineComment) {
    if (c === '\n') inLineComment = false;
    continue;
  }
  if (inBlockComment) {
    if (c === '*' && next === '/') { inBlockComment = false; i++; }
    continue;
  }
  if (inString) {
    if (c === '\\') { i++; continue; }
    if (c === stringChar) inString = false;
    continue;
  }
  
  if (c === '/' && next === '/') { inLineComment = true; i++; continue; }
  if (c === '/' && next === '*') { inBlockComment = true; i++; continue; }
  if (c === '`') { templateDepth++; continue; }
  if (c === '"' || c === "'") { inString = true; stringChar = c; continue; }
  if (c === '{') braceDepth++;
  if (c === '}') braceDepth--;
  if (c === '(') parenDepth++;
  if (c === ')') parenDepth--;
}

console.log('Brace depth:', braceDepth);
console.log('Paren depth:', parenDepth);
console.log('Template depth:', templateDepth);
if (braceDepth === 0 && parenDepth === 0 && templateDepth === 0) {
  console.log('All balanced - syntax should be OK');
} else {
  console.log('UNBALANCED - syntax error likely');
}
