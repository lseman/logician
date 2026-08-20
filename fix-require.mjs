import fs from 'fs';

const file = 'packages/agent-core/src/application/agent-bridge.ts';
let content = fs.readFileSync(file, 'utf8');

// Replace the problematic require inside Promise
const old = `return new Promise((resolve, reject) => {
			const { spawn } = require("node:child_process");
			const { getShellConfig } = await import("../../infrastructure/tools/shell.ts");`;

const newCode = `return (async () => {
			const { spawn } = require("node:child_process");
			const { getShellConfig } = await import("../../infrastructure/tools/shell.ts");`;

content = content.replace(old, newCode);
fs.writeFileSync(file, content, 'utf8');
console.log('Fixed');
