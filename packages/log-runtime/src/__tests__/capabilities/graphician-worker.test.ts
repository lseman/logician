import { afterEach, expect, test } from "bun:test";
import { GraphicianWorker } from "../../capabilities/tools/graphician-worker.ts";

const clients: Array<{ close(): void }> = [];
afterEach(() => {
	for (const client of clients.splice(0)) client.close();
});

// Speaks the graphician worker JSONL protocol; `broken` operations return
// a protocol error, and `slow` operations hang past the client timeout.
const script = `
const readline = require('node:readline');
readline.createInterface({input:process.stdin}).on('line', line => {
 const r = JSON.parse(line);
 if (r.method === 'query' && r.operation === 'broken') {
  console.log(JSON.stringify({id: r.id, ok: false, error: 'worker fixture error'}));
  return;
 }
 if (r.method === 'query' && r.operation === 'slow') return;
 let data;
 switch (r.method) {
  case 'ping': data = { pong: true }; break;
  case 'query': data = { result: { operation: r.operation, nodes: 3, edges: 4 }, build: { state: 'running', root: '/tmp/x', last_exit: null, last_output: null } }; break;
  case 'refresh': data = { started: true, build: { state: 'running', root: r.root, last_exit: null, last_output: null } }; break;
  case 'build_status': data = { build: { state: 'idle', root: null, last_exit: 0, last_output: null } }; break;
  default: data = {};
 }
 console.log(JSON.stringify({ id: r.id, ok: true, ...data }));
});`;

function worker(timeoutMs = 2000) {
	const client = new GraphicianWorker({
		binary: "unused",
		python: process.execPath,
		args: ["-e", script],
		timeoutMs,
	});
	clients.push(client);
	return client;
}

test("decodes query results and snake_case build state", async () => {
	const client = worker();
	const { result, build } = await client.query("/tmp/db", "status", {});
	expect(result.nodes).toBe(3);
	expect(result.edges).toBe(4);
	expect(build.state).toBe("running");
	expect(build.root).toBe("/tmp/x");
	expect(build.lastExit).toBe(null);
	expect(build.lastOutput).toBe(null);
});

test("decodes refresh and build_status responses", async () => {
	const client = worker();
	const refresh = await client.refresh("/tmp/db", "/tmp/root");
	expect(refresh.started).toBe(true);
	expect(refresh.build.state).toBe("running");
	expect(refresh.build.root).toBe("/tmp/root");
	const status = await client.buildStatus();
	expect(status.state).toBe("idle");
	expect(status.lastExit).toBe(0);
});

test("protocol errors reject with the worker message", async () => {
	const client = worker();
	await expect(client.query("/tmp/db", "broken", {})).rejects.toThrow(
		"worker fixture error",
	);
	// The worker stays alive after a protocol error.
	const { result } = await client.query("/tmp/db", "status", {});
	expect(result.nodes).toBe(3);
});

test("slow queries time out without killing the worker", async () => {
	const client = worker(300);
	await expect(client.query("/tmp/db", "slow", {})).rejects.toThrow(
		/timed out/,
	);
	const { result } = await client.query("/tmp/db", "status", {});
	expect(result.nodes).toBe(3);
});
