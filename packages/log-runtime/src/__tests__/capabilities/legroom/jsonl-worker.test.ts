import { afterEach, expect, test } from "bun:test";
import { LegroomWorker } from "../../../capabilities/legroom/worker.ts";
import { MemoriamWorker } from "../../../capabilities/memoriam/worker.ts";
import { JsonlWorker } from "../../../capabilities/sdk/jsonl-worker.ts";

const clients: Array<{ close(): void }> = [];
afterEach(() => {
	for (const client of clients.splice(0)) client.close();
});
const script = `
const readline = require('node:readline');
let config;
readline.createInterface({input:process.stdin}).on('line', line => {
 const r=JSON.parse(line);
 if(r.method==='init'){config=r.config;return;}
 if(r.method==='hang')return;
 if(r.method==='exit'){process.exit(2);return;}
 const response={id:r.id,ok:r.method!=='fail',error:'fixture error',result:{...r,config}};
 if(r.method==='compress')response.stats={metadata:{messages:r.messages},tokensSaved:2};
 if(r.method==='store_retrieve')response.content=r.hash;
 console.log('diagnostic noise');
 setTimeout(()=>console.log(JSON.stringify(response)),r.delay||0);
});`;
function transport(timeoutMs = 2000) {
	const client = new JsonlWorker({
		name: "Fixture",
		python: process.execPath,
		args: ["-e", script],
		timeoutMs,
	});
	clients.push(client);
	return client;
}

test("correlates concurrent responses and ignores non-protocol output", async () => {
	const client = transport();
	const [a, b] = await Promise.all([
		client.request({ method: "echo", value: "a", delay: 20 }),
		client.request({ method: "echo", value: "b" }),
	]);
	expect((a.result as { value: string }).value).toBe("a");
	expect((b.result as { value: string }).value).toBe("b");
	await expect(client.request({ method: "fail" })).rejects.toThrow(
		"fixture error",
	);
	expect((await client.request({ method: "echo" })).ok).toBe(true);
});
test("timeouts and serialization failures do not poison later requests", async () => {
	const client = transport(200);
	await expect(client.request({ method: "hang" })).rejects.toThrow("timed out");
	const circular: Record<string, unknown> = {};
	circular.self = circular;
	await expect(client.request(circular)).rejects.toThrow();
	expect((await client.request({ method: "echo" })).ok).toBe(true);
});
test("closing rejects pending requests and old exits cannot fail a restarted worker", async () => {
	const client = transport();
	const pending = client.request({ method: "hang" });
	const rejected = pending.catch((error: Error) => error);
	client.close();
	const restarted = client.request({ method: "echo", delay: 40 });
	expect(await rejected).toBeInstanceOf(Error);
	expect(((await rejected) as Error).message).toContain("closed");
	expect((await restarted).ok).toBe(true);
});
test("unexpected exit rejects requests and permits restart", async () => {
	const client = transport();
	await expect(client.request({ method: "exit" })).rejects.toThrow("exited");
	expect((await client.request({ method: "echo" })).ok).toBe(true);
});
test("Memoriam initializes every generation before sending domain payloads", async () => {
	const client = new MemoriamWorker({
		python: process.execPath,
		args: ["-e", script],
		config: { db_path: "fixture" },
	});
	clients.push(client);
	for (let i = 0; i < 2; i++) {
		const result = (await client.getSession("session")) as unknown as Record<
			string,
			unknown
		>;
		expect(result.session_id).toBe("session");
		expect(result.config).toEqual({ db_path: "fixture" });
		client.close();
	}
});
test("Legroom retains compression decoding and store request fields", async () => {
	const client = new LegroomWorker({
		python: process.execPath,
		args: ["-e", script],
		failOpen: false,
	});
	clients.push(client);
	const messages = [{ role: "user", content: "hello" }];
	expect(await client.compress(messages, "test")).toEqual(messages);
	expect(await client.storeRetrieve("store", "hash")).toBe("hash");
});

test("Legroom translates SDK fields and decodes responses by operation", async () => {
	const fixture = `
require('node:readline').createInterface({input:process.stdin}).on('line',line=>{
 const r=JSON.parse(line);let data;
 switch(r.method){
 case 'compress_with_store':
  if(r.store_id!=='store')throw Error('missing store_id');
  data={stats:{tokens_before:100,tokens_after:40,tokens_saved:60,transforms_applied:['phase_one'],metadata:{messages:r.messages,ccr_hashes:['hash']}}};break;
 case 'store_stats':data={stats:{entries:1,max_entries:20,total_bytes_before:100,total_bytes_after:40,savings:60}};break;
 case 'worker_stats':data={stats:{total_requests:2,strategy_counts:{phase_one:2}}};break;
 case 'cache_get':data={hit:true,messages:[{role:'user',content:'cached'}],stats:{tokens_before:20,tokens_after:10,tokens_saved:10}};break;
 case 'calibration_record':
  if(!Array.isArray(r.phase_reports))throw Error('missing phase_reports');
  data={calibration:{disabled_phases:[],snapshots:[{phase:'x',success_rate:1}]}};break;
 case 'worker_history':data={history:[{request_id:'r',tokens_saved:60}],total:1};break;
 }
 console.log(JSON.stringify({id:r.id,ok:true,...data}));
});`;
	const worker = new LegroomWorker({
		python: process.execPath,
		args: ["-e", fixture],
	});
	clients.push(worker);
	const result = await worker.compressWithStore(
		"store",
		[{ role: "user", content: "hello" }],
		"test",
	);
	expect(result.tokensSaved).toBe(60);
	expect(result.metadata?.ccrHashes).toEqual(["hash"]);
	expect((await worker.storeStats("store")).maxEntries).toBe(20);
	expect((await worker.workerStats()).strategyCounts).toEqual({ phase_one: 2 });
	const cached = await worker.cacheGet("key");
	expect(cached?.hit).toBe(true);
	expect(cached?.result?.messages[0].content).toBe("cached");
	expect((await worker.calibrationRecord([], 1)).snapshots[0].successRate).toBe(
		1,
	);
	expect((await worker.workerHistory()).history[0].requestId).toBe("r");
});
