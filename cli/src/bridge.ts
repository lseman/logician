import {spawn, type ChildProcessWithoutNullStreams} from 'node:child_process';
import {existsSync} from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import {fileURLToPath} from 'node:url';

/**
 * Resolve the absolute path to python_bridge.py.
 *
 * Search order:
 *   1. Sibling of the running binary (e.g. bun compiled: ./logician → ./python_bridge.py)
 *   2. cwd() — common when launching from repo root
 *   3. Sibling of the compiled entry-point source file (dev: cli/src/ → cli/)
 */
function resolveBridgePath(): string {
  const candidates: string[] = [];

  // When compiled by bun, process.execPath is the binary itself.
  // The convention is: binary lives next to python_bridge.py.
  candidates.push(path.join(path.dirname(process.execPath), 'python_bridge.py'));

  // Common repo-root launch path.
  candidates.push(path.join(process.cwd(), 'python_bridge.py'));

  // import.meta.url is available in Node ESM and tsx.
  // During dev: .../cli/src/bridge.ts → .../cli/python_bridge.py
  try {
    const currentFile = fileURLToPath(import.meta.url);
    candidates.push(path.join(path.dirname(currentFile), '..', 'python_bridge.py'));
  } catch {}

  for (const c of candidates) {
    if (existsSync(c)) {
      return c;
    }
  }

  // Return the cwd candidate anyway and let Python produce a useful error.
  return candidates[candidates.length - 1]!;
}

export type BridgeEvent =
  | {event: 'token'; token: string}
  | {event: 'tool'; name: string; args: Record<string, unknown>}
  | {event: 'skill'; skill_ids: string[]; selected_tools: string[]}
  | {event: 'phase'; state: string; note?: string}
  | {event: string; [k: string]: unknown};

type Pending = {
  resolve: (value: any) => void;
  reject: (reason?: unknown) => void;
};

type RpcResponse = {
  id: number;
  ok: boolean;
  result?: unknown;
  error?: string;
};

export class PythonBridge {
  private proc: ChildProcessWithoutNullStreams | null = null;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private onEvent: (event: BridgeEvent) => void;

  constructor(onEvent: (event: BridgeEvent) => void) {
    this.onEvent = onEvent;
  }

  start(): void {
    if (this.proc) {
      return;
    }

    const scriptPath = resolveBridgePath();
    const py = process.env.PYTHON ?? 'python3';

    this.proc = spawn(py, [scriptPath], {
      cwd: process.cwd(),
      stdio: ['pipe', 'pipe', 'pipe']
    });

    const rl = readline.createInterface({input: this.proc.stdout});
    rl.on('line', line => {
      let obj: Record<string, unknown>;
      try {
        obj = JSON.parse(line) as Record<string, unknown>;
      } catch {
        this.onEvent({event: 'bridge_log', line});
        return;
      }

      if (typeof obj.event === 'string') {
        this.onEvent(obj as BridgeEvent);
        return;
      }

      const id = Number(obj.id);
      const waiter = this.pending.get(id);
      if (!waiter) {
        return;
      }
      this.pending.delete(id);

      const resp = obj as unknown as RpcResponse;
      if (resp.ok) {
        waiter.resolve(resp.result);
      } else {
        waiter.reject(new Error(resp.error || 'Bridge call failed'));
      }
    });

    this.proc.stderr.on('data', chunk => {
      this.onEvent({event: 'bridge_stderr', text: String(chunk)});
    });

    this.proc.on('exit', code => {
      this.onEvent({event: 'bridge_exit', code});
      for (const [, waiter] of this.pending) {
        waiter.reject(new Error('Bridge process exited'));
      }
      this.pending.clear();
      this.proc = null;
    });
  }

  async call<T>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    if (!this.proc) {
      throw new Error('Bridge process is not running');
    }

    const id = this.nextId++;
    const payload = {id, method, params};

    const promise = new Promise<T>((resolve, reject) => {
      this.pending.set(id, {resolve, reject});
    });

    this.proc.stdin.write(JSON.stringify(payload) + '\n');
    return promise;
  }

  stop(): void {
    if (!this.proc) {
      return;
    }
    this.proc.kill('SIGTERM');
    this.proc = null;
  }
}
