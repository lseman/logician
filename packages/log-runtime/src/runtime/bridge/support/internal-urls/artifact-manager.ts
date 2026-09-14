// -- Session-scoped artifact storage ------------------------------------------
// Artifacts are stored in a per-session directory under .logician/artifacts/.
// Each artifact gets a sequential numeric ID (0, 1, 2, ...) and is accessible
// via local://<id> URLs (the unified local:// handler resolves numeric hosts
// to artifact entries).
//
// Usage:
//   const reg = ArtifactRegistry.instance();
//   await reg.init(cwd, sessionId);
//   const id = await reg.save(content, toolType);
//   // → local://0 is now readable

import * as fs from "node:fs/promises";
import * as path from "node:path";

// ── ArtifactManager ──────────────────────────────────────────────────────────

/** Sanitize a tool name for safe use as the middle segment of the artifact filename. */
function sanitizeToolType(toolType: string): string {
	return (
		toolType
			.replace(/[^A-Za-z0-9_-]+/g, "_")
			.slice(0, 64)
			.replace(/^_+|_+$/g, "") || "tool"
	);
}

/** Persist an artifact atomically: write to temp, then rename. */
async function writeArtifact(
	filePath: string,
	content: string,
): Promise<number> {
	const tmpFile = `${filePath}.tmp.${process.pid}.${Date.now()}`;
	const contentBytes = Buffer.byteLength(content, "utf-8");
	await fs.writeFile(tmpFile, content, "utf-8");
	const stat = await fs.stat(tmpFile);
	if (stat.size !== contentBytes) {
		await fs.unlink(tmpFile).catch(() => {});
		throw new Error(
			`Artifact write mismatch: expected ${contentBytes} bytes, got ${stat.size}`,
		);
	}
	await fs.rename(tmpFile, filePath);
	return contentBytes;
}

export class ArtifactManager {
	#nextId = 0;
	readonly #dir: string;
	#dirCreated = false;
	#initPromise: Promise<void> | null = null;

	constructor(dir: string) {
		this.#dir = dir;
	}

	get dir(): string {
		return this.#dir;
	}

	async #ensureDir(): Promise<void> {
		if (!this.#dirCreated) {
			await fs.mkdir(this.#dir, { recursive: true });
			this.#dirCreated = true;
		}
		// Memoize the first-use scan so it runs exactly once.
		this.#initPromise ??= this.#scanExistingIds();
		await this.#initPromise;
	}

	async #scanExistingIds(): Promise<void> {
		const files = await this.listFiles();
		let maxId = -1;
		for (const file of files) {
			const match = file.match(/^(\d+)\..*\.log$/);
			if (match) {
				const id = parseInt(match[1], 10);
				if (id > maxId) maxId = id;
			}
		}
		this.#nextId = maxId + 1;
	}

	allocateId(): number {
		return this.#nextId++;
	}

	async allocatePath(toolType: string): Promise<{ id: string; path: string }> {
		await this.#ensureDir();
		const id = String(this.allocateId());
		const filename = `${id}.${sanitizeToolType(toolType)}.log`;
		return { id, path: path.join(this.#dir, filename) };
	}

	async save(content: string, toolType: string): Promise<string> {
		const { id, path: filePath } = await this.allocatePath(toolType);
		await writeArtifact(filePath, content);
		return id;
	}

	async exists(id: string): Promise<boolean> {
		const files = await this.listFiles();
		return files.some(f => f.startsWith(`${id}.`));
	}

	async listFiles(): Promise<string[]> {
		try {
			return await fs.readdir(this.#dir);
		} catch {
			return [];
		}
	}

	async getPath(id: string): Promise<string | null> {
		const files = await this.listFiles();
		const match = files.find(f => f.startsWith(`${id}.`));
		return match ? path.join(this.#dir, match) : null;
	}

	/** Read artifact content by ID. */
	async read(id: string): Promise<string | null> {
		const filePath = await this.getPath(id);
		if (!filePath) return null;
		try {
			return await fs.readFile(filePath, "utf-8");
		} catch {
			return null;
		}
	}

	/** List available artifact IDs sorted numerically. */
	async listIds(): Promise<string[]> {
		const files = await this.listFiles();
		const ids = new Set<string>();
		for (const file of files) {
			const match = file.match(/^(\d+)\./);
			if (match) ids.add(match[1]);
		}
		return [...ids].sort((a, b) => parseInt(a, 10) - parseInt(b, 10));
	}
}

// ── ArtifactRegistry (singleton) ─────────────────────────────────────────────

export interface ArtifactRegistryDeps {
	cwd: string;
	sessionId: string;
}

let _instance: ArtifactRegistry | null = null;

/** Process-global registry for the active session's artifacts. */
export class ArtifactRegistry {
	#manager: ArtifactManager | null = null;
	#cwd = "";
	#sessionId = "";

	private constructor() {}

	static instance(): ArtifactRegistry {
		if (!_instance) {
			_instance = new ArtifactRegistry();
		}
		return _instance;
	}

	static resetForTests(): void {
		_instance = null;
	}

	/** Initialize the registry with session context. Call once at bridge setup. */
	init(deps: ArtifactRegistryDeps): void {
		this.#cwd = deps.cwd;
		this.#sessionId = deps.sessionId;
		const artifactDir = this.#resolveArtifactDir();
		this.#manager = new ArtifactManager(artifactDir);
	}

	/** Get the artifact manager (initialized). */
	getManager(): ArtifactManager | null {
		return this.#manager;
	}

	/** Save content as an artifact and return the ID. */
	async save(content: string, toolType: string): Promise<string | null> {
		if (!this.#manager) return null;
		return this.#manager.save(content, toolType);
	}

	/** Read artifact content by ID. */
	async read(id: string): Promise<string | null> {
		if (!this.#manager) return null;
		return this.#manager.read(id);
	}

	/** List available artifact IDs. */
	async listIds(): Promise<string[]> {
		if (!this.#manager) return [];
		return this.#manager.listIds();
	}

	/** Check if an artifact exists. */
	async exists(id: string): Promise<boolean> {
		if (!this.#manager) return false;
		return this.#manager.exists(id);
	}

	/** Get the artifact directory path (for external access). */
	getArtifactDir(): string | null {
		return this.#manager?.dir ?? null;
	}

	/** Get the on-disk path for an artifact ID, without reading its content. */
	async getPath(id: string): Promise<string | null> {
		if (!this.#manager) return null;
		return this.#manager.getPath(id);
	}

	#resolveArtifactDir(): string {
		return path.join(this.#cwd, ".logician", "artifacts", this.#sessionId);
	}
}
