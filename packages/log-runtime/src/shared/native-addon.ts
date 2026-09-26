// ── Native addon loader ──────────────────────────────────────────────────────
// Lazily loads @logician/log-natives so a missing/unbuilt build only breaks
// callers who actually need it, and shares one load across every consumer
// (ast-grep.ts, edit-file.ts, read-resource.ts).

export type NativeModule = typeof import("@logician/log-natives");

let nativePromise: Promise<NativeModule> | undefined;

export function loadNative(): Promise<NativeModule> {
	if (!nativePromise) {
		nativePromise = import("@logician/log-natives").catch(error => {
			nativePromise = undefined;
			throw new Error(
				`@logician/log-natives addon not available (run \`bun run build\` in packages/log-natives): ${
					error instanceof Error ? error.message : String(error)
				}`,
			);
		});
	}
	return nativePromise;
}

// ── Shared, session-persistent EditStore ────────────────────────────────────
// pi_edit's hashline mode validates a `[path#tag]` header by looking the tag
// up in its EditStore, not by recomputing a hash from disk — so the same
// store instance that records a snapshot when a file is read must still be
// around when that file is later edited. One process-lifetime instance,
// shared by read-resource.ts (records) and edit-file.ts (reads/validates).

type NativeEditStore = InstanceType<NativeModule["EditStore"]>;

let storePromise: Promise<NativeEditStore> | undefined;

export function getNativeEditStore(): Promise<NativeEditStore> {
	if (!storePromise) {
		storePromise = loadNative().then(native => new native.EditStore());
	}
	return storePromise;
}
