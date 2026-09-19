// ── Native tokenizer loader ──────────────────────────────────────────────────
// Lazily loads @logician/log-natives' countTokens() (a real BPE tokenizer,
// ported from oh-my-pi's crates/pi-natives/src/tokens.rs) for exact token
// counts. Resolves to null when the native addon isn't built, so callers can
// fall back to a heuristic instead of breaking the agent loop over a missing
// build artifact.

export type NativeModule = typeof import("@logician/log-natives");

let nativePromise: Promise<NativeModule | null> | undefined;

export function loadNativeTokenizer(): Promise<NativeModule | null> {
	if (!nativePromise) {
		nativePromise = import("@logician/log-natives").catch(() => null);
	}
	return nativePromise;
}
