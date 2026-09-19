// ── Native tokenizer loader ──────────────────────────────────────────────────
// Lazily loads @logician/log-natives' countTokens() (a real BPE tokenizer,
// ported from oh-my-pi's crates/pi-natives/src/tokens.rs) for exact token
// counts. Throws a clear error if the addon isn't built, matching every
// other native-backed capability (grep, glob, diff, snapcompact) rather than
// silently degrading to a worse estimate — see estimateTokensHeuristic for
// the separate, deliberately-synchronous estimator used by status-bar/live
// inspection callers that can't await this.

export type NativeModule = typeof import("@logician/log-natives");

let nativePromise: Promise<NativeModule> | undefined;

export function loadNativeTokenizer(): Promise<NativeModule> {
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
