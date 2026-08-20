// ── Deep equality ─────────────────────────────────────────────────────────
// Structural deep-equality check for plain JSON-shaped values. Used by the
// harness reducer to verify a provisioned entry's intent matches its
// eventually-committed content. Equivalent to typebox/guard's
// Guard.IsDeepEqual for the JSON-value subset the harness needs.

export function isDeepEqual(a: unknown, b: unknown): boolean {
	if (Object.is(a, b)) return true;
	if (
		typeof a !== "object" ||
		a === null ||
		typeof b !== "object" ||
		b === null
	)
		return false;

	if (Array.isArray(a) || Array.isArray(b)) {
		if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length)
			return false;
		for (let i = 0; i < a.length; i++) {
			if (!isDeepEqual(a[i], b[i])) return false;
		}
		return true;
	}

	const aKeys = Object.keys(a as Record<string, unknown>);
	const bKeys = Object.keys(b as Record<string, unknown>);
	if (aKeys.length !== bKeys.length) return false;
	for (const key of aKeys) {
		if (!Object.hasOwn(b as object, key)) return false;
		if (
			!isDeepEqual(
				(a as Record<string, unknown>)[key],
				(b as Record<string, unknown>)[key],
			)
		)
			return false;
	}
	return true;
}
