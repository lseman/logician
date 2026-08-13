// ── Pure helpers for LogicianTUI slash-command handling ───────────────────────

const LOOP_INTERVAL_UNIT_MS: Record<string, number> = {
	s: 1000,
	m: 60000,
	h: 3600000,
	d: 86400000,
};

export interface ParsedLoopInterval {
	value: string;
	unit: string;
	prompt: string;
	ms: number;
}

/** Parse `/loop <interval> <prompt>` args, e.g. "5m check the deploy". */
export function parseLoopInterval(args: string): ParsedLoopInterval | null {
	const match = args.match(/^(\d+)(s|m|h|d)\s+(.+)$/);
	if (!match) return null;
	const [, value, unit, prompt] = match;
	const ms = parseInt(value, 10) * (LOOP_INTERVAL_UNIT_MS[unit] ?? 60000);
	return { value, unit, prompt, ms };
}

export const SANDBOX_PROFILES: Record<string, string> = {
	none: "No isolation — chroot to tmpdir only",
	code: "Read-only host fs, writable /tmp, no network, no devices",
	file: "CODE + read-only bind-mount of specified directories",
	dev: "CODE + limited /dev (null, zero, random, tty)",
	full: "CODE + user namespace + mount namespace + no new privs",
};

/** Describe a single sandbox profile, or list all profile names if unknown/omitted. */
export function describeSandboxProfile(name: string): string {
	if (!name) {
		return (
			"Available profiles:\n" +
			Object.entries(SANDBOX_PROFILES)
				.map(([k, v]) => `  ${k}: ${v}`)
				.join("\n")
		);
	}
	const description = SANDBOX_PROFILES[name];
	if (description) return `${name}: ${description}`;
	return `Unknown profile: ${name}. Use one of: ${Object.keys(SANDBOX_PROFILES).join(", ")}`;
}
