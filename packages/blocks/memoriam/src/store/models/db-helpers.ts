/** Pure JSON parsing/validation helpers for reading SQLite row columns —
 * no db or workspace dependency. */

import type { ObservationClaim, ObservationProvenance } from "../../types.ts";

export function safeParseJsonArray(val: unknown): string[] {
	if (Array.isArray(val)) return val.map(String);
	if (typeof val === "string") {
		try {
			return JSON.parse(val);
		} catch {
			return [];
		}
	}
	return [];
}

export function safeParseJson(val: string): unknown {
	try {
		return JSON.parse(val);
	} catch {
		return null;
	}
}

export function parseObservationClaims(value: unknown): ObservationClaim[] {
	const parsed = typeof value === "string" ? safeParseJson(value) : value;
	if (!Array.isArray(parsed)) return [];
	return parsed.flatMap(item => {
		if (!item || typeof item !== "object" || Array.isArray(item)) return [];
		const claim = item as Record<string, unknown>;
		if (
			typeof claim.text !== "string" ||
			typeof claim.confidence !== "number" ||
			!(["tentative", "verified", "invalidated"] as unknown[]).includes(
				claim.status,
			) ||
			!Array.isArray(claim.evidenceEventIds)
		)
			return [];
		return [
			{
				text: claim.text,
				confidence: Math.max(0, Math.min(1, claim.confidence)),
				status: claim.status as ObservationClaim["status"],
				evidenceEventIds: claim.evidenceEventIds
					.filter((id): id is string => typeof id === "string")
					.slice(0, 12),
				...(Array.isArray(claim.validityPredicates)
					? {
							validityPredicates: claim.validityPredicates
								.filter(
									(
										item,
									): item is NonNullable<
										ObservationClaim["validityPredicates"]
									>[number] =>
										!!item &&
										typeof item === "object" &&
										["file_hash", "git_revision", "config_value"].includes(
											(item as { type?: string }).type || "",
										),
								)
								.slice(0, 8),
						}
					: {}),
			},
		];
	});
}

export function parseObservationProvenance(
	value: unknown,
): ObservationProvenance | undefined {
	const parsed = typeof value === "string" ? safeParseJson(value) : value;
	if (!parsed || typeof parsed !== "object" || Array.isArray(parsed))
		return undefined;
	const item = parsed as Record<string, unknown>;
	if (
		(item.source !== "model" && item.source !== "deterministic") ||
		!(["trusted_local", "external", "untrusted"] as unknown[]).includes(
			item.trust,
		) ||
		typeof item.extractorVersion !== "string" ||
		typeof item.schemaVersion !== "number"
	)
		return undefined;
	return {
		source: item.source,
		trust: item.trust as ObservationProvenance["trust"],
		extractorVersion: item.extractorVersion,
		schemaVersion: item.schemaVersion,
		rejectionReason:
			typeof item.rejectionReason === "string"
				? item.rejectionReason
				: undefined,
	};
}
