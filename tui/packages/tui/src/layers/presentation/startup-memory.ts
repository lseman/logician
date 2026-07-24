type StartupMemoryItem = {
	id: string;
	content: string;
	relevance?: string;
};

type StartupMemoryState = {
	observation_count: number;
	active_observation_count: number;
	reflection_count: number;
	dropped_count: number;
	observations: StartupMemoryItem[];
	reflections: StartupMemoryItem[];
};

const MAX_PREVIEW_LENGTH = 180;

function asCount(value: unknown): number {
	const count = Number(value);
	return Number.isFinite(count) && count >= 0 ? count : 0;
}

function asItems(value: unknown): StartupMemoryItem[] {
	if (!Array.isArray(value)) return [];
	return value.flatMap((item): StartupMemoryItem[] => {
		if (!item || typeof item !== "object") return [];
		const record = item as Record<string, unknown>;
		const id = String(record.id || "").trim();
		const content = String(record.content || "").trim();
		if (!id || !content) return [];
		const relevance = String(record.relevance || "").trim();
		return [{ id, content, ...(relevance ? { relevance } : {}) }];
	});
}

function preview(content: string): string {
	const singleLine = content.replace(/\s+/g, " ").trim();
	return singleLine.length > MAX_PREVIEW_LENGTH
		? `${singleLine.slice(0, MAX_PREVIEW_LENGTH - 1)}…`
		: singleLine;
}

export function formatStartupMemory(state: Record<string, unknown>): string[] {
	const raw = state.observational_memory;
	if (!raw || typeof raw !== "object") return [];

	const memory = raw as Record<string, unknown>;
	const parsed: StartupMemoryState = {
		observation_count: asCount(memory.observation_count),
		active_observation_count: asCount(memory.active_observation_count),
		reflection_count: asCount(memory.reflection_count),
		dropped_count: asCount(memory.dropped_count),
		observations: asItems(memory.observations),
		reflections: asItems(memory.reflections),
	};
	if (
		parsed.observation_count === 0 &&
		parsed.reflection_count === 0 &&
		parsed.dropped_count === 0
	) {
		return [];
	}

	const lines = [
		"",
		"## Observational memory",
		`${parsed.active_observation_count} active observations · ${parsed.reflection_count} reflections · ${parsed.dropped_count} archived`,
	];
	if (parsed.observations.length) {
		lines.push(
			"",
			"### Recent observations",
			...parsed.observations.map(
				(item) =>
					`- [${item.relevance || "memory"}] ${preview(item.content)} (${item.id})`,
			),
		);
	}
	if (parsed.reflections.length) {
		lines.push(
			"",
			"### Recent reflections",
			...parsed.reflections.map(
				(item) => `- ${preview(item.content)} (${item.id})`,
			),
		);
	}
	return lines;
}
