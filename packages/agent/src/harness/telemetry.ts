// ── Telemetry context ─────────────────────────────────────────────────────
// Minimal span-based telemetry contract, ported from pi-telemetry's core
// interfaces (index.ts + noop.ts). The typed-schema span vocabulary
// (AI_TELEMETRY_SCHEMA / HARNESS_TELEMETRY_SCHEMA and friends) is not ported
// yet — this is just the TelemetryContext shape the harness threads through,
// defaulting to a no-op implementation until a real exporter is wired up.

export type AttributeValue =
	| string
	| number
	| boolean
	| readonly string[]
	| readonly number[]
	| readonly boolean[];

export interface SpanAttributes {
	[name: string]: AttributeValue | undefined;
}

export interface SpanOptions {
	name: string;
	attributes?: SpanAttributes;
}

export type SpanStatus =
	| { status: "ok" }
	| { status: "error"; error?: { name: string; message: string } };

export interface TelemetryContext {
	startSpan<T>(
		options: SpanOptions,
		callback: (span: TelemetrySpan) => T | Promise<T>,
	): Promise<T>;
}

export interface TelemetrySpan extends TelemetryContext {
	addEvent(name: string, attributes?: SpanAttributes): void;
	setAttributes(attributes: SpanAttributes): void;
	setStatus(status: SpanStatus): void;
}

function startNoopSpan<T>(
	_options: SpanOptions,
	callback: (span: TelemetrySpan) => T | Promise<T>,
): Promise<T> {
	try {
		return Promise.resolve(callback(noopTelemetrySpan));
	} catch (error) {
		return Promise.reject(error);
	}
}

const noopTelemetrySpan: TelemetrySpan = {
	startSpan: startNoopSpan,
	addEvent: () => {},
	setAttributes: () => {},
	setStatus: () => {},
};
Object.freeze(noopTelemetrySpan);

/** Shared telemetry context used when an application does not provide one. */
export const NOOP_TELEMETRY_CONTEXT: TelemetryContext = noopTelemetrySpan;
