import type {
	ThinkingDisplayStyle,
	ToolExecution,
} from "@logician/log-runtime/sessions";

export interface SanitizedStringCache {
	raw?: string;
	safe?: string;
}

export interface SanitizedToolCache {
	result: SanitizedStringCache;
	partialResult: SanitizedStringCache;
	streamOutput: SanitizedStringCache;
	argsSource?: ToolExecution["args"];
	argsSafe?: ToolExecution["args"];
}

/** Read-only rendering services and per-display caches shared by tool renderers. */
export interface RenderCtx {
	toolsExpanded: boolean;
	spinnerFrame: () => string;
	maxMessageLength: number;
	sanitizedToolCache: WeakMap<ToolExecution, SanitizedToolCache>;
	sanitizationMetrics: { cacheHits: number; scannedCharacters: number };
	currentWidth: number;
	thinkingMode: ThinkingDisplayStyle;
	isAgentExpanded?: (toolCallId: string, taskIndex: number) => boolean;
	isChildToolExpanded?: (parentKey: string, childToolCallId: string) => boolean;
	_taskHitRegions?: Array<{ start: number; end: number; key: string }>;
	renderNestedTool?: (
		ctx: RenderCtx,
		tool: ToolExecution,
		width: number,
		expanded: boolean,
	) => string[];
	detailSection?: (label: string, meta?: string) => string;
	previewBlock?: (
		ctx: RenderCtx,
		text: string,
		width: number,
		maxChars?: number,
	) => string[];
	computeBatchTally?: (
		ctx: RenderCtx,
		tool: ToolExecution,
	) => {
		total: number;
		completed: number;
		failed: number;
		running: number;
		liveStatus: Map<number, "running" | "completed" | "failed">;
		taskElapsedMs: Map<number, number>;
	};
}
