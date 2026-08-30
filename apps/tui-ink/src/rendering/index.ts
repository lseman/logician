// ── Ink TUI Rendering Module — Barrel Export ────────────────────────────────

export {
	// Core primitives
	RESET,
	BOLD,
	DIM,
	CURSOR_MARKER,
	visibleWidth,
	clampLineToWidth,
	compositeTuiLine,
	sanitizeTerminalText,
	tokenStr,
} from "./primitives.ts";

export {
	// Widget types
	type Alignment,
	type FillMode,
	type WidgetLayout,
	type WidgetStyle,
	type BuiltinWidgetId,
	type WidgetId,
	type WidgetData,
	type ContributedWidget,
	type FooterConfig,
	DEFAULT_WIDGET_LAYOUTS,
	createDefaultConfig,
} from "./widget-types.ts";

export {
	// Widget factory
	type WidgetFactoryStatus,
	produceWidgets,
} from "./widget-factory.ts";

export {
	// Layout engine / status bar
	FooterStatusBar,
} from "./layout.ts";
