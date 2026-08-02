export interface InkOverlayListItem {
	label: string;
	metadata?: string;
	selected?: boolean;
	current?: boolean;
	status?: "active" | "success" | "warning" | "error" | "muted";
}

export interface InkListOverlayModel {
	kind: "list";
	title: string;
	subtitle?: string;
	hints?: string;
	headerLines?: string[];
	items: InkOverlayListItem[];
	emptyText: string;
	footer: string;
	selectedIndex: number;
	maxRows?: number;
}

export interface InkOverlayModelProvider {
	getInkOverlayModel(): InkListOverlayModel;
}

export function hasInkOverlayModel(value: unknown): value is InkOverlayModelProvider {
	return typeof (value as { getInkOverlayModel?: unknown })?.getInkOverlayModel === "function";
}
