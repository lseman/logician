import type {
	TuiSessionService,
	TuiSessionTreeNode,
} from "@logician/log-runtime/sessions";
import { BOLD, type Component, RESET } from "../terminal/core.ts";
import { theme } from "../terminal/theme.ts";
import {
	clampPopupLines,
	POPUP_FRAME_OVERHEAD,
	renderSeparator,
	renderStatusLine,
} from "./popup-utils.ts";
import { SelectorController } from "./selector-controller.ts";

export type SessionTreeSummaryMode = "none" | "summarize" | "custom";

export type SessionTreeAction =
	| { type: "close" }
	| {
			type: "navigate";
			entryId: string;
			summaryMode: SessionTreeSummaryMode;
			customPrompt?: string;
	  };

type Mode = "tree" | "summary" | "custom";

/** A flat list entry for the session tree display. */
interface FlatNode {
	id: string;
	parentId?: string;
	label: string;
	isCurrent: boolean;
	/** Whether this node has children (turns that branched off from here). */
	hasChildren: boolean;
	/** Number of children at this node. */
	childCount: number;
}

export class SessionTreeOverlay implements Component {
	private store: TuiSessionService | null = null;
	private sessionId: string | null = null;
	private nodes: FlatNode[] = [];
	private selection = new SelectorController();
	private summarySelection = new SelectorController();
	private selectedEntryId: string | null = null;
	private customPrompt = "";
	private mode: Mode = "tree";
	private visible = false;
	private cachedLines: string[] | null = null;
	private cachedWidth = -1;
	private actionCallback: ((action: SessionTreeAction) => void) | null = null;

	setStore(store: TuiSessionService): void {
		this.store = store;
	}

	setActionCallback(callback: (action: SessionTreeAction) => void): void {
		this.actionCallback = callback;
	}

	show(sessionId?: string): void {
		this.sessionId = sessionId ?? this.store?.getCurrentSessionId() ?? null;
		const storeNodes = this.sessionId
			? (this.store?.getTurnTree(this.sessionId) ?? [])
			: [];
		this.nodes = this.buildFlatTree(storeNodes);
		const current = this.nodes.findIndex(node => node.isCurrent);
		this.selection.set(
			current >= 0 ? current : Math.max(0, this.nodes.length - 1),
			this.nodes.length,
		);
		this.summarySelection.set(0, 3);
		this.selectedEntryId = null;
		this.customPrompt = "";
		this.mode = "tree";
		this.visible = true;
		this.invalidate();
	}

	/** Build a flat list: all turns at the same level, only show branch indicators. */
	private buildFlatTree(storeNodes: TuiSessionTreeNode[]): FlatNode[] {
		// Build parent -> children map to detect branches
		const byParent = new Map<string | undefined, TuiSessionTreeNode[]>();
		for (const node of storeNodes) {
			const siblings = byParent.get(node.parentId) ?? [];
			siblings.push(node);
			byParent.set(node.parentId, siblings);
		}

		// Build result: flat list with branch info
		const result: FlatNode[] = [];
		for (const node of storeNodes) {
			const children = byParent.get(node.id) ?? [];
			result.push({
				id: node.id,
				parentId: node.parentId,
				label: node.label,
				isCurrent: node.isCurrent,
				hasChildren: children.length > 0,
				childCount: children.length,
			});
		}

		return result;
	}

	hide(): void {
		this.visible = false;
		this.invalidate();
	}

	isVisibleOverlay(): boolean {
		return this.visible;
	}

	handleInput(data: string): void {
		if (data === "\x03") return this.close();
		if (data === "\x1b") {
			if (this.mode === "tree") return this.close();
			this.mode = this.mode === "custom" ? "summary" : "tree";
			this.invalidate();
			return;
		}
		if (this.mode === "custom") return this.handleCustomInput(data);
		const controller =
			this.mode === "tree" ? this.selection : this.summarySelection;
		const count = this.mode === "tree" ? this.nodes.length : 3;
		if (data === "\x1b[A" || data === "\x1bOA" || data === "k") {
			controller.move(-1, count);
			this.invalidate();
			return;
		}
		if (data === "\x1b[B" || data === "\x1bOB" || data === "j") {
			controller.move(1, count);
			this.invalidate();
			return;
		}
		if (data !== "\r" && data !== "\n") return;
		if (this.mode === "tree") {
			const node = this.nodes[this.selection.index];
			if (!node) return;
			if (node.isCurrent) return this.close();
			this.selectedEntryId = node.id;
			this.mode = "summary";
			this.invalidate();
			return;
		}
		const choices: SessionTreeSummaryMode[] = ["none", "summarize", "custom"];
		const choice = choices[this.summarySelection.index] ?? "none";
		if (choice === "custom") {
			this.mode = "custom";
			this.invalidate();
			return;
		}
		this.navigate(choice);
	}

	private handleCustomInput(data: string): void {
		if (data === "\r" || data === "\n") {
			if (this.customPrompt.trim())
				this.navigate("custom", this.customPrompt.trim());
			return;
		}
		if (data === "\x7f" || data === "\b") {
			this.customPrompt = this.customPrompt.slice(0, -1);
		} else if (data.length === 1 && data >= " ") {
			this.customPrompt += data;
		}
		this.invalidate();
	}

	private navigate(
		summaryMode: SessionTreeSummaryMode,
		customPrompt?: string,
	): void {
		if (!this.selectedEntryId) return;
		this.actionCallback?.({
			type: "navigate",
			entryId: this.selectedEntryId,
			summaryMode,
			customPrompt,
		});
		this.hide();
	}

	private close(): void {
		this.actionCallback?.({ type: "close" });
		this.hide();
	}

	invalidate(): void {
		this.cachedLines = null;
	}

	render(width: number): string[] {
		if (!this.visible) return [];
		if (width === this.cachedWidth && this.cachedLines) return this.cachedLines;
		this.cachedWidth = width;
		const header = theme.fg("header", "");
		const lines: string[] = [`${header}${"─".repeat(width)}${RESET}`];

		if (this.mode === "tree") {
			lines.push(
				`${header} ${BOLD}Session Tree${RESET}${theme.fg("muted", "  ↑/↓ select · Enter navigate · Esc close")}`,
			);
			lines.push(renderSeparator(width));

			const labelWidth = Math.max(1, width - POPUP_FRAME_OVERHEAD - 4);

			for (let i = 0; i < this.nodes.length; i++) {
				const node = this.nodes[i];
				const isSelected = i === this.selection.index;

				// Branch indicator: only show when there are children (branches)
				let branchIndicator = "";
				if (node.hasChildren) {
					branchIndicator = ` (${node.childCount} branch${node.childCount > 1 ? "es" : ""})`;
				}

				// Truncate label to fit
				const indicatorLen = branchIndicator.length;
				const label =
					node.label.length > labelWidth - indicatorLen
						? node.label.slice(0, labelWidth - indicatorLen - 1) + "…"
						: node.label;

				const line = `${isSelected ? theme.fg("selected", "> ") : "  "}${label}${branchIndicator}${node.isCurrent ? " ●" : ""}`;
				lines.push(line);
			}

			if (this.nodes.length === 0) {
				lines.push(`  No completed turns in this session.`);
			}
		} else if (this.mode === "summary") {
			const options = [
				"No summary",
				"Summarize",
				"Summarize with custom prompt",
			];
			lines.push(`${header} ${BOLD}Summarize abandoned branch?${RESET}`);
			lines.push(renderSeparator(width));
			for (const [index, option] of options.entries()) {
				lines.push(
					renderStatusLine(
						`${index === this.summarySelection.index ? ">" : " "} ${option}`,
						Math.max(1, width - POPUP_FRAME_OVERHEAD),
					),
				);
			}
		} else {
			lines.push(`${header} ${BOLD}Custom summarization instructions${RESET}`);
			lines.push(renderSeparator(width));
			lines.push(
				renderStatusLine(
					`${this.customPrompt}${theme.fg("selected", "_")}`,
					Math.max(1, width - POPUP_FRAME_OVERHEAD),
				),
			);
			lines.push(
				renderStatusLine(
					"Enter confirm · Esc back",
					Math.max(1, width - POPUP_FRAME_OVERHEAD),
				),
			);
		}

		lines.push(renderSeparator(width));
		lines.push(`${header}${"─".repeat(width)}${RESET}`);
		this.cachedLines = clampPopupLines(lines, width);
		return this.cachedLines;
	}
}
