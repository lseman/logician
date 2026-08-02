import {
	ansiToInkTextRow,
	type InkTextComponent,
	type InkTextSpan,
	type InkTextRow,
	padInkTextRow,
} from "../terminal/core.ts";
import { theme } from "../terminal/theme.ts";
import type { TurnPhase } from "../state/turn-state.ts";

const MUTATING_TOOLS = new Set(["edit_file", "write_file"]);
const FILE_TOOLS = new Set(["read_file", "edit_file", "write_file"]);

interface ToolRecord {
	name: string;
	args: Record<string, unknown>;
}

interface Evidence {
	tools: number;
	changed: Set<string>;
	commands: number;
	failures: number;
	diagnostics: number;
}

/** Composer-attached working set and evidence for the current/last turn. */
export class WorkSurface implements InkTextComponent {
	private workingSet: string[] = [];
	private calls = new Map<string, ToolRecord>();
	private evidence: Evidence = this.emptyEvidence();
	private active = false;
	private context = "";
	private onInvalidate: (() => void) | null = null;

	setOnInvalidate(cb: () => void): void {
		this.onInvalidate = cb;
	}

	startTurn(): void {
		this.active = true;
		this.calls.clear();
		this.evidence = this.emptyEvidence();
		this.onInvalidate?.();
	}

	endTurn(): void {
		this.active = false;
		this.onInvalidate?.();
	}

	setPhase(phase: TurnPhase): void {
		this.active = !["idle", "complete", "failed", "waiting", "approval"].includes(
			phase,
		);
		this.onInvalidate?.();
	}

	setContext(tokens: number, maxTokens?: number): void {
		this.context = maxTokens
			? `${tokens.toLocaleString()}/${maxTokens.toLocaleString()}`
			: tokens > 0 ? tokens.toLocaleString() : "";
		this.onInvalidate?.();
	}

	recordToolStart(
		id: string | undefined,
		name: string,
		args: Record<string, unknown> = {},
	): void {
		const key = id || `${name}:${this.evidence.tools}`;
		this.calls.set(key, { name, args });
		this.evidence.tools++;
		const file = typeof args.path === "string" ? args.path : "";
		if (file && FILE_TOOLS.has(name)) this.touch(file);
		if (file && MUTATING_TOOLS.has(name)) this.evidence.changed.add(file);
		if (name === "bash") this.evidence.commands++;
		this.onInvalidate?.();
	}

	recordToolEnd(
		id: string | undefined,
		name: string,
		result = "",
		isError = false,
	): void {
		if (isError) this.evidence.failures++;
		if (result.includes("<post_edit_diagnostics")) this.evidence.diagnostics++;
		const call = id ? this.calls.get(id) : undefined;
		const file = call && typeof call.args.path === "string" ? call.args.path : "";
		if (file && (isError || result.includes("<post_edit_diagnostics"))) this.touch(file);
		this.onInvalidate?.();
	}

	getWorkingSet(): string[] {
		return this.workingSet;
	}

	getInkTextRows(width: number): InkTextRow[] {
		if (this.workingSet.length === 0 && this.evidence.tools === 0 && !this.context) {
			return [];
		}
		const rows: InkTextRow[] = [];
		const work = this.workingSet.slice(0, 8).join("  ·  ");
		if (work) {
			rows.push(this.row([
				this.themedSpan("muted", "Working set"),
				{ text: "  " },
				this.themedSpan("text", work),
			], width));
		}
		if (this.evidence.tools > 0 || this.context) {
			const label = this.active ? "Activity" : "Turn summary";
			const state = this.active ? "● running" : "✓";
			const parts = [
				`${this.evidence.tools} tools`,
				this.evidence.changed.size ? `${this.evidence.changed.size} changed` : "",
				this.evidence.commands ? `${this.evidence.commands} commands` : "",
				this.evidence.diagnostics ? `${this.evidence.diagnostics} diagnostics` : "",
				this.evidence.failures ? `${this.evidence.failures} failed` : "",
				this.context ? `context ${this.context}` : "",
			].filter(Boolean);
			rows.push(this.row([
				this.themedSpan("muted", label),
				{ text: "  " },
				this.themedSpan(this.active ? "warning" : "success", state),
				{ text: "  " },
				this.themedSpan("text", parts.join(" · ")),
			], width));
		}
		return rows;
	}

	private touch(file: string): void {
		this.workingSet = [file, ...this.workingSet.filter((item) => item !== file)]
			.slice(0, 8);
	}

	private emptyEvidence(): Evidence {
		return { tools: 0, changed: new Set(), commands: 0, failures: 0, diagnostics: 0 };
	}

	private themedSpan(color: Parameters<typeof theme.fg>[0], text: string): InkTextSpan {
		const [span] = ansiToInkTextRow(theme.fg(color, text));
		return span ?? { text };
	}

	private row(spans: InkTextRow, width: number): InkTextRow {
		return padInkTextRow(spans, width);
	}
}
