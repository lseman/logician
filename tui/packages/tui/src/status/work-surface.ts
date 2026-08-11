import type { TurnPhase } from "../state/turn-state.ts";
import {
	type Component,
	clampLineToWidth,
	RESET,
	visibleWidth,
} from "../terminal/core.ts";
import { theme } from "../terminal/theme.ts";

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
export class WorkSurface implements Component {
	private workingSet: string[] = [];
	private calls = new Map<string, ToolRecord>();
	private evidence: Evidence = this.emptyEvidence();
	private active = false;
	private context = "";
	private loopIteration = 0;
	private turnCount = 0;
	private onInvalidate: (() => void) | null = null;
	private revision = 0;
	private cachedWidth = -1;
	private cachedRevision = -1;
	private cachedLines: string[] | null = null;

	setOnInvalidate(cb: () => void): void {
		this.onInvalidate = cb;
	}

	/** Start a user-visible agent run and reset all per-run activity. */
	startRun(): void {
		this.active = true;
		this.calls.clear();
		this.evidence = this.emptyEvidence();
		this.turnCount = 0;
		this.revision++;
		this.onInvalidate?.();
	}

	/** Record one provider/tool turn inside the active agent run. */
	startTurn(): void {
		this.active = true;
		this.calls.clear();
		this.turnCount++;
		this.revision++;
		this.onInvalidate?.();
	}

	endTurn(): void {
		this.active = false;
		this.revision++;
		this.onInvalidate?.();
	}

	setPhase(phase: TurnPhase): void {
		this.active = ![
			"idle",
			"complete",
			"failed",
			"waiting",
			"approval",
		].includes(phase);
		this.revision++;
		this.onInvalidate?.();
	}

	setContext(tokens: number, maxTokens?: number): void {
		this.context = maxTokens
			? `${tokens.toLocaleString()}/${maxTokens.toLocaleString()}`
			: tokens > 0
				? tokens.toLocaleString()
				: "";
		this.revision++;
		this.onInvalidate?.();
	}

	setLoopIteration(iteration: number): void {
		const changed = this.loopIteration !== iteration;
		this.loopIteration = iteration;
		if (changed) {
			this.turnCount = 0;
		}
		this.revision++;
		this.onInvalidate?.();
	}

	/** Reset all accumulated counters and state (called when a new session/run starts). */
	reset(): void {
		this.active = false;
		this.workingSet = [];
		this.calls.clear();
		this.evidence = this.emptyEvidence();
		this.context = "";
		this.turnCount = 0;
		this.loopIteration = 0;
		this.revision++;
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
		this.revision++;
		this.onInvalidate?.();
	}

	recordToolEnd(
		id: string | undefined,
		_name: string,
		result = "",
		isError = false,
	): void {
		if (isError) this.evidence.failures++;
		if (result.includes("<post_edit_diagnostics")) this.evidence.diagnostics++;
		const call = id ? this.calls.get(id) : undefined;
		const file =
			call && typeof call.args.path === "string" ? call.args.path : "";
		if (file && (isError || result.includes("<post_edit_diagnostics")))
			this.touch(file);
		this.revision++;
		this.onInvalidate?.();
	}

	getWorkingSet(): string[] {
		return this.workingSet;
	}

	render(width: number): string[] {
		if (
			this.cachedLines !== null &&
			this.cachedWidth === width &&
			this.cachedRevision === this.revision
		) {
			return this.cachedLines;
		}
		this.cachedWidth = width;
		this.cachedRevision = this.revision;
		this.cachedLines = this.renderUncached(width);
		return this.cachedLines;
	}

	private renderUncached(width: number): string[] {
		if (
			this.workingSet.length === 0 &&
			this.evidence.tools === 0 &&
			!this.context
		) {
			return [];
		}
		const lines: string[] = [];
		const work = this.workingSet.slice(0, 8).join("  ·  ");
		if (work) {
			lines.push(
				this.line(
					`${theme.fg("muted", "Working set")}  ${theme.fg("text", work)}${RESET}`,
					width,
				),
			);
		}
		if (this.evidence.tools > 0 || this.context) {
			const label = this.active ? "Activity" : "Run summary";
			const turnLabel =
				this.turnCount > 0
					? `${this.turnCount} turn${this.turnCount === 1 ? "" : "s"}`
					: "0 turns";
			const state = this.active
				? theme.fg("warning", "● running")
				: theme.fg("success", "✓");
			const parts = [
				turnLabel,
				this.loopIteration > 0 ? `loop ${this.loopIteration}` : "",
				`${this.evidence.tools} tools`,
				this.evidence.changed.size
					? `${this.evidence.changed.size} changed`
					: "",
				this.evidence.commands ? `${this.evidence.commands} commands` : "",
				this.evidence.diagnostics
					? `${this.evidence.diagnostics} diagnostics`
					: "",
				this.evidence.failures ? `${this.evidence.failures} failed` : "",
				this.context ? `context ${this.context}` : "",
			].filter(Boolean);
			lines.push(
				this.line(
					`${theme.fg("muted", label)}  ${state}${RESET}  ${theme.fg("dim", parts.join(" · "))}${RESET}`,
					width,
				),
			);
		}
		return lines;
	}

	private touch(file: string): void {
		this.workingSet = [
			file,
			...this.workingSet.filter(item => item !== file),
		].slice(0, 8);
	}

	private emptyEvidence(): Evidence {
		return {
			tools: 0,
			changed: new Set(),
			commands: 0,
			failures: 0,
			diagnostics: 0,
		};
	}

	private line(value: string, width: number): string {
		const output = clampLineToWidth(value, width);
		return output + " ".repeat(Math.max(0, width - visibleWidth(output)));
	}
}
