import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { resolve } from "node:path";
import type { GroveRepository, GroveSession, GroveState } from "./model.ts";
import { filterSessions, render } from "./render.ts";

const ALT_ON = "\u001b[?1049h\u001b[?25l";
const ALT_OFF = "\u001b[?25h\u001b[?1049l";
const CLEAR = "\u001b[2J\u001b[H";

export interface SessionLaunchCommand {
	readonly executable: string;
	readonly args: readonly string[];
}

/** Build a command that executes TypeScript through tsx, even when Grove runs under Node. */
export function sessionLaunchCommand(sessionId: string): SessionLaunchCommand {
	const require = createRequire(import.meta.url);
	const tsxCli = require.resolve("tsx/cli");
	const entry = resolve(import.meta.dirname, "../../tui/src/index.ts");
	return {
		executable: process.execPath,
		args: [tsxCli, entry, "--session", sessionId],
	};
}

export class GroveApp {
	private sessions: readonly GroveSession[] = [];
	private state: GroveState = {
		screen: { kind: "forest" },
		selection: 0,
		scroll: 0,
		query: "",
	};
	private searching = false;
	private running = false;

	constructor(
		private readonly repository: GroveRepository,
		private readonly cwd: string,
	) {}

	start(): void {
		if (!process.stdin.isTTY || !process.stdout.isTTY) {
			throw new Error("The Logician grove requires an interactive terminal.");
		}
		this.running = true;
		this.refresh();
		process.stdout.write(ALT_ON);
		process.stdin.setRawMode(true);
		process.stdin.resume();
		process.stdin.on("data", this.onInput);
		process.stdout.on("resize", this.onResize);
		this.draw();
	}

	stop(): void {
		if (!this.running) return;
		this.running = false;
		process.stdin.off("data", this.onInput);
		process.stdout.off("resize", this.onResize);
		process.stdin.setRawMode(false);
		process.stdin.pause();
		process.stdout.write(ALT_OFF);
	}

	private refresh(): void {
		this.sessions = this.repository.list(this.cwd);
		this.normalizeSelection();
	}

	private normalizeSelection(): void {
		const count = filterSessions(this.sessions, this.state.query).length;
		const selection = Math.max(
			0,
			Math.min(this.state.selection, Math.max(0, count - 1)),
		);
		const visible = Math.max(1, (process.stdout.rows ?? 24) - 7);
		const scroll = Math.max(0, Math.min(this.state.scroll, selection));
		this.state = {
			...this.state,
			selection,
			scroll: selection >= scroll + visible ? selection - visible + 1 : scroll,
		};
	}

	private selected(): GroveSession | undefined {
		const screen = this.state.screen;
		if (screen.kind === "tree") {
			return this.sessions.find(item => item.id === screen.sessionId);
		}
		return filterSessions(this.sessions, this.state.query)[
			this.state.selection
		];
	}

	private readonly onResize = (): void => this.draw();

	private readonly onInput = (chunk: Buffer): void => {
		const key = chunk.toString("utf8");
		if (this.searching) {
			if (key === "\r" || key === "\n" || key === "\u001b")
				this.searching = false;
			else if (key === "\u007f")
				this.state = {
					...this.state,
					query: this.state.query.slice(0, -1),
					selection: 0,
					scroll: 0,
				};
			else if (key.length === 1 && key >= " ")
				this.state = {
					...this.state,
					query: this.state.query + key,
					selection: 0,
					scroll: 0,
				};
			this.normalizeSelection();
			this.draw();
			return;
		}
		if (key === "q" || key === "\u0003") {
			this.stop();
			return;
		}
		if (key === "r") {
			this.refresh();
			this.draw();
			return;
		}
		if (key === "a" || key === "\r" || key === "\n") {
			const selected = this.selected();
			if (selected) void this.openSession(selected);
			return;
		}
		if (this.state.screen.kind === "tree") {
			if (key === "\u001b" || key === "\u001b[D" || key === "h") {
				this.state = { ...this.state, screen: { kind: "forest" } };
				this.draw();
			}
			return;
		}
		if (key === "/") {
			this.searching = true;
			this.state = { ...this.state, query: "", selection: 0, scroll: 0 };
			this.draw();
			return;
		}
		if (key === "\u001b[C" || key === "l" || key === "t") {
			const selected = this.selected();
			if (selected)
				this.state = {
					...this.state,
					screen: { kind: "tree", sessionId: selected.id },
				};
			this.draw();
			return;
		}
		const delta =
			key === "\u001b[A" || key === "k"
				? -1
				: key === "\u001b[B" || key === "j"
					? 1
					: 0;
		if (delta) {
			this.state = { ...this.state, selection: this.state.selection + delta };
			this.normalizeSelection();
			this.draw();
		}
	};

	private draw(): void {
		if (!this.running) return;
		process.stdout.write(
			`${CLEAR}${render(this.sessions, this.state, process.stdout.columns ?? 80, process.stdout.rows ?? 24)}`,
		);
	}

	private async openSession(session: GroveSession): Promise<void> {
		process.stdin.off("data", this.onInput);
		process.stdin.setRawMode(false);
		process.stdout.write(ALT_OFF);
		const command = sessionLaunchCommand(session.id);
		await new Promise<void>(done => {
			const child = spawn(command.executable, command.args, {
				cwd: session.cwd,
				stdio: "inherit",
			});
			child.once("exit", () => done());
			child.once("error", error => {
				process.stderr.write(`Failed to open Logician: ${error.message}\n`);
				done();
			});
		});
		if (!this.running) return;
		this.refresh();
		process.stdout.write(ALT_ON);
		process.stdin.setRawMode(true);
		process.stdin.resume();
		process.stdin.on("data", this.onInput);
		this.draw();
	}
}
