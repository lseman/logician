import assert from "node:assert/strict";
import test from "node:test";
import {
	CURSOR_MARKER,
	diffTerminalLine,
	diffTerminalLineWithMetrics,
	TUI,
	type RendererMetrics,
} from "../terminal/core.ts";
import { renderTerminalScreen } from "../testing/terminal-screen.ts";

void test("cell diff writes only the changed run on the second frame", () => {
	const update = diffTerminalLine("alpha bravo", "alpha Xravo", 0, 19);

	assert.ok(update.includes("\x1b[1;7H"));
	assert.ok(update.includes("\x1b[0mX"));
	assert.doesNotMatch(update, /alpha/);
	assert.doesNotMatch(update, /\r|\n/);
});

void test("cell diff addresses columns after a wide character correctly", () => {
	const update = diffTerminalLine("界abc", "界aXc", 0, 19);

	assert.ok(update.includes("\x1b[1;4H"));
	assert.ok(update.includes("\x1b[0mX"));
});

void test("applying a diff produces exactly the requested logical row", () => {
	const previous = "alpha bravo";
	const next = "alpha Xravo";
	const update = diffTerminalLine(previous, next, 0, 19);
	const screen = renderTerminalScreen(
		`\x1b[1;1H${previous}\x1b[?2026h${update}\x1b[?2026l`,
		20,
		2,
	);

	assert.equal(screen.line(0), next);
	assert.deepEqual(screen.diagnostics(), {
		cursorBoundsViolations: 0,
		lastColumnWrites: 0,
		printableWrites: previous.length + 1,
		synchronizedUpdateDepth: 0,
	});
});

void test("unchanged rows emit no output or cursor movement", () => {
	const diff = diffTerminalLineWithMetrics("unchanged", "unchanged", 3, 79);

	assert.deepEqual(diff, {
		output: "",
		changedCells: 0,
		cursorMoves: 0,
	});
});

void test("a one-cell update stays inside the small-update budget", () => {
	const diff = diffTerminalLineWithMetrics(
		"spinner ⠋ ready",
		"spinner ⠙ ready",
		4,
		79,
	);

	assert.equal(diff.changedCells, 1);
	assert.equal(diff.cursorMoves, 1);
	assert.ok(Buffer.byteLength(diff.output) <= 100);
	const screen = renderTerminalScreen(
		`\x1b[?2026h${diff.output}\x1b[?2026l`,
		80,
		24,
	);
	assert.deepEqual(screen.diagnostics(), {
		cursorBoundsViolations: 0,
		lastColumnWrites: 0,
		printableWrites: 1,
		synchronizedUpdateDepth: 0,
	});
});

void test("an unchanged TUI frame performs zero terminal writes", () => {
	const columnsDescriptor = Object.getOwnPropertyDescriptor(process.stdout, "columns");
	const rowsDescriptor = Object.getOwnPropertyDescriptor(process.stdout, "rows");
	const writes: string[] = [];
	const tui = new TUI({} as NodeJS.WriteStream);
	tui.setInputBarComponent({
		render: () => [`› prompt${CURSOR_MARKER}`],
	});
	tui.setFixedBottomComponent({ render: () => ["ready"] });
	const internal = tui as unknown as {
		doRender(): void;
		write(data: string): void;
		getLastRenderMetrics(): RendererMetrics;
	};
	internal.write = (data) => writes.push(data);

	try {
		Object.defineProperty(process.stdout, "columns", {
			configurable: true,
			value: 40,
		});
		Object.defineProperty(process.stdout, "rows", {
			configurable: true,
			value: 8,
		});
		internal.doRender();
		writes.length = 0;

		internal.doRender();

		const metrics = internal.getLastRenderMetrics();
		assert.equal(metrics.bytesWritten, 0);
		assert.equal(metrics.changedCells, 0);
		assert.equal(metrics.cursorMoves, 0);
		assert.equal(metrics.dirtyRows, 0);
		assert.equal(metrics.dirtyRegion, null);
		assert.ok(metrics.frameTimeMs >= 0);
		assert.ok(metrics.layoutTimeMs >= 0);
		assert.ok(metrics.diffTimeMs >= 0);
		assert.ok(metrics.writeTimeMs >= 0);
		assert.deepEqual(writes, []);
	} finally {
		if (columnsDescriptor) {
			Object.defineProperty(process.stdout, "columns", columnsDescriptor);
		}
		if (rowsDescriptor) {
			Object.defineProperty(process.stdout, "rows", rowsDescriptor);
		}
	}
});

void test("streaming updates always re-park the cursor at the composer", () => {
	const columnsDescriptor = Object.getOwnPropertyDescriptor(process.stdout, "columns");
	const rowsDescriptor = Object.getOwnPropertyDescriptor(process.stdout, "rows");
	const writes: string[] = [];
	let status = "streaming 1";
	const tui = new TUI({} as NodeJS.WriteStream);
	tui.setInputBarComponent({
		render: () => [`› prompt${CURSOR_MARKER}`],
	});
	tui.setFixedBottomComponent({ render: () => [status] });
	const internal = tui as unknown as {
		doRender(): void;
		write(data: string): void;
		getLastRenderMetrics(): RendererMetrics;
	};
	internal.write = (data) => writes.push(data);

	try {
		Object.defineProperty(process.stdout, "columns", {
			configurable: true,
			value: 40,
		});
		Object.defineProperty(process.stdout, "rows", {
			configurable: true,
			value: 8,
		});
		internal.doRender();
		writes.length = 0;
		status = "streaming 2";

		internal.doRender();

		assert.equal(writes.length, 1);
		const update = writes[0];
		assert.ok(update.includes("\x1b[8;11H"));
		assert.ok(update.includes("\x1b[6;9H"));
		assert.ok(
			update.indexOf("\x1b[6;9H") < update.indexOf("\x1b[?2026l"),
			"cursor restoration must happen before synchronized update ends",
		);
		const metrics = internal.getLastRenderMetrics();
		assert.equal(metrics.dirtyRows, 1);
		assert.deepEqual(metrics.dirtyRegion, { top: 7, bottom: 7 });
		assert.equal(metrics.changedCells, 1);
		assert.ok(metrics.frameTimeMs >= metrics.layoutTimeMs);
		assert.ok(metrics.frameTimeMs >= metrics.diffTimeMs);
		assert.ok(metrics.frameTimeMs >= metrics.writeTimeMs);
	} finally {
		if (columnsDescriptor) {
			Object.defineProperty(process.stdout, "columns", columnsDescriptor);
		}
		if (rowsDescriptor) {
			Object.defineProperty(process.stdout, "rows", rowsDescriptor);
		}
	}
});

void test("a renderer crash closes terminal state and invalidates the frame cache", () => {
	const tui = new TUI({} as NodeJS.WriteStream);
	const internal = tui as unknown as {
		doRender(): void;
		previousLines: string[];
		previousCursorRow: number;
		previousCursorCol: number;
		previousCursorVisible: boolean | null;
		overlayStack: Array<{
			component: { render(): string[] };
			hidden: boolean;
			focusOrder: number;
		}>;
	};
	internal.previousLines = ["stale frame"];
	internal.previousCursorRow = 4;
	internal.previousCursorCol = 9;
	internal.previousCursorVisible = false;
	internal.overlayStack = [{
		component: {
			render: () => {
				throw new Error("broken overlay");
			},
		},
		hidden: false,
		focusOrder: 1,
	}];

	let recovery = "";
	const originalStderrWrite = process.stderr.write;
	const originalConsoleError = console.error;
	process.stderr.write = ((chunk: string | Uint8Array) => {
		recovery += String(chunk);
		return true;
	}) as typeof process.stderr.write;
	console.error = () => {};
	try {
		internal.doRender();
	} finally {
		process.stderr.write = originalStderrWrite;
		console.error = originalConsoleError;
	}

	assert.deepEqual(internal.previousLines, []);
	assert.equal(internal.previousCursorRow, -1);
	assert.equal(internal.previousCursorCol, -1);
	assert.equal(internal.previousCursorVisible, null);
	assert.ok(recovery.includes("\x1b[?2026l"));
	assert.ok(recovery.includes("\x1b]8;;\x1b\\"));
	assert.ok(recovery.includes("\x1b[0m\x1b[2J\x1b[H\x1b[?25h"));
	assert.match(recovery, /TUI render error.*broken overlay/s);
});
