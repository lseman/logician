import { visibleWidth } from "../terminal/core.ts";

export interface TerminalCursor {
	row: number;
	column: number;
	visible: boolean;
}

export interface TerminalScreenDiagnostics {
	cursorBoundsViolations: number;
	lastColumnWrites: number;
	printableWrites: number;
	synchronizedUpdateDepth: number;
}

/**
 * Small VT screen model for integration tests. It intentionally models screen
 * behavior rather than terminal styling: cursor movement, erasing, wrapping,
 * scrolling, and printable cells are applied while SGR/OSC/APC metadata is
 * ignored.
 */
export class TerminalScreen {
	private cells: string[][];
	private row = 0;
	private column = 0;
	private savedRow = 0;
	private savedColumn = 0;
	private cursorVisible = true;
	private cursorBoundsViolations = 0;
	private lastColumnWrites = 0;
	private printableWrites = 0;
	private synchronizedUpdateDepth = 0;

	constructor(
		public readonly columns: number,
		public readonly rows: number,
	) {
		this.cells = this.blankScreen();
	}

	write(data: string): void {
		let index = 0;
		while (index < data.length) {
			const char = data[index];
			if (char === "\x1b") {
				index = this.consumeEscape(data, index);
				continue;
			}
			if (char === "\r") {
				this.column = 0;
				index++;
				continue;
			}
			if (char === "\n") {
				this.lineFeed();
				index++;
				continue;
			}
			if (char === "\b") {
				this.column = Math.max(0, this.column - 1);
				index++;
				continue;
			}
			if (char === "\t") {
				this.column = Math.min(
					this.columns - 1,
					Math.ceil((this.column + 1) / 8) * 8,
				);
				index++;
				continue;
			}
			const codePoint = data.codePointAt(index);
			if (codePoint === undefined) break;
			const printable = String.fromCodePoint(codePoint);
			index += printable.length;
			if (codePoint < 0x20 || codePoint === 0x7f) continue;
			this.put(printable);
		}
	}

	line(row: number): string {
		return (this.cells[row] ?? []).join("").replace(/\s+$/u, "");
	}

	lines(): string[] {
		return this.cells.map((_, row) => this.line(row));
	}

	text(): string {
		const lines = this.lines();
		while (lines.at(-1) === "") lines.pop();
		return lines.join("\n");
	}

	cursor(): TerminalCursor {
		return {
			row: this.row,
			column: this.column,
			visible: this.cursorVisible,
		};
	}

	diagnostics(): TerminalScreenDiagnostics {
		return {
			cursorBoundsViolations: this.cursorBoundsViolations,
			lastColumnWrites: this.lastColumnWrites,
			printableWrites: this.printableWrites,
			synchronizedUpdateDepth: this.synchronizedUpdateDepth,
		};
	}

	private blankScreen(): string[][] {
		return Array.from({ length: this.rows }, () =>
			Array<string>(this.columns).fill(" "),
		);
	}

	private consumeEscape(data: string, start: number): number {
		const kind = data[start + 1];
		if (kind === "[") {
			let end = start + 2;
			while (end < data.length) {
				const code = data.charCodeAt(end);
				if (code >= 0x40 && code <= 0x7e) break;
				end++;
			}
			if (end >= data.length) return data.length;
			this.applyCsi(data.slice(start + 2, end), data[end]);
			return end + 1;
		}
		if (kind === "]" || kind === "_") {
			let end = start + 2;
			while (end < data.length) {
				if (data[end] === "\x07") return end + 1;
				if (data[end] === "\x1b" && data[end + 1] === "\\") return end + 2;
				end++;
			}
			return data.length;
		}
		// Charset selection and other two/three-byte ESC sequences do not alter
		// the logical screen used by these tests.
		return Math.min(data.length, start + (kind === "(" || kind === ")" ? 3 : 2));
	}

	private applyCsi(parameterText: string, final: string): void {
		const privateMode = parameterText.startsWith("?");
		const normalized = parameterText.replace(/^[?>!]/u, "");
		const parameters = normalized
			.split(";")
			.map((value) => (value === "" ? 0 : Number(value)))
			.map((value) => (Number.isFinite(value) ? value : 0));
		const first = parameters[0] ?? 0;

		if (privateMode && (final === "h" || final === "l")) {
			if (first === 25) this.cursorVisible = final === "h";
			if (first === 2026) {
				this.synchronizedUpdateDepth += final === "h" ? 1 : -1;
			}
			if (first === 1049 && final === "h") {
				this.cells = this.blankScreen();
				this.row = 0;
				this.column = 0;
			}
			return;
		}

		switch (final) {
			case "A":
				this.row = Math.max(0, this.row - (first || 1));
				break;
			case "B":
				this.row = Math.min(this.rows - 1, this.row + (first || 1));
				break;
			case "C":
				this.column = Math.min(this.columns - 1, this.column + (first || 1));
				break;
			case "D":
				this.column = Math.max(0, this.column - (first || 1));
				break;
			case "G":
				this.column = this.clampColumn((first || 1) - 1);
				break;
			case "H":
			case "f": {
				const requestedRow = (parameters[0] || 1) - 1;
				const requestedColumn = (parameters[1] || 1) - 1;
				if (
					requestedRow < 0 ||
					requestedRow >= this.rows ||
					requestedColumn < 0 ||
					requestedColumn >= this.columns
				) {
					this.cursorBoundsViolations++;
				}
				this.row = this.clampRow(requestedRow);
				this.column = this.clampColumn(requestedColumn);
				break;
			}
			case "J":
				if (first === 2 || first === 3) this.cells = this.blankScreen();
				break;
			case "K":
				this.eraseLine(first);
				break;
			case "X":
				for (
					let column = this.column;
					column < Math.min(this.columns, this.column + (first || 1));
					column++
				) {
					this.cells[this.row][column] = " ";
				}
				break;
			case "s":
				this.savedRow = this.row;
				this.savedColumn = this.column;
				break;
			case "u":
				this.row = this.savedRow;
				this.column = this.savedColumn;
				break;
		}
	}

	private eraseLine(mode: number): void {
		const start = mode === 1 || mode === 2 ? 0 : this.column;
		const end = mode === 0 ? this.columns : this.column + 1;
		for (let column = start; column < end; column++) {
			this.cells[this.row][column] = " ";
		}
	}

	private put(char: string): void {
		const width = Math.max(1, visibleWidth(char));
		if (this.column + width > this.columns) {
			this.column = 0;
			this.lineFeed();
		}
		this.printableWrites += width;
		if (this.column + width >= this.columns) this.lastColumnWrites++;
		this.cells[this.row][this.column] = char;
		for (let offset = 1; offset < width && this.column + offset < this.columns; offset++) {
			this.cells[this.row][this.column + offset] = "";
		}
		this.column += width;
		if (this.column >= this.columns) {
			this.column = 0;
			this.lineFeed();
		}
	}

	private lineFeed(): void {
		this.row++;
		if (this.row < this.rows) return;
		this.cells.shift();
		this.cells.push(Array<string>(this.columns).fill(" "));
		this.row = this.rows - 1;
	}

	private clampRow(row: number): number {
		return Math.min(this.rows - 1, Math.max(0, row));
	}

	private clampColumn(column: number): number {
		return Math.min(this.columns - 1, Math.max(0, column));
	}
}

export function renderTerminalScreen(
	output: string,
	columns: number,
	rows: number,
): TerminalScreen {
	const screen = new TerminalScreen(columns, rows);
	screen.write(output);
	return screen;
}
