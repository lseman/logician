// ── TaskManager ─────────────────────────────────────────────────────────────
// Manages background asynchronous bash processes, log streams, stdin interaction,
// status inspection, and lifecycle termination.

import type { ChildProcess } from "node:child_process";
import {
	createWriteStream,
	existsSync,
	mkdirSync,
	readFileSync,
	type WriteStream,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { killProcessTree, untrackDetachedChildPid } from "./shell.ts";
import { OutputAccumulator } from "./truncate.ts";

export type TaskStatus =
	| "running"
	| "completed"
	| "failed"
	| "timed_out"
	| "killed"
	| "aborted";

export interface BackgroundTaskSummary {
	id: string;
	command: string;
	cwd: string;
	pid: number | undefined;
	startTime: number;
	endTime?: number;
	durationMs: number;
	status: TaskStatus;
	exitCode: number | null;
	signal: string | null;
	logFilePath: string;
}

export interface BackgroundTaskStatusDetails extends BackgroundTaskSummary {
	recentOutput: string;
	totalLines: number;
	totalBytes: number;
}

export interface TaskEntry {
	id: string;
	command: string;
	cwd: string;
	pid: number | undefined;
	child: ChildProcess;
	startTime: number;
	endTime?: number;
	status: TaskStatus;
	exitCode: number | null;
	signal: string | null;
	logFilePath: string;
	output: OutputAccumulator;
	logStream?: WriteStream;
}

export class TaskManager {
	private tasks = new Map<string, TaskEntry>();
	private taskCounter = 0;
	private baseLogDir: string;

	constructor(customLogDir?: string) {
		this.baseLogDir =
			customLogDir ?? join(tmpdir(), "logician-tasks");
		if (!existsSync(this.baseLogDir)) {
			try {
				mkdirSync(this.baseLogDir, { recursive: true });
			} catch {
				// Fallback to tmpdir directly
				this.baseLogDir = tmpdir();
			}
		}
	}

	createTaskId(): string {
		this.taskCounter += 1;
		return `task-${this.taskCounter}`;
	}

	getLogPathForTask(taskId: string): string {
		return join(this.baseLogDir, `${taskId}.log`);
	}

	registerTask(options: {
		id?: string;
		command: string;
		cwd: string;
		child: ChildProcess;
		output?: OutputAccumulator;
		logFilePath?: string;
		startTime?: number;
	}): TaskEntry {
		const id = options.id ?? this.createTaskId();
		const logFilePath =
			options.logFilePath ?? this.getLogPathForTask(id);
		const output =
			options.output ??
			new OutputAccumulator({ tempFilePrefix: `task-${id}` });

		let logStream: WriteStream | undefined;
		try {
			logStream = createWriteStream(logFilePath, { flags: "a" });
		} catch {
			// WriteStream creation failed, output still tracked via OutputAccumulator
		}

		const entry: TaskEntry = {
			id,
			command: options.command,
			cwd: options.cwd,
			pid: options.child.pid,
			child: options.child,
			startTime: options.startTime ?? Date.now(),
			status: "running",
			exitCode: null,
			signal: null,
			logFilePath,
			output,
			logStream,
		};

		this.tasks.set(id, entry);

		// Pipe any new data to the logStream
		const handleData = (data: Buffer | string) => {
			if (entry.logStream && !entry.logStream.destroyed) {
				entry.logStream.write(data);
			}
		};

		options.child.stdout?.on("data", handleData);
		options.child.stderr?.on("data", handleData);

		options.child.on("close", (code, signal) => {
			entry.endTime = Date.now();
			entry.exitCode = code;
			entry.signal = signal;
			if (entry.status === "running") {
				entry.status = code === 0 ? "completed" : "failed";
			}
			if (entry.pid) {
				untrackDetachedChildPid(entry.pid);
			}
			if (entry.logStream) {
				entry.logStream.end();
			}
			entry.output.finish();
		});

		options.child.on("error", () => {
			entry.endTime = Date.now();
			entry.status = "failed";
			if (entry.pid) {
				untrackDetachedChildPid(entry.pid);
			}
			if (entry.logStream) {
				entry.logStream.end();
			}
			entry.output.finish();
		});

		return entry;
	}

	getTask(id: string): TaskEntry | undefined {
		return this.tasks.get(id);
	}

	listTasks(): BackgroundTaskSummary[] {
		const now = Date.now();
		return Array.from(this.tasks.values()).map(entry => {
			const durationMs = (entry.endTime ?? now) - entry.startTime;
			return {
				id: entry.id,
				command: entry.command,
				cwd: entry.cwd,
				pid: entry.pid,
				startTime: entry.startTime,
				endTime: entry.endTime,
				durationMs,
				status: entry.status,
				exitCode: entry.exitCode,
				signal: entry.signal,
				logFilePath: entry.logFilePath,
			};
		});
	}

	getTaskStatus(
		id: string,
		maxLines = 50,
	): BackgroundTaskStatusDetails | null {
		const entry = this.tasks.get(id);
		if (!entry) return null;

		const now = Date.now();
		const durationMs = (entry.endTime ?? now) - entry.startTime;

		let recentOutput = "";
		let totalLines = 0;
		let totalBytes = 0;

		if (existsSync(entry.logFilePath)) {
			try {
				const content = readFileSync(entry.logFilePath, "utf8");
				totalBytes = Buffer.byteLength(content, "utf8");
				const lines = content.split("\n");
				totalLines = lines.length;
				const sliced = lines.slice(-maxLines);
				recentOutput = sliced.join("\n");
			} catch {
				const snap = entry.output.snapshot();
				recentOutput = snap.content;
				totalLines = snap.truncation.totalLines;
				totalBytes = snap.truncation.totalBytes;
			}
		} else {
			const snap = entry.output.snapshot();
			recentOutput = snap.content;
			totalLines = snap.truncation.totalLines;
			totalBytes = snap.truncation.totalBytes;
		}

		return {
			id: entry.id,
			command: entry.command,
			cwd: entry.cwd,
			pid: entry.pid,
			startTime: entry.startTime,
			endTime: entry.endTime,
			durationMs,
			status: entry.status,
			exitCode: entry.exitCode,
			signal: entry.signal,
			logFilePath: entry.logFilePath,
			recentOutput,
			totalLines,
			totalBytes,
		};
	}

	sendInput(
		id: string,
		input: string,
	): { success: boolean; message: string } {
		const entry = this.tasks.get(id);
		if (!entry) {
			return { success: false, message: `Task "${id}" not found.` };
		}
		if (entry.status !== "running") {
			return {
				success: false,
				message: `Task "${id}" is not running (status: ${entry.status}).`,
			};
		}
		if (!entry.child.stdin || entry.child.stdin.destroyed) {
			return {
				success: false,
				message: `Task "${id}" stdin is not available.`,
			};
		}

		const data = input.endsWith("\n") ? input : `${input}\n`;
		entry.child.stdin.write(data);
		return {
			success: true,
			message: `Sent ${Buffer.byteLength(data, "utf8")} bytes to task "${id}".`,
		};
	}

	killTask(id: string): { success: boolean; message: string } {
		const entry = this.tasks.get(id);
		if (!entry) {
			return { success: false, message: `Task "${id}" not found.` };
		}
		if (entry.status !== "running") {
			return {
				success: false,
				message: `Task "${id}" is already finished (status: ${entry.status}).`,
			};
		}

		entry.status = "killed";
		entry.endTime = Date.now();

		if (entry.pid) {
			killProcessTree(entry.pid);
			untrackDetachedChildPid(entry.pid);
		} else {
			entry.child.kill("SIGKILL");
		}

		if (entry.logStream) {
			entry.logStream.end();
		}
		entry.output.finish();

		return {
			success: true,
			message: `Task "${id}" (PID: ${entry.pid ?? "unknown"}) terminated.`,
		};
	}

	cleanupAll(): void {
		for (const entry of this.tasks.values()) {
			if (entry.status === "running") {
				entry.status = "killed";
				entry.endTime = Date.now();
				if (entry.pid) {
					killProcessTree(entry.pid);
					untrackDetachedChildPid(entry.pid);
				} else {
					entry.child.kill("SIGKILL");
				}
				if (entry.logStream) {
					entry.logStream.end();
				}
				entry.output.finish();
			}
		}
	}
}

/** Global default TaskManager instance. */
export const defaultTaskManager = new TaskManager();
