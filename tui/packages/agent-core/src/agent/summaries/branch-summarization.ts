// ── Branch summarization for session branching ────────────────────────────────
// Generates structured summaries of abandoned branches when navigating
// back to a previous conversation point.

import type { Message } from "../types.ts";

// ============================================================================
// Types
// ============================================================================

/** Structured sections of a branch summary. */
export interface BranchSummary {
	/** What was the user trying to accomplish in this branch? */
	goal: string;
	/** Constraints, preferences, or requirements mentioned. */
	constraints: string[];
	/** Progress tracking. */
	progress: BranchProgressRaw;
	/** Key decisions with rationale. */
	keyDecisions: Array<{ decision: string; rationale: string }>;
	/** Next steps to continue this work. */
	nextSteps: string[];
	/** Human-readable full summary combining all sections. */
	full: string;
}

/** Progress sub-section. */
export interface BranchProgressRaw {
	done: string[];
	inProgress: string[];
	blocked: string[];
}

/** File operations tracked during branch execution. */
export interface FileOperations {
	read: Set<string>;
	modified: Set<string>;
}

// ============================================================================
// File Operations Extraction
// ============================================================================

/** Create an empty file operations tracker. */
export function createFileOps(): FileOperations {
	return { read: new Set(), modified: new Set() };
}

/** Extract file operations from messages (reads from tool calls, modifications from assistant messages). */
export function extractFileOpsFromMessages(messages: Message[]): FileOperations {
	const ops = createFileOps();

	for (const msg of messages) {
		if (!msg) continue;
		if (msg.role !== "assistant" || !msg.tool_calls) continue;

		for (const tc of msg.tool_calls) {
			const name = tc.name;
			const argsStr = typeof tc.arguments === "string" ? tc.arguments : JSON.stringify(tc.arguments ?? {});

			try {
				const args = JSON.parse(argsStr);

				// Read operations
				if (name === "read_file" || name === "read" || name === "cat") {
					const path = args.path ?? args.file ?? args.filename;
					if (path) ops.read.add(String(path));
				}

				// Write/modify operations
				if (
					name === "edit_file" ||
					name === "write_file" ||
					name === "edit" ||
					name === "create_file" ||
					name === "file_diff" ||
					name === "sed"
				) {
					const path = args.path ?? args.file ?? args.filename;
					if (path) ops.modified.add(String(path));
				}

				// Git operations imply file tracking
				if (name === "git") {
					const subCmd = args.subcommand ?? args.command ?? "";
					const paths = args.paths ?? args.files ?? (typeof args.command === "string" ? extractGitPaths(args.command) : []);
					for (const p of paths) {
						ops.modified.add(p);
					}
					if (subCmd.includes("diff") || subCmd.includes("show")) {
						for (const p of paths) ops.read.add(p);
					}
				}
			} catch (_e: unknown) {
				// Malformed JSON — skip
			}
		}
	}

	return ops;
}

/** Extract file paths from git command strings like "add file.txt" or "checkout src/main.ts". */
function extractGitPaths(cmd: string): string[] {
	const paths: string[] = [];
	// Common git commands that take file paths
	const gitActions = ["add", "checkout", "restore", "rm", "mv", "commit", "show", "diff", "log", "blame"];
	for (const action of gitActions) {
		const idx = cmd.indexOf(action);
		if (idx >= 0) {
			const after = cmd.slice(idx + action.length).trim();
			// Split by space, take non-flag tokens
			for (const token of after.split(/\s+/)) {
				if (!token.startsWith("-")) paths.push(token);
			}
		}
	}
	return paths;
}

/** Format file operations for appending to a summary string. */
export function formatFileOperations(readFiles: string[], modifiedFiles: string[]): string {
	const parts: string[] = [];
	if (readFiles.length > 0) parts.push(`Read files: ${readFiles.join(", ")}`);
	if (modifiedFiles.length > 0) parts.push(`Modified files: ${modifiedFiles.join(", ")}`);
	return parts.length > 0 ? `\n\n${parts.join("\n")}` : "";
}

/** Compute deduplicated file lists from file operations. */
export function computeFileLists(ops: FileOperations): { readFiles: string[]; modifiedFiles: string[] } {
	return {
		readFiles: sorted(ops.read),
		modifiedFiles: sorted(ops.modified),
	};
}

function sorted(set: Set<string>): string[] {
	return [...set].sort();
}

// ============================================================================
// Entry Collection
// ============================================================================

/** Result of collecting entries between two branch positions. */
export interface BranchCollectionResult {
	/** Messages to summarize, in chronological order. */
	messages: Message[];
	/** Common ancestor index in the messages array (or -1 if no common ancestor). */
	commonAncestorIndex: number;
	/** Total estimated tokens in collected messages. */
	totalTokens: number;
	/** File operations from the branch. */
	fileOps: FileOperations;
}

/**
 * Collect messages for branch summarization.
 * Finds the common ancestor between the current history and the fork point,
 * then collects messages from the fork point to the divergence point.
 *
 * @param currentHistory - Current conversation history
 * @param forkParent - Parent history at fork point
 * @param forkedAt - Index where the branch diverged
 * @param tokenBudget - Max tokens to include (0 = no limit, includes all)
 * @returns Messages to summarize and file operations
 */
export function collectMessagesForBranchSummary(
	currentHistory: Message[],
	forkParent: Message[],
	forkedAt: number,
	tokenBudget = 0,
): BranchCollectionResult {
	const fileOps = createFileOps();
	const messages: Message[] = [];
	let totalTokens = 0;

	// Find common ancestor by matching messages between parent and current history
	let commonAncestorIndex = -1;
	const maxCommon = Math.min(forkParent.length, forkedAt);
	for (let i = maxCommon - 1; i >= 0; i--) {
		if (messageEquals(currentHistory[i], forkParent[i])) {
			commonAncestorIndex = i;
			break;
		}
	}

	// Collect messages from fork point to divergence point (newest first, then reverse)
	const branchMessages = currentHistory.slice(forkedAt);

	// Walk from newest to oldest with token budget
	for (let i = branchMessages.length - 1; i >= 0; i--) {
		const msg = branchMessages[i];
		const tokens = estimateMessageTokens(msg);

		if (tokenBudget > 0 && totalTokens + tokens > tokenBudget && messages.length > 0) {
			// Stop at budget, but always include at least one message
			break;
		}

		messages.unshift(msg);
		totalTokens += tokens;
	}

	// Extract file ops from collected messages
	const extractedOps = extractFileOpsFromMessages(messages);
	for (const f of extractedOps.read) fileOps.read.add(f);
	for (const f of extractedOps.modified) fileOps.modified.add(f);

	return { messages, commonAncestorIndex, totalTokens, fileOps };
}

/** Check if two messages are semantically equal (for common ancestor detection). */
function messageEquals(a: Message, b: Message): boolean {
	if (!a || !b) return false;
	if (a.role !== b.role) return false;
	if (typeof a.content !== typeof b.content) return false;
	if (typeof a.content === "string" && a.content !== b.content) return false;
	if (Array.isArray(a.tool_calls) && Array.isArray(b.tool_calls)) {
		if (a.tool_calls.length !== b.tool_calls.length) return false;
	}
	return true;
}

/** Estimate tokens for a single message. */
function estimateMessageTokens(msg: Message): number {
	if (!msg) return 0;
	const text = typeof msg.content === "string" ? msg.content : "";
	return Math.max(1, Math.floor(text.length / 4));
}

// ============================================================================
// Summary Parsing (for post-processing LLM output)
// ============================================================================

/**
 * Parse a structured branch summary string into typed fields.
 * Robust parser that handles variations in LLM output format.
 */
export function parseBranchSummary(text: string): Partial<BranchSummary> {
	const result: Partial<BranchSummary> = {};

	// Goal
	const goalMatch = text.match(/## Goal\s*\n([\s\S]*?)(?=\n##(?!\d|#)|$)/);
	if (goalMatch) {
		result.goal = goalMatch[1].trim().split("\n")[0].trim();
	}

	// Constraints
	const constraintsMatch = text.match(/## Constraints & Preferences\s*\n([\s\S]*?)(?=\n##(?!\d|#)|$)/);
	if (constraintsMatch) {
		const items = parseListItems(constraintsMatch[1]);
		result.constraints = items.filter((item) => !item.match(/^(none|\(none\))$/i) || item.match(/^\(none\) if none/));
		if (result.constraints.length === 1 && /^(none|\(none\))$/i.test(result.constraints[0])) {
			result.constraints = [];
		}
	}

	// Progress
	const progressMatch = text.match(/## Progress\s*\n([\s\S]*?)(?=\n##(?!\d|#)|$)/);
	if (progressMatch) {
		const progText = progressMatch[1];
		const doneMatch = progText.match(/### Done\s*\n([\s\S]*?)(?=\n###(?!\d|#)|$)/);
		const inProgMatch = progText.match(/### In Progress\s*\n([\s\S]*?)(?=\n###(?!\d|#)|$)/);
		const blockedMatch = progText.match(/### Blocked\s*\n([\s\S]*?)(?=\n##(?!\d|#)|$)/);

		result.progress = {
			done: doneMatch ? parseListItems(doneMatch[1]) : [],
			inProgress: inProgMatch ? parseListItems(inProgMatch[1]) : [],
			blocked: blockedMatch ? parseListItems(blockedMatch[1]) : [],
		};
	}

	// Key Decisions
	const decisionsMatch = text.match(/## Key Decisions\s*\n([\s\S]*?)(?=\n##(?!\d|#)|$)/);
	if (decisionsMatch) {
		const items = parseListItems(decisionsMatch[1]);
		result.keyDecisions = items.map((item) => {
			const colonIdx = item.indexOf(":");
			if (colonIdx > 0) {
				// Extract decision, stripping markdown bold markers
				const decision = item.slice(0, colonIdx).replace(/^[-*]?\s*/, "").replace(/[*`]/g, "").trim();
				const rationale = item.slice(colonIdx + 1).trim();
				return { decision, rationale };
			}
			return { decision: item.replace(/^[-*]?\s*/, "").trim(), rationale: "" };
		});
	}

	// Next Steps
	const stepsMatch = text.match(/## Next Steps\s*\n([\s\S]*?)(?=\n##(?!\d|#)|$)/);
	if (stepsMatch) {
		result.nextSteps = parseListItems(stepsMatch[1]);
	}

	return result;
}

/** Parse list items from a markdown list section. */
function parseListItems(text: string): string[] {
	return text
		.split("\n")
		.map((line) => line.replace(/^[\-\*]\s*(?:\[[xX ]\]\s*)?/, "").replace(/^\d+\.\s*/, "").trim())
		.filter((line) => line.length > 0 && !line.startsWith("###"));
}

// ============================================================================
// Message Serialization
// ============================================================================

/** Serialize messages to a flat text representation for summarization. */
export function serializeMessages(messages: Message[]): string {
	const parts: string[] = [];

	for (const msg of messages) {
		if (!msg) continue;
		const role = msg.role === "assistant" ? "Assistant" : msg.role === "user" ? "User" : msg.role === "system" ? "System" : msg.role;

		if (msg.role === "assistant" && msg.tool_calls && msg.tool_calls.length > 0) {
			for (const tc of msg.tool_calls) {
				const argsStr = typeof tc.arguments === "string" ? tc.arguments : JSON.stringify(tc.arguments ?? {});
				parts.push(`[Tool Call: ${tc.name}(${argsStr})]`);
			}
			if (msg.content && typeof msg.content === "string" && msg.content) {
				parts.push(`[${role}]: ${msg.content}`);
			}
		} else if (msg.role === "tool") {
			const content = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content ?? "");
			parts.push(`[Tool Result (${msg.tool_call_id ?? "unknown"})]: ${content.slice(0, 200)}${content.length > 200 ? "..." : ""}`);
		} else {
			const content = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content ?? "");
			parts.push(`[${role}]: ${content.slice(0, 500)}${content.length > 500 ? "..." : ""}`);
		}
	}

	return parts.join("\n");
}
