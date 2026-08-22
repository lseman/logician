/** Zero-LLM observation compression: infers a type and builds a compact
 * title/narrative/facts/concepts summary from a raw hook payload using
 * pattern matching alone, no closure/db dependency. */

import type {
	CompressedObservation,
	ObservationType,
	RawObservation,
} from "../types.js";

function inferType(payload: unknown, hookType: string): ObservationType {
	// post_tool_failure is always an error, regardless of payload
	if (hookType === "post_tool_failure") return "error";
	if (hookType === "prompt_submit") return "conversation";
	if (hookType === "notification") return "notification";

	if (typeof payload === "object" && payload !== null) {
		const d = payload as Record<string, unknown>;
		const name = ((d as any).tool_name || (d as any).name || "") as string;
		const lower = name.toLowerCase();

		if (lower.includes("read") || lower.includes("cat")) return "file_read";
		if (
			lower.includes("write") ||
			lower.includes("append") ||
			lower.includes("overwrite")
		)
			return "file_write";
		if (lower.includes("edit")) return "file_edit";
		if (
			lower.includes("bash") ||
			lower.includes("shell") ||
			lower.includes("exec") ||
			lower.includes("run")
		)
			return "command_run";
		if (lower.includes("search") || lower.includes("grep")) return "search";
		if (
			lower.includes("fetch") ||
			lower.includes("curl") ||
			lower.includes("http")
		)
			return "web_fetch";
	}
	return "other";
}

export function buildSyntheticCompression(raw: RawObservation): CompressedObservation {
	const { id, sessionId, timestamp, hookType, raw: data } = raw;
	let type = inferType(data, hookType);

	let title = "Observation";
	let narrative = "";
	const facts: string[] = [];
	const concepts: string[] = [];
	const files: string[] = [];
	let importance = 5;

	if (typeof data === "object" && data !== null) {
		const d = data as Record<string, unknown>;
		const toolName = ((d as any).tool_name ||
			(d as any).name ||
			hookType) as string;

		// Extract file references
		const filePatterns = [
			/(?:file_path|path|file|filename|target_file|output_file)["']?\s*[:=]\s*["']?([^\s"'`,]+)/gi,
			/(?:read|write|edit|open)\s+["']?([^\s"'`,]+\.[\w.]+)/gi,
		];
		for (const pattern of filePatterns) {
			let match;
			const str = JSON.stringify(d);
			while ((match = pattern.exec(str)) !== null) {
				if (match[1]?.includes("/")) files.push(match[1].slice(0, 300));
			}
		}

		// Extract concepts from keywords
		const conceptKeywords = [
			"error",
			"bug",
			"fix",
			"crash",
			"panic",
			"timeout",
			"retry",
			"config",
			"setting",
			"env",
			"environment",
			"auth",
			"permission",
			"access",
			"token",
			"database",
			"schema",
			"migration",
			"query",
			"connection",
			"api",
			"endpoint",
			"route",
			"middleware",
			"login",
			"auth",
			"test",
			"unit",
			"integration",
			"mock",
			"stub",
			"build",
			"deploy",
			"pipeline",
			"ci",
			"cd",
			"refactor",
			"optimize",
			"performance",
			"memory",
			"cpu",
			"security",
			"vulnerability",
			"sanitize",
			"escape",
		];
		const lowerStr = JSON.stringify(d).toLowerCase();
		for (const kw of conceptKeywords) {
			if (lowerStr.includes(kw)) concepts.push(kw);
		}

		// Build title and narrative
		const output =
			(d as any).tool_output || (d as any).output || (d as any).result || "";
		const error = (d as any).error || "";
		const input =
			(d as any).tool_input || (d as any).input || (d as any).arguments || {};
		const inputStr =
			typeof input === "string" ? input : JSON.stringify(input).slice(0, 500);

		// For failures, error field takes precedence
		const effectiveOutput = error || output;

		if (typeof effectiveOutput === "string" && effectiveOutput.length > 0) {
			const truncated = effectiveOutput.slice(0, 1000);
			title = truncate(truncated, 80) || toolName;
			narrative = `${toolName}: ${truncated.slice(0, 300)}`;
			facts.push(truncated.slice(0, 500));
		} else if (typeof inputStr === "string" && inputStr.length > 0) {
			title = `${toolName}: ${inputStr.slice(0, 80)}`;
			narrative = `${toolName}(input)`;
			facts.push(inputStr.slice(0, 500));
		} else {
			title = `${toolName}`;
			narrative = `${hookType} → ${toolName}`;
		}

		// Boost importance for errors
		const outputStr = typeof output === "string" ? output.toLowerCase() : "";
		const errorStr = typeof error === "string" ? error.toLowerCase() : "";
		const combinedErr = `${outputStr} ${errorStr}`;
		if (type === "error") {
			importance = 8;
		} else if (type === "file_write" || type === "file_edit") {
			importance = 7;
		} else if (
			type === "file_read" ||
			type === "search" ||
			type === "web_fetch"
		) {
			importance = 3;
		} else if (type === "command_run") {
			if (
				/\b(?:error|fail(?:ed|ure)?|panic|crash|exception|timeout|refused)\b/.test(
					combinedErr,
				)
			) {
				importance = 8;
			} else {
				importance = /\b(?:pass(?:ed)?|success|built|compiled|deployed)\b/.test(
					outputStr,
				)
					? 6
					: 4;
			}
		} else if (type === "other") {
			importance = 4;
		}

		// Add files to facts
		const uniqueFiles = [...new Set(files)];
		if (uniqueFiles.length > 0) {
			facts.push(`Files: ${uniqueFiles.slice(0, 5).join(", ")}`);
		}

		// Add concepts to narrative
		if (concepts.length > 3) {
			facts.push(`Concepts: ${concepts.slice(0, 5).join(", ")}`);
		}

		if (hookType === "prompt_submit") {
			const prompt =
				typeof d.prompt === "string"
					? d.prompt.trim()
					: typeof d.userPrompt === "string"
						? d.userPrompt.trim()
						: raw.userPrompt?.trim() || "";
			if (prompt) {
				title = `User request: ${truncate(prompt, 100)}`;
				narrative = prompt.slice(0, 2000);
				facts.length = 0;
				facts.push(prompt.slice(0, 1000));
				if (/^(?:hi|hello|hey|thanks|thank you|ok|okay)[!. ]*$/i.test(prompt)) {
					importance = 1;
				} else if (
					/\b(?:decide|decision|must|requirement|architecture|security|breaking|never|always)\b/i.test(
						prompt,
					)
				) {
					type = "decision";
					importance = 7;
				} else if (prompt.length < 20) {
					importance = 2;
				} else {
					importance = 5;
				}
			}
		}
	} else if (typeof data === "string") {
		title = truncate(data, 80) || hookType;
		narrative = data.slice(0, 500);
		facts.push(data.slice(0, 500));
	}

	// Extract concepts from narrative
	const extractedConcepts = new Set(concepts);
	const conceptPatterns = [
		/#[\w]+/g, // hashtags
		/\b([A-Z][a-z]+(?:[A-Z][a-z]+)*\w*)\b/g, // camelCase/PascalCase words
	];
	for (const pattern of conceptPatterns) {
		let match;
		const str = `${narrative} ${facts.join(" ")}`;
		while ((match = pattern.exec(str)) !== null) {
			const word = match[0].replace(/#/g, "");
			if (word.length >= 3 && !extractedConcepts.has(word)) {
				extractedConcepts.add(word);
			}
			if (extractedConcepts.size >= 10) break;
		}
	}

	const uniqueFiles = [...new Set(files)];

	return {
		id,
		sessionId,
		timestamp,
		type,
		title: title.slice(0, 200),
		narrative: narrative.slice(0, 2000),
		facts,
		concepts: [...extractedConcepts].slice(0, 10),
		files: uniqueFiles,
		importance: Math.max(1, Math.min(10, importance)),
		consolidated: false,
		provenance: {
			source: "deterministic",
			trust: "trusted_local",
			extractorVersion: "synthetic-compression/2",
			schemaVersion: 1,
		},
	};
}

function truncate(text: string, maxLen: number): string {
	if (text.length <= maxLen) return text;
	return `${text.slice(0, maxLen - 3)}...`;
}
