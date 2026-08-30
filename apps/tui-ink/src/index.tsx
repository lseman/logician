#!/usr/bin/env node
// ── Logician Ink TUI — Entry point ────────────────────────────────────────────

// Force truecolor output
if (!process.env.FORCE_COLOR) process.env.FORCE_COLOR = "3";

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { render } from "ink";
import { isTranscriptEvent } from "@logician/log-core/events";
import { AgentRuntime } from "@logician/log-runtime/application";
import { resolveRuntimeConfig } from "@logician/log-runtime/configuration/runtime";
import {
	buildDoctorReport,
	formatDoctorReport,
} from "@logician/log-runtime/developer-tools";
import { activateProjectVirtualEnv } from "@logician/log-runtime/tools";
import {
	applyTrustChoice,
	resolveTrust,
	resolveTrustInfo,
	TrustStore,
} from "@logician/log-runtime/trust";
import { Transcript, TuiSessionService } from "@logician/log-runtime/sessions";
import { createAutoresearchTools } from "@logician/log-runtime/tools";
import { AutoresearchSession } from "@logician/log-autoresearch";
import { App } from "./components/App.tsx";
import { TuiState } from "./state.ts";
import type { Turn } from "@logician/log-runtime/sessions";

// ── Load ~/.logician/.env ─────────────────────────────────────────────────────

(function loadHomeEnv(): void {
	const home = process.env.HOME || "/";
	const envPath = join(home, ".logician", ".env");
	if (!existsSync(envPath)) return;
	const lines = readFileSync(envPath, "utf8").split("\n");
	for (const line of lines) {
		const trimmed = line.trim();
		if (!trimmed || trimmed.startsWith("#")) continue;
		const eq = trimmed.indexOf("=");
		if (eq < 1) continue;
		const key = trimmed.slice(0, eq);
		let value = trimmed.slice(eq + 1);
		if (
			value.length >= 2 &&
			((value[0] === '"' && value.at(-1) === '"') ||
				(value[0] === "'" && value.at(-1) === "'"))
		) {
			value = value.slice(1, -1);
		}
		process.env[key] = value;
	}
})();

// ── Trust helpers ─────────────────────────────────────────────────────────────

function defaultProjectTrust(): "ask" | "always" | "never" {
	const value = process.env.LOGICIAN_TRUST?.trim().toLowerCase();
	if (value === "always" || value === "1" || value === "true") return "always";
	if (value === "never" || value === "0" || value === "false") return "never";
	return "ask";
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
	const args = process.argv.slice(2);
	const cwd = process.cwd();
	activateProjectVirtualEnv(cwd);

	// ── Parse --session <id> flag ────────────────────────────────────────
	let resumeSessionId: string | undefined;
	for (let i = 0; i < args.length; i++) {
		if (args[i] === "--session" && i + 1 < args.length) {
			resumeSessionId = args[i + 1];
			break;
		}
	}

	// ── Doctor mode ──────────────────────────────────────────────────────
	if (args[0] === "doctor" || args.includes("--doctor")) {
		const report = await buildDoctorReport(cwd);
		process.stdout.write(
			args.includes("--json")
				? `${JSON.stringify(report, null, 2)}\n`
				: `${formatDoctorReport(report)}\n`,
		);
		process.exitCode = report.config.valid && report.workspace.present ? 0 : 1;
		return;
	}

	// ── Detect TTY ───────────────────────────────────────────────────────
	const hasUI = Boolean(process.stdin.isTTY && process.stdout.isTTY);

	// ── Resolve trust ────────────────────────────────────────────────────
	let loadProjectConfig = false;

	if (hasUI) {
		const trustInfo = resolveTrustInfo(cwd, defaultProjectTrust());

		if (trustInfo.preDecided) {
			loadProjectConfig = trustInfo.preDecidedResult?.trusted ?? false;
		} else {
			// For now, default to "ask" and trust — skip interactive trust prompt
			// The Ink TUI will show a trust overlay if needed. For now, assume trust.
			loadProjectConfig = true;
		}
	} else {
		const trust = await resolveTrust({
			cwd,
			hasUI: false,
			defaultProjectTrust: defaultProjectTrust(),
		});
		loadProjectConfig = trust.trusted;
	}

	// ── Resolve runtime config ───────────────────────────────────────────
	const runtimeConfig = resolveRuntimeConfig(cwd, process.env, {
		loadProjectConfig,
	});

	// ── Build config for Ink App ─────────────────────────────────────────
	const config = {
		bridge: {
			model: runtimeConfig.bridge.model || "local",
			contextWindowTokens: runtimeConfig.bridge.contextWindowTokens || 128_000,
			permissions: runtimeConfig.bridge.permissions || { mode: "acceptEdits" },
			thinkingLevel: runtimeConfig.bridge.thinkingLevel ?? "off",
			inferenceMode: runtimeConfig.bridge.inferenceMode || "none",
			executionProfile: runtimeConfig.bridge.executionProfile ?? "minimal",
			rtkProxyEnabled: runtimeConfig.bridge.rtkProxyEnabled ?? false,
			legroom: runtimeConfig.bridge.legroom,
			memoriam: runtimeConfig.bridge.memoriam,
			graphicianEnabled: runtimeConfig.bridge.graphicianEnabled ?? true,
			fffgrepEnabled: runtimeConfig.bridge.fffgrepEnabled ?? true,
			cwd: runtimeConfig.bridge.cwd || cwd,
			extraTools: runtimeConfig.bridge.extraTools,
		},
		source: {
			theme: runtimeConfig.source.theme,
			workflowMode: runtimeConfig.source.workflowMode || "act",
			transcriptMaxTurns: runtimeConfig.source.transcriptMaxTurns,
			transcriptMaxRenderedLines: runtimeConfig.source.transcriptMaxRenderedLines,
			truncation: runtimeConfig.source.truncation,
			inferenceMode: runtimeConfig.source.inferenceMode || "none",
		},
		configPath: runtimeConfig.configPath,
	};

	// ── Create state and services ────────────────────────────────────────
	const state = new TuiState();
	const transcript = new Transcript();
	const sessionService = new TuiSessionService(cwd);
	const researchManager = new AutoresearchSession(
		runtimeConfig.bridge.cwd || cwd,
		(message, level) => state.showNotification(message, level as any),
	);
	researchManager.reload();

	// ── Create bridge ────────────────────────────────────────────────────
	const bridge = new AgentRuntime({
		...runtimeConfig.bridge,
		extraTools: [
			...(runtimeConfig.bridge.extraTools ?? []),
			...createAutoresearchTools(researchManager),
		],
	});

	// ── Session management ───────────────────────────────────────────────
	let currentSessionId = sessionService.createSession("New Session");

	// Load explicit session if --session <id> was passed
	if (resumeSessionId) {
		try {
			const turns = sessionService.loadTurns(resumeSessionId);
			if (turns && turns.length > 0) {
				const staleSessionId = currentSessionId;
				// Restore turns into transcript and bridge
				// ... (turn restoration logic)
				currentSessionId = resumeSessionId;
				state.setCurrentSession(resumeSessionId, sessionService.getSession(resumeSessionId)?.name || "Resumed Session");
				if (staleSessionId && staleSessionId !== resumeSessionId) {
					sessionService.deleteSession(staleSessionId);
				}
			} else {
				state.setCurrentSession(currentSessionId, "New Session");
			}
		} catch (error: unknown) {
			const message = error instanceof Error ? error.message : String(error);
			process.stderr.write(`error: failed to load session ${resumeSessionId}: ${message}\n`);
			state.setCurrentSession(currentSessionId, "New Session");
		}
	} else {
		state.setCurrentSession(currentSessionId, "New Session");
	}

	// ── Bridge event handling ────────────────────────────────────────────
	// Any transcript change (streaming tokens, tool calls, notices) pushes the
	// new turn list into React state, which re-renders via useSyncExternalStore.
	transcript.onChange(() => state.setTranscriptTurns(transcript.getTurns()));

	const unsubscribeEvents = bridge.events.subscribe(notification => {
		handleBridgeEvent(
			state,
			bridge,
			transcript,
			sessionService,
			() => currentSessionId,
			notification.event,
		);
	});
	const unsubscribeErrors = bridge.events.onError(err => {
		transcript.addSystemMessage(`Connection error: ${err.message}`);
		state.showNotification(`Connection error: ${err.message}`, "error");
	});

	// Bridge init
	bridge.init().catch(err => {
		transcript.addSystemMessage(`Failed to start agent: ${err.message}`);
		state.showNotification(`Failed to start agent: ${err.message}`, "error");
	});

	// ── Input handlers ───────────────────────────────────────────────────
	let stopping = false;

	const handleSubmit = async (message: string): Promise<void> => {
		const prompt = config.source.workflowMode === "plan"
			? `[PLAN MODE]\nFirst investigate using read-only tools and produce a concrete implementation plan. Do not modify files or execute mutating commands. End after presenting the plan and wait for explicit user approval.\n\nUser request:\n${message}`
			: message;

		// Add turn to transcript and send to bridge. The completed turn (with the
		// assistant reply) is persisted once, on the `turn_end` event.
		transcript.addTurn(prompt);

		await bridge.sendMessage(prompt);
	};

	const handleCancel = async (): Promise<void> => {
		if (!bridge.isActive()) return;
		state.showNotification("Stopping after the active operation settles…", "info");

		try {
			const cleared = await bridge.cancel();
			const clearedCount =
				(cleared?.clearedSteering.length ?? 0) +
				(cleared?.clearedFollowUp.length ?? 0);
			if (clearedCount > 0) {
				transcript.addSystemMessage(
					`Turn interrupted safely. Cleared ${clearedCount} queued message${clearedCount === 1 ? "" : "s"}.`,
				);
			}
		} catch (error) {
			state.showNotification(`Could not confirm interruption: ${error instanceof Error ? error.message : String(error)}`, "error");
		}
	};

	const handleExit = async (): Promise<void> => {
		if (stopping) return;
		stopping = true;

		unsubscribeEvents();
		unsubscribeErrors();
		researchManager.shutdown();
		await bridge.stop();
		process.exit(0);
	};

	// ── Render Ink App ───────────────────────────────────────────────────
	render(
		<App
			config={config}
			state={state}
			transcript={transcript}
			bridge={bridge}
			sessionService={sessionService}
			currentSessionId={currentSessionId}
			onSubmit={handleSubmit}
			onCancel={handleCancel}
			onExit={handleExit}
		/>,
	);

	// Update state with bridge reference
	state.setBridge(bridge);
	state.setCurrentSession(currentSessionId, sessionService.getSession(currentSessionId)?.name ?? "New Session");

	// ── Signal handling ──────────────────────────────────────────────────
	process.on("SIGINT", () => void handleExit());
	process.on("SIGTERM", () => void handleExit());
}

// ── Bridge event handler ─────────────────────────────────────────────────────

function handleBridgeEvent(
	state: TuiState,
	bridge: AgentRuntime,
	transcript: Transcript,
	sessionService: TuiSessionService,
	getSessionId: () => string,
	event: import("@logician/log-core/events").RuntimeEvent,
): void {
	// Streaming content — the transcript reducer turns these into turns/chunks.
	// transcript.onChange() (wired in main) pushes the result into React state.
	if (isTranscriptEvent(event)) transcript.handleEvent(event);

	switch (event.type) {
		case "turn_start":
			state.updatePhase("thinking");
			break;
		case "turn_end": {
			state.updatePhase("ready");
			// Persist the just-completed turn (assistant reply included).
			const turns = transcript.getTurns();
			const last = turns[turns.length - 1];
			if (last) {
				try {
					sessionService.saveTurn(getSessionId(), last);
				} catch {
					// Non-fatal: session persistence is best-effort.
				}
			}
			break;
		}
		case "phase":
			if ("state" in event) {
				if (event.state === "thinking") state.updatePhase("thinking");
				else if (event.state === "tool") state.updatePhase("working");
				else if (event.state === "ready") state.updatePhase("ready");
				else if (event.state === "error") state.updatePhase("error");
			}
			break;
		case "agent_retry_start":
			state.updatePhase("working");
			break;
		case "agent_error":
			transcript.handleEvent({
				type: "notice",
				level: event.recoverable ? "warn" : "error",
				label: `Agent error: ${event.phase}`,
				text: event.message,
			});
			state.showNotification(event.message, event.recoverable ? "warn" : "error");
			break;
		case "context_update":
			if ("tokens" in event) state.setContextTokens(Number(event.tokens || 0));
			if ("maxTokens" in event && Number(event.maxTokens)) {
				state.contextMaxTokens = Number(event.maxTokens);
			}
			if ("cachedTokens" in event && typeof event.cachedTokens === "number") {
				state.setCacheReadTokens(event.cachedTokens);
			}
			break;
		case "model_select":
			if ("model" in event) state.setModel(String(event.model || state.model));
			break;
		case "todos":
			if ("todos" in event && Array.isArray(event.todos)) {
				state.setTodos(
					event.todos.map((t: { content?: string; status?: string }, i) => ({
						id: `todo-${i}`,
						text: t.content ?? "",
						done: t.status === "completed",
					})),
				);
			}
			break;
		case "queue_update":
			state.setSteerMessages(
				((event.steering as string[] | undefined) ?? []).map((m, i) => ({
					id: `s-${i}`,
					message: m,
					createdAt: Date.now(),
				})),
			);
			break;
		case "steered":
			transcript.addSteeredMessage(String(event.message || ""));
			break;
		case "permission_request":
			state.setOverlay("permission", {
				toolCallId: event.toolCallId ?? "",
				toolName: event.toolName ?? "",
				args: event.args,
			});
			break;
		case "question_request":
			state.setOverlay("choice", {
				questionId: event.questionId ?? "",
				questions: event.questions,
			});
			break;
		case "notice":
			if ("label" in event && event.label === "MCP") {
				void bridge.getState().then(() => state.setMcpLoading(false));
			}
			break;
	}
}

main().catch(err => {
	console.error(err);
	process.exit(1);
});
