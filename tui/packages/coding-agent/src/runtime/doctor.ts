import { constants } from "node:fs";
import { access, stat } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { loadLogicianConfig } from "../configuration/config.ts";
import { loadSkills } from "../skills.ts";
import { getToolPath } from "../tools/shared/tools-manager.ts";

export interface DoctorReport {
	version: string;
	workspace: {
		path: string;
		present: boolean;
		readable: boolean;
		writable: boolean;
	};
	config: {
		path: string | null;
		valid: boolean;
		warnings: string[];
		error?: string;
	};
	backend: { baseUrl: string; model: string | null; probed: false };
	dependencies: { node: string; rg: boolean; fd: boolean };
	mcp: { configured: number; liveHealthChecked: false };
	skills: { roots: string[]; loaded: number; warnings: number };
	permissions: { mode: string };
	diagnostics: { postEditEnabled: boolean };
	sandbox: { enforced: false; kind: null };
}

async function canAccess(target: string, mode: number): Promise<boolean> {
	try {
		await access(target, mode);
		return true;
	} catch {
		return false;
	}
}

function skillRoots(cwd: string): string[] {
	const roots = [
		path.join(cwd, ".agents", "skills"),
		path.join(cwd, ".claude", "skills"),
		path.join(cwd, "skills"),
	];
	const home = os.homedir();
	if (home) {
		roots.push(path.join(home, ".agents", "skills"));
		roots.push(path.join(home, ".logician", "skills"));
	}
	return roots;
}

export async function buildDoctorReport(
	cwd = process.cwd(),
): Promise<DoctorReport> {
	const workspacePath = path.resolve(cwd);
	let present = false;
	try {
		present = (await stat(workspacePath)).isDirectory();
	} catch {
		present = false;
	}

	let loaded: ReturnType<typeof loadLogicianConfig> | undefined;
	let configError: string | undefined;
	try {
		loaded = loadLogicianConfig(workspacePath);
	} catch (error: unknown) {
		configError = error instanceof Error ? error.message : String(error);
	}
	const config = loaded?.config ?? {};
	const configuredMcp = {
		...(config.mcp ?? {}),
		...(config.mcpServers ?? {}),
	};
	const roots = skillRoots(workspacePath);
	const skillResult = await loadSkills(roots);
	const [rg, fd, readable, writable] = await Promise.all([
		getToolPath("rg"),
		getToolPath("fd"),
		canAccess(workspacePath, constants.R_OK),
		canAccess(workspacePath, constants.W_OK),
	]);

	return {
		version: "0.2.0",
		workspace: { path: workspacePath, present, readable, writable },
		config: {
			path: loaded?.path ?? null,
			valid: configError === undefined,
			warnings: loaded?.warnings ?? [],
			...(configError ? { error: configError } : {}),
		},
		backend: {
			baseUrl:
				process.env.LOGICIAN_LLM_URL?.trim() ||
				config.llmUrl ||
				config.baseUrl ||
				"http://127.0.0.1:8080",
			model: process.env.LOGICIAN_MODEL?.trim() || config.model || null,
			probed: false,
		},
		dependencies: {
			node: process.version,
			rg: rg !== null,
			fd: fd !== null,
		},
		mcp: {
			configured: Object.keys(configuredMcp).length,
			liveHealthChecked: false,
		},
		skills: {
			roots,
			loaded: skillResult.skills.length,
			warnings: skillResult.diagnostics.length,
		},
		permissions: { mode: config.permissionMode ?? "acceptAll" },
		diagnostics: {
			postEditEnabled: process.env.LOGICIAN_POST_EDIT_DIAGNOSTICS !== "0",
		},
		sandbox: { enforced: false, kind: null },
	};
}

export function formatDoctorReport(report: DoctorReport): string {
	const status = (ok: boolean): string => (ok ? "ok" : "unavailable");
	return [
		`Logician ${report.version}`,
		`Workspace: ${report.workspace.path} (${status(report.workspace.present && report.workspace.readable)})`,
		`Config: ${report.config.path ?? "defaults"} (${report.config.valid ? "valid" : "invalid"})`,
		`Backend: ${report.backend.baseUrl} (${report.backend.model ?? "model not set"}; not probed)`,
		`Dependencies: node ${report.dependencies.node}, rg ${status(report.dependencies.rg)}, fd ${status(report.dependencies.fd)}`,
		`MCP: ${report.mcp.configured} configured (not probed)`,
		`Skills: ${report.skills.loaded} loaded, ${report.skills.warnings} warning(s)`,
		`Permissions: ${report.permissions.mode}`,
		`Post-edit diagnostics: ${report.diagnostics.postEditEnabled ? "enabled" : "disabled"}`,
		"Sandbox: none (approval and path policy only)",
		...(report.config.error ? [`Error: ${report.config.error}`] : []),
	].join("\n");
}
