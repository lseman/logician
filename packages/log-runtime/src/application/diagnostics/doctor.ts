import { constants, existsSync } from "node:fs";
import { access, stat } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { loadSkills } from "../../capabilities/skills/loader.ts";
import { getToolPath } from "../../infrastructure/tools/external-tools.ts";
import { resolveRuntimeConfig } from "../configuration/runtime-config.ts";

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
	sandbox: {
		enforced: false;
		kind: "none" | "bubblewrap";
		bwrapAvailable: boolean;
		bwrapPath: string | null;
	};
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
	environment: NodeJS.ProcessEnv = process.env,
): Promise<DoctorReport> {
	const workspacePath = path.resolve(cwd);
	let present = false;
	try {
		present = (await stat(workspacePath)).isDirectory();
	} catch {
		present = false;
	}

	let loaded: ReturnType<typeof resolveRuntimeConfig> | undefined;
	let configError: string | undefined;
	try {
		loaded = resolveRuntimeConfig(workspacePath, environment, {
			loadProjectConfig: true,
		});
	} catch (error: unknown) {
		configError = error instanceof Error ? error.message : String(error);
	}
	const config = loaded?.source ?? {};
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

	// Detect bubblewrap
	const pathEnv = process.env.PATH ?? "";
	const pathEntries = pathEnv.split(path.delimiter);
	let bwrapPath: string | null = null;
	for (const dir of pathEntries) {
		const fullPath = path.join(dir, "bwrap");
		if (existsSync(fullPath)) {
			bwrapPath = fullPath;
			break;
		}
	}
	let bwrapAvailable = false;
	if (bwrapPath && process.platform === "linux") {
		try {
			const { spawnSync } = await import("node:child_process");
			const result = spawnSync(bwrapPath, ["--version"], {
				timeout: 5000,
				stdio: ["ignore", "pipe", "pipe"],
			});
			if (result.status === 0 && result.stdout) {
				const versionStr = result.stdout.toString().trim();
				const match = versionStr.match(/bubblewrap\s+([\d.]+)/);
				if (match) {
					const version = match[1].split(".").map(Number);
					bwrapAvailable =
						version.length >= 3 &&
						(version[0] > 0 ||
							(version[0] === 0 && version[1] > 4) ||
							(version[0] === 0 && version[1] === 4 && version[2] >= 1));
				}
			}
		} catch {
			// best effort
		}
	}

	return {
		version: "0.2.0",
		workspace: { path: workspacePath, present, readable, writable },
		config: {
			path: loaded?.configPath ?? null,
			valid: configError === undefined,
			warnings: loaded?.warnings ?? [],
			...(configError ? { error: configError } : {}),
		},
		backend: {
			baseUrl: loaded?.bridge.baseUrl ?? "http://127.0.0.1:8080",
			model: loaded?.bridge.model || null,
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
		permissions: {
			mode: loaded?.bridge.permissionMode ?? "acceptEdits",
		},
		diagnostics: {
			postEditEnabled: environment.LOGICIAN_POST_EDIT_DIAGNOSTICS !== "0",
		},
		sandbox: {
			enforced: false,
			kind: bwrapAvailable ? "bubblewrap" : "none",
			bwrapAvailable,
			bwrapPath,
		},
	};
}

export function formatDoctorReport(report: DoctorReport): string {
	const W = process.stdout.columns || 80;
	const sep = "\u2501".repeat(W);
	const thin = "\u2500".repeat(W);
	const pad = (text: string, target: number): string =>
		text + " ".repeat(Math.max(0, target - text.length));

	const icon = (ok: boolean): string => (ok ? "\u2713" : "\u2717");
	const tag = (ok: boolean): string =>
		ok ? "\x1b[32mOK\x1b[0m" : "\x1b[31mFAIL\x1b[0m";
	const warnTag = (w: number): string =>
		w > 0 ? `\x1b[33m${w} warning(s)\x1b[0m` : "\x1b[32mnone\x1b[0m";
	const dim = (t: string): string => `\x1b[2m${t}\x1b[0m`;

	// Header
	const lines: string[] = [
		`\x1b[1m${sep}\x1b[0m`,
		"\x1b[1m  Logician Doctor \u2014 System Diagnostics\x1b[0m",
		`\x1b[1m${sep}\x1b[0m`,
		"",
	];

	// Version
	lines.push(`  \x1b[1mVersion\x1b[0m  ${dim(`v${report.version}`)}`);

	// Workspace
	const wsOk = report.workspace.present && report.workspace.readable;
	lines.push(
		"",
		"  \x1b[1mWorkspace\x1b[0m" +
			`  ${icon(wsOk)} ${tag(wsOk)}` +
			`  ${report.workspace.path}`,
	);
	if (!report.workspace.present) {
		lines.push(`    ${dim("Directory does not exist")}`);
	}
	if (report.workspace.present && !report.workspace.readable) {
		lines.push(`    ${dim("Not readable")}`);
	}
	if (report.workspace.present && !report.workspace.writable) {
		lines.push(`    ${dim("Not writable")}`);
	}

	// Config
	const cfgOk = report.config.valid;
	lines.push(
		"",
		"  \x1b[1mConfiguration\x1b[0m" +
			`  ${icon(cfgOk)} ${tag(cfgOk)}` +
			`  ${report.config.path ?? dim("defaults only")}`,
	);
	if (report.config.error) {
		lines.push(`    ${dim("Error: ") + report.config.error}`);
	}
	for (const w of report.config.warnings) {
		lines.push(`    ${dim("Warning: ") + w}`);
	}

	// Backend
	lines.push(
		"",
		"  \x1b[1mBackend\x1b[0m" +
			`  ${dim("not probed")}` +
			`  ${report.backend.baseUrl}`,
	);
	lines.push(
		`    ${pad("model:", 12)}${report.backend.model ?? dim("not set")}` +
			`  ${dim("(will be verified at runtime)")}`,
	);

	// Dependencies
	const deps = [
		{ name: "node", ver: report.dependencies.node, ok: true },
		{
			name: "rg",
			ver: report.dependencies.rg ? "found" : "missing",
			ok: report.dependencies.rg,
		},
		{
			name: "fd",
			ver: report.dependencies.fd ? "found" : "missing",
			ok: report.dependencies.fd,
		},
	];
	lines.push(
		"",
		"  \x1b[1mDependencies\x1b[0m" +
			`  ${deps.every(d => d.ok) ? tag(true) : tag(false)}`,
	);
	for (const d of deps) {
		lines.push(`    ${pad(`${d.name}:`, 12)}${d.ver}` + `  ${icon(d.ok)}`);
	}

	// MCP
	lines.push(
		"",
		"  \x1b[1mMCP Servers\x1b[0m" +
			`  ${dim("not probed")}` +
			`  ${report.mcp.configured} configured`,
	);

	// Skills
	lines.push(
		"",
		"  \x1b[1mSkills\x1b[0m" +
			`  ${icon(report.skills.warnings === 0)} ${warnTag(report.skills.warnings)}` +
			`  ${report.skills.loaded} loaded`,
	);
	if (report.skills.roots.length > 0) {
		lines.push(`    ${dim("Search paths:")}`);
		for (const r of report.skills.roots) {
			lines.push(`      ${dim(r)}`);
		}
	}

	// Policies
	lines.push(
		"",
		"  \x1b[1mPolicies\x1b[0m" +
			`  ${tag(report.permissions.mode !== "denyAll")}`,
	);
	lines.push(`    ${pad("permission mode:", 18)}${report.permissions.mode}`);
	lines.push(
		`    ${pad("post-edit diagnostics:", 23)}${report.diagnostics.postEditEnabled ? "enabled" : "disabled"}`,
	);
	const sbOk = report.sandbox.bwrapAvailable;
	const sbKind = report.sandbox.kind;
	const sbExtra =
		sbKind === "bubblewrap" && report.sandbox.bwrapPath
			? ` (${report.sandbox.bwrapPath})`
			: "";
	const sbInfo =
		sbKind === "bubblewrap"
			? `bubblewrap${sbExtra}`
			: "none (approval & path policy)";
	lines.push(
		`    ${pad("sandbox:", 23)}${sbInfo}${sbKind === "bubblewrap" ? `  ${icon(sbOk)}` : ""}`,
	);

	// Footer
	lines.push("", `\x1b[1m${thin}\x1b[0m`);

	return lines.join("\n");
}
