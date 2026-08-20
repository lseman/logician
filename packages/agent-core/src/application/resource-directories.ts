import os from "node:os";
import path from "node:path";
import { runPluginBackend } from "../infrastructure/tools/index.ts";

export async function getSkillsDirs(cwd: string): Promise<string[]> {
	const dirs: string[] = [];
	try {
		const registry = await runPluginBackend("list", []);
		for (const plugin of registry.plugins || []) {
			const installPath = String(plugin.install_path || "");
			if (plugin.enabled !== false && plugin.on_disk !== false && installPath) {
				dirs.push(path.join(installPath, "skills"));
			}
		}
	} catch (_error: unknown) {
		// Plugin discovery is optional during early startup.
	}

	dirs.push(path.join(os.homedir(), ".agents", "skills"));
	dirs.push(...getProjectSkillDirs(cwd));
	return Array.from(new Set(dirs));
}

export function getProjectSkillDirs(cwd: string): string[] {
	return Array.from(new Set(projectResourceDirs(cwd, "skills")));
}

export function getProjectPromptDirs(cwd: string): string[] {
	return Array.from(
		new Set([
			path.join(os.homedir(), ".logician", "prompts"),
			...projectResourceDirs(cwd, "prompts"),
		]),
	);
}

function projectResourceDirs(cwd: string, resource: string): string[] {
	const dirs: string[] = [];
	let current = path.resolve(cwd);
	while (true) {
		dirs.push(path.join(current, ".logician", resource));
		dirs.push(path.join(current, resource));
		const parent = path.dirname(current);
		if (parent === current) break;
		current = parent;
	}
	return dirs;
}
