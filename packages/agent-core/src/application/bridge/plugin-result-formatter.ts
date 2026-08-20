// ── Plugin result formatter ──────────────────────────────────────────────
// Renders a PluginCommandResult (from the /plugins command) as markdown for
// the transcript. Pure formatting, no bridge state — extracted from
// AgentCoreBridge.

import type { PluginCommandResult } from "../../adapters/claude-code/plugin-manager.ts";
import { tableRow } from "../../tui-utils.ts";

export function formatPluginResult(
	action: string,
	result: PluginCommandResult,
): string {
	if (result.status === "error") {
		return `/plugins failed: ${result.message || "unknown error"}`;
	}

	if (action === "list") {
		const plugins = result.plugins || [];
		const hooks = result.session_start_hooks || {};
		const lines = [
			"# Installed plugins",
			`Registry: ${result.plugins_dir || "unknown"}`,
		];
		if (!plugins.length) {
			lines.push("", "No plugins installed.");
			return lines.join("\n");
		}
		lines.push("", "| Plugin | Version | State | Hooks | Path |");
		lines.push("|--------|---------|-------|-------|------|");
		for (const plugin of plugins) {
			const id = String(plugin.plugin_id || plugin.name || "");
			const hookCount = hooks[id] || 0;
			const state = plugin.enabled ? "enabled" : "disabled";
			const onDisk = plugin.on_disk === false ? " missing" : "";
			lines.push(
				tableRow([
					id,
					String(plugin.version || ""),
					`${state}${onDisk}`,
					hookCount ? `SessionStart x${hookCount}` : "-",
					String(plugin.install_path || ""),
				]),
			);
		}
		return lines.join("\n");
	}

	if (action === "hooks") {
		const hooks = result.hooks || [];
		const source = String(result.source || "startup");
		const lines = [
			"# Plugin SessionStart hooks",
			`Source: ${source}`,
			`Registry: ${result.plugins_dir || "unknown"}`,
		];
		if (!hooks.length) {
			lines.push("", "No enabled SessionStart hooks matched this source.");
			return lines.join("\n");
		}
		lines.push("", "| Plugin | Matcher | Commands |");
		lines.push("|--------|---------|----------|");
		for (const hook of hooks) {
			const commands = Array.isArray(hook.commands)
				? hook.commands
						.map(
							(cmd: { type?: string; command?: string }) =>
								`${cmd.type}${cmd.command ? `: ${cmd.command}` : ""}`,
						)
						.join("<br>")
				: "";
			lines.push(
				tableRow([
					String(hook.plugin_id || hook.plugin_name || ""),
					String(hook.matcher || "*"),
					commands || "-",
				]),
			);
		}
		return lines.join("\n");
	}

	if (action === "run-hooks") {
		const lines = [
			"# Plugin hooks executed",
			`Source: ${result.source || "startup"}`,
			`Hooks: ${result.hook_count || 0}`,
			`Contexts added: ${(result.additional_contexts || []).length}`,
		];
		const errors = result.errors || [];
		if (errors.length) {
			lines.push("", "Errors:");
			lines.push(...errors.map(err => `- ${err}`));
		}
		if ((result.additional_contexts || []).length) {
			lines.push("", "Hook context has been applied to future agent turns.");
		}
		return lines.join("\n");
	}

	if (action === "update" && Array.isArray(result.updates)) {
		const lines = ["# Plugin updates"];
		for (const update of result.updates) {
			lines.push(
				`- ${update.message || update.status || JSON.stringify(update)}`,
			);
		}
		return lines.join("\n");
	}

	if (action === "deps") {
		const issues = result.issues || [];
		if (!issues.length) return "All plugin dependencies OK.";
		const lines = ["# Plugin dependency issues"];
		for (const issue of issues) {
			lines.push(
				`- ${issue.plugin_id || "plugin"}: ${issue.status || "issue"}`,
			);
			if (Array.isArray(issue.missing) && issue.missing.length) {
				lines.push(`  Missing: ${issue.missing.join(", ")}`);
			}
		}
		return lines.join("\n");
	}

	return String(
		result.message || result.status || JSON.stringify(result, null, 2),
	);
}
