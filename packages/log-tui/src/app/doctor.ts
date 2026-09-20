import { realpathSync } from "node:fs";
import { getThemePath, inspectTheme } from "../terminal/theme.ts";

declare const LOGICIAN_BUILD_REVISION: string;

export function inspectInstallation(themeName: string) {
	let palette;
	try {
		palette = { ...inspectTheme(themeName), valid: true, error: null };
	} catch (error) {
		palette = {
			name: themeName,
			path: getThemePath(themeName),
			valid: false,
			error: error instanceof Error ? error.message : String(error),
		};
	}
	return {
		executable: realpathSync(process.execPath),
		entrypoint: process.argv[1] ?? null,
		revision:
			typeof LOGICIAN_BUILD_REVISION === "string"
				? LOGICIAN_BUILD_REVISION
				: null,
		platform: `${process.platform}-${process.arch}`,
		bun: process.versions.bun ?? null,
		terminal: {
			term: process.env.TERM ?? null,
			colorterm: process.env.COLORTERM ?? null,
			tty: Boolean(process.stdout.isTTY),
		},
		theme: palette,
	};
}

export function formatInstallation(
	report: ReturnType<typeof inspectInstallation>,
): string {
	return [
		`Executable: ${report.executable}`,
		`Entrypoint: ${report.entrypoint ?? "unknown"}`,
		`Build revision: ${report.revision ?? "source run (not compiled)"}`,
		`Runtime: Bun ${report.bun ?? "unavailable"} (${report.platform})`,
		`Terminal: TERM=${report.terminal.term ?? "unset"}, COLORTERM=${report.terminal.colorterm ?? "unset"}`,
		`Theme: ${report.theme.name} (${report.theme.path})`,
		report.theme.error === null
			? `Color mode: ${report.theme.mode}; input border ANSI: ${JSON.stringify(report.theme.inputBarBorder)}`
			: `Theme error: ${report.theme.error}`,
	].join("\n");
}
