/**
 * Local live dashboard: an SSE-based HTTP server serving a static HTML
 * shell plus the session's raw JSONL log, with a `/events` stream that
 * pings on each new run so the page reloads the log without polling.
 * Self-contained — no dependency on AutoresearchSession.
 */

import { spawn } from "node:child_process";
import * as fs from "node:fs";
import { createServer, type Server, type ServerResponse } from "node:http";
import { tmpdir } from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { extractAutoresearchSessionName } from "./jsonl.ts";
import { sessionFilePath } from "./paths.ts";

export type NotifyLevel = "info" | "warning" | "error";
export type NotifyFn = (message: string, level?: NotifyLevel) => void;

const TITLE_PLACEHOLDER = "__AUTORESEARCH_TITLE__";
const LOGO_PLACEHOLDER = "__AUTORESEARCH_LOGO__";

let cachedPackageRoot: string | null = null;

function packageRoot(): string {
	if (cachedPackageRoot) return cachedPackageRoot;
	const extensionDir = fs.realpathSync(
		path.dirname(fileURLToPath(import.meta.url)),
	);
	cachedPackageRoot = path.resolve(extensionDir, "..");
	return cachedPackageRoot;
}

function templatePath(): string {
	return path.join(packageRoot(), "assets", "template.html");
}

function logoDataUrl(): string {
	const logoPath = path.join(packageRoot(), "assets", "logo.webp");
	const bytes = fs.readFileSync(logoPath);
	return `data:image/webp;base64,${bytes.toString("base64")}`;
}

let dashboardServer: Server | null = null;
let dashboardServerPort: number | null = null;
let dashboardServerWorkDir: string | null = null;
const dashboardSSEClients = new Set<ServerResponse>();

export function stopDashboardServer(): void {
	for (const client of dashboardSSEClients) {
		try {
			client.end();
		} catch {
			/* ignore */
		}
	}
	dashboardSSEClients.clear();

	if (dashboardServer) {
		try {
			dashboardServer.close();
		} catch {
			/* ignore */
		}
	}

	dashboardServer = null;
	dashboardServerPort = null;
	dashboardServerWorkDir = null;
}

function escapeHtml(text: string): string {
	return text
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;")
		.replace(/'/g, "&#39;");
}

function openInBrowser(url: string): void {
	const child =
		process.platform === "win32"
			? spawn("cmd", ["/c", "start", "", url], {
					detached: true,
					shell: true,
					stdio: "ignore",
				})
			: spawn(process.platform === "darwin" ? "open" : "xdg-open", [url], {
					detached: true,
					stdio: "ignore",
				});
	child.on("error", () => {
		/* ignore */
	});
	child.unref();
}

export function broadcastDashboardUpdate(workDir: string): void {
	if (!dashboardServer || dashboardServerWorkDir !== workDir) return;
	for (const res of dashboardSSEClients) {
		try {
			res.write("event: jsonl-updated\n");
			res.write(`data: ${Date.now()}\n\n`);
		} catch {
			dashboardSSEClients.delete(res);
		}
	}
}

async function startDashboardServer(
	workDir: string,
	dashboardHtmlPath: string,
): Promise<number> {
	return new Promise((resolve, reject) => {
		const resolvedWorkDir = path.resolve(workDir);
		const resolvedHtmlPath = path.resolve(dashboardHtmlPath);

		if (
			dashboardServer &&
			dashboardServerWorkDir === resolvedWorkDir &&
			dashboardServerPort
		) {
			resolve(dashboardServerPort);
			return;
		}

		stopDashboardServer();

		const server = createServer((req, res) => {
			const url = new URL(req.url ?? "/", "http://127.0.0.1");

			if (url.pathname === "/events") {
				res.writeHead(200, {
					"Content-Type": "text/event-stream",
					"Cache-Control": "no-cache",
					Connection: "keep-alive",
				});
				res.write("retry: 1000\n\n");
				dashboardSSEClients.add(res);
				res.on("close", () => dashboardSSEClients.delete(res));
				return;
			}

			if (url.pathname === "/") {
				fs.readFile(resolvedHtmlPath, (err, data) => {
					if (err) {
						res.writeHead(404);
						res.end();
						return;
					}
					res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
					res.end(data);
				});
				return;
			}

			if (url.pathname === "/autoresearch.jsonl") {
				const jsonlPath = sessionFilePath(resolvedWorkDir, "log");
				fs.readFile(jsonlPath, (err, data) => {
					if (err) {
						res.writeHead(404);
						res.end();
						return;
					}
					res.writeHead(200, { "Content-Type": "application/jsonl" });
					res.end(data);
				});
				return;
			}

			res.writeHead(404);
			res.end();
		});

		server.listen(0, "127.0.0.1", () => {
			const addr = server.address();
			if (!addr || typeof addr === "string") {
				reject(new Error("Failed to bind dashboard server"));
				return;
			}
			dashboardServer = server;
			dashboardServerPort = addr.port;
			dashboardServerWorkDir = resolvedWorkDir;
			resolve(addr.port);
		});

		server.on("error", reject);
	});
}

function writeDashboardFile(workDir: string): string {
	const jsonlContent = fs
		.readFileSync(sessionFilePath(workDir, "log"), "utf-8")
		.trim();
	const sessionName = extractAutoresearchSessionName(jsonlContent);
	const template = fs.readFileSync(templatePath(), "utf-8");
	const html = template
		.replace(TITLE_PLACEHOLDER, escapeHtml(sessionName))
		.replace(LOGO_PLACEHOLDER, logoDataUrl());
	const exportDir = fs.mkdtempSync(
		path.join(tmpdir(), "logician-autoresearch-dashboard-"),
	);
	const dest = path.join(exportDir, "index.html");
	fs.writeFileSync(dest, html);
	return dest;
}

export async function exportDashboard(
	notify: NotifyFn,
	workDir: string,
): Promise<void> {
	const jsonlPath = sessionFilePath(workDir, "log");
	if (!fs.existsSync(jsonlPath)) {
		notify(
			`No ${path.basename(jsonlPath)} found — run some experiments first`,
			"error",
		);
		return;
	}

	try {
		const dashboardHtmlPath = writeDashboardFile(workDir);
		const port = await startDashboardServer(workDir, dashboardHtmlPath);
		const url = `http://127.0.0.1:${port}`;
		openInBrowser(url);
		notify(`Dashboard at ${url} (live updates)`, "info");
	} catch (error) {
		notify(
			`Export failed: ${error instanceof Error ? error.message : String(error)}`,
			"error",
		);
	}
}
