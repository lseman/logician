// ── @logician/memory — HTTP + WebSocket Server (Memory Viewer Dashboard) ─────

import type { MemoryStore, ObservationType } from "../types.js";

interface DashboardStats {
	workspace: string;
	sessions: number;
	memories: number;
	observations: number;
	observationsToday: number;
	memoriesByType: Record<string, number>;
	sessionsByStatus: Record<string, number>;
}

interface HealthStats {
	rss: number;
	heapUsed: number;
	heapTotal: number;
	eventLoopLag: number;
	uptime: number;
}

export interface ViewerOptions {
	port?: number;
	host?: string;
	store: MemoryStore;
	secret?: string;
}

let boundPort: number | null = null;

export function getBoundViewerPort(): number | null {
	return boundPort;
}

export function startViewerServer(opts: ViewerOptions): {
	server: { port: number; stop: (force?: boolean) => void };
	stop: (force?: boolean) => void;
} {
	const port = opts.port ?? 3200;
	const host = opts.host ?? "0.0.0.0";
	const store = opts.store;
	const secret = opts.secret;

	function getHealth(): HealthStats {
		const mem = process.memoryUsage();
		return {
			rss: mem.rss,
			heapUsed: mem.heapUsed,
			heapTotal: mem.heapTotal,
			eventLoopLag: 0,
			uptime: Date.now(),
		};
	}

	function getStats(): DashboardStats {
		const sessions = store.listSessions();
		const memories = store.list({ limit: 1000 });
		const observations = store.listRecentObservations(1000);
		const today = new Date().toISOString().slice(0, 10);
		const memoriesByType: Record<string, number> = {};
		for (const m of memories)
			memoriesByType[m.type] = (memoriesByType[m.type] || 0) + 1;
		const sessionsByStatus: Record<string, number> = {};
		for (const s of sessions)
			sessionsByStatus[s.status] = (sessionsByStatus[s.status] || 0) + 1;
		return {
			workspace: store.getCurrentWorkspace(),
			sessions: sessions.length,
			memories: memories.length,
			observations: observations.length,
			observationsToday: observations.filter(observation =>
				observation.timestamp.startsWith(today),
			).length,
			memoriesByType,
			sessionsByStatus,
		};
	}

	function checkAuth(auth: string | null): boolean {
		if (!secret) return true;
		return auth === `Bearer ${secret}`;
	}

	// Async request handler
	async function handleRequest(req: Request): Promise<Response> {
		const url = new URL(req.url);
		const path = url.pathname;

		// Favicon
		if (path === "/favicon.svg") {
			return new Response(
				`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#67e8f9"/><stop offset="1" stop-color="#a78bfa"/></linearGradient></defs><circle cx="50" cy="50" r="42" fill="#0e1420" stroke="url(#g)" stroke-width="5"/><text x="50" y="63" text-anchor="middle" fill="url(#g)" font-size="40" font-family="monospace" font-weight="bold">M</text></svg>`,
				{ headers: { "Content-Type": "image/svg+xml" } },
			);
		}

		// WebSocket upgrade
		if (path === "/ws") {
			if (server.upgrade(req)) return new Response(null, { status: 101 });
			return new Response("Upgrade failed", { status: 400 });
		}

		// API routes
		if (path.startsWith("/api/")) {
			const auth = req.headers.get("Authorization");
			if (!checkAuth(auth))
				return new Response("Unauthorized", { status: 401 });

			const segments = path
				.replace(/^\/api\//, "")
				.split("/")
				.filter(Boolean);

			if (
				segments.length === 0 ||
				(segments[0] === "stats" && segments.length === 1)
			) {
				return new Response(
					JSON.stringify({ stats: getStats(), health: getHealth() }),
					{ headers: { "Content-Type": "application/json" } },
				);
			}

			if (segments[0] === "sessions" && segments.length === 1) {
				return new Response(JSON.stringify(store.listSessions()), {
					headers: { "Content-Type": "application/json" },
				});
			}

			if (segments[0] === "sessions" && segments.length === 2) {
				const session = store.getSession(segments[1]);
				if (!session || session.workspace !== store.getCurrentWorkspace()) {
					return new Response("Not found", { status: 404 });
				}
				return new Response(JSON.stringify(session), {
					headers: { "Content-Type": "application/json" },
				});
			}

			if (segments[0] === "observations" && segments.length === 1) {
				const sessionId = url.searchParams.get("sessionId");
				const limit = Math.min(
					Math.max(parseInt(url.searchParams.get("limit") || "100", 10), 1),
					1000,
				);
				const minImportance = parseInt(
					url.searchParams.get("minImportance") || "0",
					10,
				);
				const type = url.searchParams.get("type") || undefined;
				const search = url.searchParams.get("search")?.trim();
				const scopedSession = sessionId ? store.getSession(sessionId) : null;
				const obs = sessionId
					? scopedSession?.workspace === store.getCurrentWorkspace()
						? store.listObservations(sessionId, limit)
						: []
					: search
						? store
								.searchObservations(search, limit)
								.map(result => result.observation)
						: store.listRecentObservations(
								limit,
								type as ObservationType | undefined,
							);
				const filtered = obs.filter(
					(o: any) =>
						o.importance >= minImportance && (!type || o.type === type),
				);
				return new Response(JSON.stringify(filtered), {
					headers: { "Content-Type": "application/json" },
				});
			}

			if (segments[0] === "memories" && segments.length === 1) {
				const search = url.searchParams.get("search") || undefined;
				const type = url.searchParams.get("type") || undefined;
				const minStrength = parseInt(
					url.searchParams.get("minStrength") || "0",
					10,
				);
				const limit = parseInt(url.searchParams.get("limit") || "100", 10);
				const memories = store.list({
					search,
					type: type as any,
					minStrength,
					limit,
				});
				return new Response(JSON.stringify(memories), {
					headers: { "Content-Type": "application/json" },
				});
			}

			if (segments[0] === "memories" && segments.length === 2) {
				const memory = store.get(segments[1]);
				if (!memory) return new Response("Not found", { status: 404 });
				return new Response(JSON.stringify(memory), {
					headers: { "Content-Type": "application/json" },
				});
			}

			if (segments[0] === "auto-tier") {
				const result = store.autoTierMemories();
				return new Response(JSON.stringify(result), {
					headers: { "Content-Type": "application/json" },
				});
			}

			if (segments[0] === "auto-forget") {
				const body = await req.json().catch(() => ({}));
				const result = store.autoForget(
					body.ttlMs as number,
					body.minImportance as number,
					body.maxDeletes as number,
				);
				return new Response(JSON.stringify(result), {
					headers: { "Content-Type": "application/json" },
				});
			}

			if (segments[0] === "forget") {
				const id = url.searchParams.get("id");
				if (!id) return new Response("id required", { status: 400 });
				const deleted = store.remove(id);
				return new Response(JSON.stringify({ deleted, id }), {
					headers: { "Content-Type": "application/json" },
				});
			}

			if (segments[0] === "working-memory") {
				const tiered: Record<string, any> = {};
				const memories = store.list({ limit: 1000 });
				for (const m of memories) {
					tiered[m.id] = {
						tier: store.getWorkingMemoryTier(m.id),
						strength: m.strength,
						type: m.type,
						content: m.content.slice(0, 100),
					};
				}
				return new Response(JSON.stringify(tiered), {
					headers: { "Content-Type": "application/json" },
				});
			}

			if (segments[0] === "activity") {
				const limit = parseInt(url.searchParams.get("limit") || "50", 10);
				const sessions = store.listSessions();
				const activity: any[] = [];
				for (const session of sessions) {
					const obs = store.listObservations(session.id, Math.min(limit, 5));
					for (const o of obs) {
						activity.push({
							type: "observation",
							sessionId: session.id,
							sessionProject: session.project,
							observation: o,
							timestamp: o.timestamp,
						});
					}
				}
				activity.sort(
					(a: any, b: any) =>
						new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
				);
				return new Response(JSON.stringify(activity.slice(0, limit)), {
					headers: { "Content-Type": "application/json" },
				});
			}

			if (segments[0] === "audit") {
				const limit = parseInt(url.searchParams.get("limit") || "100", 10);
				const memoryList = store.list({ limit: limit * 2 });
				const audit = memoryList.map((m: any) => ({
					id: m.id,
					operation: "create",
					resource: "memory",
					type: m.type,
					timestamp: m.createdAt,
					strength: m.strength,
				}));
				return new Response(JSON.stringify(audit), {
					headers: { "Content-Type": "application/json" },
				});
			}

			return new Response("Not found", { status: 404 });
		}

		// Serve dashboard HTML
		if (path === "/" || path === "/dashboard") {
			return new Response(documentHTML, {
				headers: {
					"Content-Type": "text/html; charset=utf-8",
					"Cache-Control": "no-cache",
				},
			});
		}

		return new Response("Not found", { status: 404 });
	}

	const server = Bun.serve({
		port,
		hostname: host,
		fetch: handleRequest,
		websocket: {
			open(ws) {
				ws.subscribe("observations");
			},
			message() {
				// Client messages (e.g. subscribe ack) require no server-side action.
			},
			close(ws) {
				ws.unsubscribe("observations");
			},
		},
	});

	boundPort = server.port ?? port;
	console.log(`[memory/viewer] Dashboard: http://localhost:${boundPort}`);

	const wrapped = {
		port: server.port ?? port,
		stop: (force?: boolean) => server.stop(force),
	};
	return {
		server: wrapped,
		stop: (force?: boolean) => {
			server.stop(force);
			boundPort = null;
		},
	};
}

import HTML from "./viewer-document.js";

const documentHTML = HTML as string;
