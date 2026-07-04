// Hook execution metrics: track latency, errors, and event counts.

export interface HookMetrics {
	eventType: string;
	count: number;
	totalMs: number;
	minMs: number;
	maxMs: number;
	avgMs: number;
	errors: number;
	lastRunMs?: number;
	lastError?: Error;
}

export class HookMetricsCollector {
	private metrics = new Map<string, HookMetrics>();

	record(eventType: string, durationMs: number, error?: Error): void {
		let m = this.metrics.get(eventType);
		if (!m) {
			m = {
				eventType,
				count: 0,
				totalMs: 0,
				minMs: Infinity,
				maxMs: 0,
				avgMs: 0,
				errors: 0,
			};
			this.metrics.set(eventType, m);
		}

		m.count++;
		m.totalMs += durationMs;
		m.minMs = Math.min(m.minMs, durationMs);
		m.maxMs = Math.max(m.maxMs, durationMs);
		m.avgMs = m.totalMs / m.count;
		m.lastRunMs = durationMs;

		if (error) {
			m.errors++;
			m.lastError = error;
		}
	}

	get(eventType: string): HookMetrics | undefined {
		return this.metrics.get(eventType);
	}

	getAll(): HookMetrics[] {
		return Array.from(this.metrics.values());
	}

	getSlowHooks(thresholdMs: number): HookMetrics[] {
		return Array.from(this.metrics.values()).filter(
			(m) => m.avgMs >= thresholdMs,
		);
	}

	getFailingHooks(): HookMetrics[] {
		return Array.from(this.metrics.values()).filter((m) => m.errors > 0);
	}

	clear(): void {
		this.metrics.clear();
	}

	summary(): string {
		const all = this.getAll();
		if (all.length === 0) return "No hook metrics recorded";

		const lines = all.map((m) => {
			const errRate = m.count > 0 ? ((m.errors / m.count) * 100).toFixed(1) : "0";
			return `${m.eventType}: ${m.count} calls, avg ${m.avgMs.toFixed(1)}ms (min ${m.minMs.toFixed(1)}, max ${m.maxMs.toFixed(1)}), errors ${m.errors} (${errRate}%)`;
		});
		return lines.join("\n");
	}
}
