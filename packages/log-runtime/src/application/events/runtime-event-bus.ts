import {
	createNotification,
	type RuntimeEvent,
} from "@logician/log-protocol";
import type { ErrorCallback, ProtocolCallback } from "../bridge/types.ts";

/** Ordered runtime notifications and asynchronous error delivery. */
export class RuntimeEventBus {
	private subscribers = new Set<ProtocolCallback>();
	private errorCallback: ErrorCallback | null = null;
	private sequence = 0;

	subscribe(callback: ProtocolCallback): () => void {
		this.subscribers.add(callback);
		return () => this.subscribers.delete(callback);
	}

	onError(callback: ErrorCallback): () => void {
		this.errorCallback = callback;
		return () => {
			if (this.errorCallback === callback) this.errorCallback = null;
		};
	}

	reportError(error: unknown): void {
		const normalized =
			error instanceof Error ? error : new Error(String(error));
		this.emit({
			type: "notice",
			level: "error",
			label: "Error",
			text: normalized.message,
		});
		this.notifyError(normalized);
	}

	/** Deliver an error without adding a second transcript notice. */
	notifyError(error: unknown): void {
		const normalized =
			error instanceof Error ? error : new Error(String(error));
		this.errorCallback?.(normalized);
	}

	emit(event: RuntimeEvent): void {
		const notification = createNotification(event, ++this.sequence);
		for (const callback of this.subscribers) {
			try {
				callback(notification);
			} catch {
				// A client subscriber cannot interrupt the runtime.
			}
		}
	}
}
