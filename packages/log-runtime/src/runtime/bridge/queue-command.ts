interface QueueCommandTarget {
	flushSteeringNow(): number;
	getQueues(): {
		steering: string[];
		followUp: string[];
		nextTurn: string[];
	};
	clearQueues(): {
		steering: string[];
		followUp: string[];
		nextTurn: string[];
	};
	dropQueuedMessage(displayIndex: number): string | undefined;
}

export interface QueueCommandResult {
	text: string;
	level: "info" | "warn";
}

/** Interpret runtime-owned queue commands against core queue primitives. */
export function runQueueCommand(
	session: QueueCommandTarget | null,
	command: string,
): QueueCommandResult | null {
	if (!session) return null;

	if (command === "/steer-now") {
		const count = session.flushSteeringNow();
		return {
			text:
				count > 0
					? `Processing ${count} queued steering message${count === 1 ? "" : "s"} now.`
					: "No queued steering messages to process.",
			level: count > 0 ? "info" : "warn",
		};
	}

	if (command === "/queue") {
		const { steering, followUp } = session.getQueues();
		const rows = [
			...steering.map((message, index) => `${index + 1}. ▸ ${message}`),
			...followUp.map(
				(message, index) => `${steering.length + index + 1}. ↳ ${message}`,
			),
		];
		return {
			text: rows.length ? rows.join("\n") : "Queue is empty.",
			level: "info",
		};
	}

	if (command === "/queue-clear") {
		const cleared = session.clearQueues();
		const count =
			cleared.steering.length +
			cleared.followUp.length +
			cleared.nextTurn.length;
		return {
			text: `Cleared ${count} queued message${count === 1 ? "" : "s"}.`,
			level: "info",
		};
	}

	if (command === "/queue-drop" || command.startsWith("/queue-drop ")) {
		const value = Number.parseInt(
			command.slice("/queue-drop".length).trim(),
			10,
		);
		const removed =
			Number.isInteger(value) && value > 0
				? session.dropQueuedMessage(value - 1)
				: undefined;
		return {
			text: removed ? `Removed: ${removed}` : "Usage: /queue-drop <number>",
			level: removed ? "info" : "warn",
		};
	}

	return null;
}
