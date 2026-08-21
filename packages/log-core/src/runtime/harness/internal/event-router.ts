import {
	isExtensionLifecycleEvent,
	type RuntimeEvent,
} from "../../events/contracts.ts";
import type { ExtensionRunner } from "../../../system/extension/runner.ts";

export interface HarnessEventRouterDeps {
	reduce(event: RuntimeEvent): void;
	notifyApplication(event: RuntimeEvent): void;
	persistCompletedMessage(event: RuntimeEvent): void;
	getExtensionRunner(): ExtensionRunner | undefined;
	getExtensionContext(): { sessionId: string; cwd: string };
}

/** Routes one runtime stream to its state, application, storage, and extension consumers. */
export class HarnessEventRouter {
	constructor(private readonly deps: HarnessEventRouterDeps) {}

	async route(event: RuntimeEvent): Promise<void> {
		this.deps.reduce(event);
		// The primary application path is synchronous and latency-sensitive.
		this.deps.notifyApplication(event);
		this.deps.persistCompletedMessage(event);

		const runner = this.deps.getExtensionRunner();
		if (!runner || !isExtensionLifecycleEvent(event)) return;
		await runner.emitToAll({
			type: event.type,
			context: { ...this.deps.getExtensionContext(), ...event },
		});
	}
}
