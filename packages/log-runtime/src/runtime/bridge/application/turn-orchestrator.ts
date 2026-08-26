import type { RuntimeEvent } from "@logician/log-core/events";
import { RuntimeRunCoordinator } from "./run-coordinator.ts";

export interface TurnOrchestratorDependencies {
	extensionsReady: () => Promise<void>;
	hasSession: () => boolean;
	steer: (message: string) => void;
	emit: (event: RuntimeEvent) => void;
	ensureStartup: () => Promise<void>;
	isMcpLoaded: () => boolean;
	loadMcp: () => Promise<void>;
	reportMcpError: (error: unknown) => void;
	runTurn: (message: string) => Promise<void>;
}

/** Owns submission serialization, steering, startup, and turn prerequisites. */
export class TurnOrchestrator {
	private readonly dependencies: TurnOrchestratorDependencies;
	private readonly runs = new RuntimeRunCoordinator();

	constructor(dependencies: TurnOrchestratorDependencies) {
		this.dependencies = dependencies;
	}

	async submit(message: string): Promise<void> {
		await this.dependencies.extensionsReady();
		return this.runs.submit({
			message,
			canSteer: () => this.dependencies.hasSession(),
			steer: text => {
				this.dependencies.steer(text);
				this.dependencies.emit({ type: "steered", message: text });
			},
			execute: text => this.execute(text),
		});
	}

	isActive(): boolean {
		return this.runs.isActive();
	}

	reset(): void {
		this.runs.reset();
	}

	private async execute(message: string): Promise<void> {
		await this.dependencies.ensureStartup();
		if (!this.dependencies.isMcpLoaded()) {
			void this.dependencies
				.loadMcp()
				.catch(error => this.dependencies.reportMcpError(error));
		}
		await this.dependencies.runTurn(message);
	}
}
