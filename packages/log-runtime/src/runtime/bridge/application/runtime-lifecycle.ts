export interface RuntimeLifecycleDependencies {
	cancel: () => Promise<unknown>;
	resetTurns: () => void;
	dropSession: () => void;
	clearSession: () => void;
	resetIdentity: () => void;
	endPluginSession: (reason: string) => Promise<void>;
	resetPlugin: (options?: { clearResult?: boolean }) => void;
	refreshPluginContext: () => void;
	resetInjectedContext: () => void;
	resetDiscoveredResources: () => void;
	injectSkills: () => Promise<void>;
	injectPrompts: () => Promise<void>;
	reloadExtensions: () => Promise<void>;
	reportExtensionError: (error: unknown) => void;
	extensionsReady: () => Promise<unknown>;
	ensurePluginsStarted: () => Promise<void>;
	ensureSession: () => void;
	loadMcp: () => Promise<void>;
	reportMcpError: (error: unknown) => void;
	waitForMemory: () => Promise<unknown>;
	closeResources: () => Promise<void>;
	resetActivity: () => void;
	publishUsage: () => void;
	emitTurnEnd: (turnId: string) => void;
}

/** Owns ordering and cleanup invariants for the runtime lifecycle. */
export class RuntimeLifecycle {
	private readonly dependencies: RuntimeLifecycleDependencies;

	constructor(dependencies: RuntimeLifecycleDependencies) {
		this.dependencies = dependencies;
	}

	async initialize(): Promise<void> {
		await this.dependencies.extensionsReady();
		await this.dependencies.ensurePluginsStarted();
		this.dependencies.ensureSession();
		void this.dependencies
			.loadMcp()
			.catch(error => this.dependencies.reportMcpError(error));
	}

	async reload(): Promise<void> {
		void this.dependencies.cancel();
		this.dependencies.resetTurns();
		this.dependencies.dropSession();
		this.dependencies.resetIdentity();
		this.dependencies.resetDiscoveredResources();
		this.dependencies.resetPlugin({ clearResult: true });
		await this.dependencies.injectSkills();
		await this.dependencies.injectPrompts();
		await this.dependencies
			.reloadExtensions()
			.catch(error => this.dependencies.reportExtensionError(error));
		await this.dependencies.loadMcp();
		this.dependencies.emitTurnEnd("reload");
	}

	reset(): void {
		void this.dependencies.endPluginSession("reset");
		this.dependencies.clearSession();
		this.dependencies.resetIdentity();
		this.dependencies.resetInjectedContext();
		this.dependencies.resetPlugin();
		this.dependencies.refreshPluginContext();
		this.dependencies.resetActivity();
		this.dependencies.publishUsage();
		this.dependencies.emitTurnEnd("reset");
	}

	async stop(): Promise<void> {
		void this.dependencies.cancel();
		await this.dependencies.waitForMemory();
		await this.dependencies.endPluginSession("shutdown");
		await this.dependencies.closeResources();
		this.dependencies.resetTurns();
	}
}
