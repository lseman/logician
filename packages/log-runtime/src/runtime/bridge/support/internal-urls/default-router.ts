import { AgentProtocolHandler } from "./agent-protocol.ts";
import { ConflictProtocolHandler } from "./conflict-protocol.ts";
import { IssueProtocolHandler } from "./issue-protocol.ts";
import { LocalProtocolHandler } from "./local-protocol.ts";
import { LogProtocolHandler } from "./log-protocol.ts";
import { McpProtocolHandler } from "./mcp-protocol.ts";
import { MemoryProtocolHandler } from "./memory-protocol.ts";
import { PrProtocolHandler } from "./pr-protocol.ts";
import { RagProtocolHandler } from "./rag-protocol.ts";
import { RuleProtocolHandler } from "./rule-protocol.ts";
import { InternalUrlRouter } from "./router.ts";
import { SkillProtocolHandler } from "./skill-protocol.ts";
import { SshProtocolHandler } from "./ssh-protocol.ts";

/** Fresh handler table per session; backend services retain their own lifecycles. */
export function createInternalUrlRouter(
	skills: Array<{ name: string; content: string; path: string }> = [],
): InternalUrlRouter {
	const router = new InternalUrlRouter();

	// Configure skill handler with available skills for promptDoc
	const skillHandler = new SkillProtocolHandler();
	skillHandler.skills = skills;

	for (const handler of [
		skillHandler,
		new MemoryProtocolHandler(),
		new LocalProtocolHandler(),
		new McpProtocolHandler(),
		new RagProtocolHandler(),
		new AgentProtocolHandler(),
		new LogProtocolHandler(),
		new SshProtocolHandler(),
		new ConflictProtocolHandler(),
		new IssueProtocolHandler(),
		new PrProtocolHandler(),
		new RuleProtocolHandler(),
	])
		router.register(handler);
	return router;
}
