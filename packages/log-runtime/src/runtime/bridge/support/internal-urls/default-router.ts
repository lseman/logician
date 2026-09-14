import { AgentProtocolHandler } from "./agent-protocol.ts";
import { ConflictProtocolHandler } from "./conflict-protocol.ts";
import { IssueProtocolHandler } from "./issue-protocol.ts";
import { LocalProtocolHandler } from "./local-protocol.ts";
import { LogProtocolHandler } from "./log-protocol.ts";
import { McpProtocolHandler } from "./mcp-protocol.ts";
import { MemoryProtocolHandler } from "./memory-protocol.ts";
import { PrProtocolHandler } from "./pr-protocol.ts";
import { RagProtocolHandler } from "./rag-protocol.ts";
import { InternalUrlRouter } from "./router.ts";
import { SkillProtocolHandler } from "./skill-protocol.ts";
import { SshProtocolHandler } from "./ssh-protocol.ts";

/** Fresh handler table per session; backend services retain their own lifecycles. */
export function createInternalUrlRouter(): InternalUrlRouter {
	const router = new InternalUrlRouter();
	for (const handler of [
		new SkillProtocolHandler(),
		new MemoryProtocolHandler(),
		new LocalProtocolHandler(),
		new McpProtocolHandler(),
		new AgentProtocolHandler(),
		new LogProtocolHandler(),
		new SshProtocolHandler(),
		new ConflictProtocolHandler(),
		new IssueProtocolHandler(),
		new PrProtocolHandler(),
	])
		router.register(handler);
	return router;
}
