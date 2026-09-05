import type {
	NamedAgentStopPolicy,
	StopPolicyContext,
} from "./execution-policy.ts";
import type { MutationReceipt } from "../../system/types/types-messages.ts";

const MUTATION_TOOLS = new Set(["apply_patch", "edit_file", "write_file"]);
const VERIFICATION_TOOLS = new Set(["bash", "sandbox"]);
const VERIFICATION_COMMAND =
	/\b(?:bun\s+(?:test|run\s+(?:test|check|lint|build|typecheck))|npm\s+(?:test|run\s+(?:test|check|lint|build|typecheck))|pnpm\s+(?:test|run\s+(?:test|check|lint|build|typecheck))|yarn\s+(?:test|run\s+(?:test|check|lint|build|typecheck))|cargo\s+(?:test|check|build)|go\s+test|python\s+-m\s+pytest|pytest|make(?:\s+\w+)?|cmake\s+--build|tsc|biome\s+check|eslint)\b/i;
const FAILED_RESULT =
	/(?:exit(?:ed)?(?: with)?(?: code)?\s*[1-9]\d*|\bfailed\b|\berror:|not ok)/i;

function verificationMissing(context: StopPolicyContext): boolean {
	let lastMutation = -1;
	const calls = new Map<string, { index: number; name: string; verification: boolean }>();
	let verifiedAfterMutation = false;

	for (const [index, message] of context.newMessages.entries()) {
		for (const call of message.tool_calls ?? []) {
			let args = "";
			try {
				args = JSON.stringify(JSON.parse(call.arguments));
			} catch {
				args = call.arguments;
			}
			calls.set(call.id, {
				index,
				name: call.name,
				verification:
					VERIFICATION_TOOLS.has(call.name) && VERIFICATION_COMMAND.test(args),
			});
		}
		if (message.role !== "tool" || !message.tool_call_id) continue;
		const call = calls.get(message.tool_call_id);
		const receipt = message.details?.mutation as MutationReceipt | undefined;
		if (call && MUTATION_TOOLS.has(call.name) && receipt?.kind === "mutation") {
			if (receipt.applied && receipt.changed) {
				lastMutation = call.index;
				verifiedAfterMutation = false;
			}
		}
		if (
			call?.verification &&
			call.index > lastMutation &&
			!FAILED_RESULT.test(message.content ?? "")
		) {
			verifiedAfterMutation = true;
		}
	}
	return lastMutation >= 0 && !verifiedAfterMutation;
}

export function createVerifiedStopPolicy(): NamedAgentStopPolicy {
	return {
		id: "verified-stop",
		description:
			"Require successful verification after the final file mutation",
		kind: "deterministic",
		evaluate: context =>
			verificationMissing(context)
				? {
						action: "continue",
						messages: [
							{
								role: "user",
								content:
									"Before finishing, run the narrowest relevant verification command for the changes and address any failure.",
							},
						],
					}
				: undefined,
	};
}
