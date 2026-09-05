/**
 * Provider refusal detection and filtering.
 *
 * Detects API-level provider refusals (safety / content-policy) and removes
 * them from replayed context. Prevents refusal messages from polluting
 * conversation history and wasting tokens on future turns.
 *
 * OpenAI-compatible providers return a `refusal` field in the assistant
 * response when the model refuses to answer. These messages carry no
 * useful dialogue content — they are terminal policy rejections, not
 * conversation turns.
 */

import type { Message } from "../../system/types/types-messages.ts";

/**
 * Detects provider refusals in a response.
 * A refusal is a safety/content-policy rejection that should not be
 * stored in conversation history or replayed to the provider.
 */
export function isProviderRefusal(
	_content: string | null | undefined,
	refusal: string | undefined,
): boolean {
	if (!refusal) return false;
	// If the provider sent refusal text, this is a refusal message.
	// The refusal text itself is policy metadata, not useful dialogue.
	return true;
}

/**
 * Detects a refusal flag on an assistant message (set during response
 * processing by the loop layer). Messages with `details.refusal === true`
 * are provider refusals.
 */
export function isProviderRefusalMessage(message: Message): boolean {
	if (message.role !== "assistant") return false;
	const details = message.details as Record<string, unknown> | undefined;
	return details?.refusal === true;
}

/**
 * Removes provider refusal assistant messages from the message list.
 * Preserves all non-assistant messages and assistant messages that are
 * not refusals. Used when preparing messages for replay to the provider.
 */
export function filterProviderReplayMessages(
	messages: readonly Message[],
): Message[] {
	return messages.filter(
		message => message.role !== "assistant" || !isProviderRefusalMessage(message),
	);
}
