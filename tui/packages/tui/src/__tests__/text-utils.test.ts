import { test } from "bun:test";
import assert from "node:assert/strict";
import { renderMarkdownLine } from "../rendering/transcript/text-utils.ts";
import { initTheme, theme } from "../terminal/theme.ts";

initTheme("dark");

const plain = (value: string): string =>
	value.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "");

test("skill metadata renders XML entities and Markdown escapes as readable text", () => {
	const rendered = plain(
		renderMarkdownLine(
			"<skill display\\_name=&quot;context-mode&quot; slash\\_command=&quot;/context-mode:context-mode&quot;>Triggers: &quot;analyze logs&quot; &amp; tests</skill>",
			theme.fgRaw("assistantText"),
		),
	);

	assert.equal(
		rendered,
		'<skill display_name="context-mode" slash_command="/context-mode:context-mode">Triggers: "analyze logs" & tests</skill>',
	);
});
