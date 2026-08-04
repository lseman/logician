import assert from "node:assert/strict";
import { test } from "node:test";
import { FileMentionPopup } from "../overlays/file-mention-popup.ts";
import { InputBar } from "../input/input-bar.ts";
import { initTheme } from "../terminal/theme.ts";

initTheme("dark");

void test("file mention popup fuzzy-matches by basename and renders count", () => {
	const popup = new FileMentionPopup();
	popup.setFiles([
		"packages/tui/src/app/tui.ts",
		"packages/tui/src/input/input-bar.ts",
		"packages/agent-core/src/agent/agent-loop-runner.ts",
		"README.md",
	]);
	popup.setQuery("tui.");
	popup.show();
	const rendered = popup.render(120).join("\n");
	assert.match(rendered, /files.*\(1\)/);
	assert.equal(popup.currentFile(), "packages/tui/src/app/tui.ts");
});

void test("file mention popup windows long match lists with more-above/below", () => {
	const popup = new FileMentionPopup();
	const files = Array.from({ length: 15 }, (_, i) => `file-${i}.ts`);
	popup.setFiles(files);
	popup.setQuery("");
	popup.show();
	let rendered = popup.render(120).join("\n");
	assert.match(rendered, /files.*\(15\)/);
	assert.match(rendered, /more below/);
	for (let i = 0; i < files.length - 1; i++) popup.moveSelection(1);
	rendered = popup.render(120).join("\n");
	assert.match(rendered, /more above/);
});

void test("file mention popup reports no matches without throwing", () => {
	const popup = new FileMentionPopup();
	popup.setFiles(["src/a.ts", "src/b.ts"]);
	popup.setQuery("zzz-nonexistent");
	popup.show();
	const rendered = popup.render(120).join("\n");
	assert.match(rendered, /No matching files/);
	assert.equal(popup.currentFile(), null);
	assert.equal(popup.hasMatches(), false);
});

void test("input bar detects the active @-mention token at the cursor", () => {
	const bar = new InputBar();
	bar.valueText = "check @tui-co";
	assert.equal(bar.getActiveMentionQuery(), "tui-co");

	const noMention = new InputBar();
	noMention.valueText = "no mention here";
	assert.equal(noMention.getActiveMentionQuery(), null);

	const finished = new InputBar();
	finished.valueText = "email me@example.com please";
	assert.equal(finished.getActiveMentionQuery(), null);
});

void test("input bar splices the accepted mention path into the active token", () => {
	const bar = new InputBar();
	bar.valueText = "check @tui";
	bar.insertMention("packages/tui/src/app/tui.ts");
	assert.equal(bar.valueText, "check @packages/tui/src/app/tui.ts ");
});
