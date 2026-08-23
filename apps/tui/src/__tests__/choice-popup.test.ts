import { test } from "bun:test";
import assert from "node:assert/strict";
import { ChoicePopup } from "../overlays/choice-popup.ts";
import { SettingsSelectorOverlay } from "../overlays/settings-overlay.ts";
import { ThemeSelectorOverlay } from "../overlays/theme-selector.ts";
import { visibleWidth } from "../terminal/core.ts";
import { initTheme, theme } from "../terminal/theme.ts";

initTheme("dark");

const plain = (value: string): string =>
	// biome-ignore lint/suspicious/noControlCharactersInRegex: ANSI CSI escape sequence
	value.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "");

function popup(): ChoicePopup {
	const result = new ChoicePopup();
	result.setQuestion("How should we approach the next implementation?");
	result.setChoices([
		{
			value: "focused",
			label: "Focused fix",
			description:
				"Make the smallest safe change and keep the current structure.",
		},
		{
			value: "balanced",
			label: "Balanced refactor",
			description: "Improve the design while keeping the scope practical.",
		},
	]);
	result.show();
	return result;
}

void test("choice popup renders a compact ask-user card with stacked details", () => {
	const component = popup();
	const lines = component.render(64);
	const output = plain(lines.join("\n"));

	assert.match(output, /ASK {2}choose one/);
	assert.match(output, /● Focused fix {2}1/);
	assert.match(output, /○ Balanced refactor {2}2/);
	assert.match(output, /Make the smallest safe change/);
	assert.match(output, /↑↓ move {3}enter answer {3}esc dismiss/);
	const selectedDescription = lines.find(line =>
		plain(line).includes("Make the smallest safe change"),
	);
	assert.ok(selectedDescription?.includes(theme.fgRaw("active")));
	const selectedLabel = lines.find(line =>
		plain(line).includes("● Focused fix"),
	);
	assert.ok(selectedLabel?.includes(theme.fgRaw("selected")));
	assert.ok(lines.every(line => visibleWidth(line) <= 64));
});

void test("choice popup wraps copy and remains width-safe in narrow terminals", () => {
	const lines = popup().render(34);
	const output = plain(lines.join("\n"));

	assert.match(output, /next implementation\?/);
	assert.match(output, /current structure\./);
	assert.ok(lines.every(line => visibleWidth(line) <= 34));
});

void test("choice popup supports arrows, vim keys, and direct number selection", () => {
	const component = popup();
	component.handleInput("j");
	assert.equal(component.getSelected()?.value, "balanced");
	component.handleInput("k");
	assert.equal(component.getSelected()?.value, "focused");
	assert.equal(component.handleInput("2"), null);
	assert.equal(component.getSelected()?.value, "balanced");
	assert.deepEqual(component.handleInput("\n"), {
		type: "submit",
		answers: { answer: "balanced" },
	});
});

void test("multi-question popup uses tabs and submits a structured answer", () => {
	const component = new ChoicePopup();
	component.setQuestions([
		{
			id: "scope",
			header: "Scope",
			question: "How broad should the change be?",
			choices: [
				{ value: "small", label: "Focused" },
				{ value: "large", label: "Broad" },
			],
		},
		{
			id: "tests",
			header: "Tests",
			question: "Which validation level?",
			choices: [
				{ value: "unit", label: "Unit tests" },
				{ value: "full", label: "Full suite" },
			],
		},
	]);
	component.show();

	assert.equal(component.handleInput("\n"), null);
	assert.match(plain(component.render(80).join("\n")), /✓ Scope.*Tests/);
	component.handleInput("2");
	assert.equal(component.handleInput("\n"), null);
	const submit = plain(component.render(80).join("\n"));
	assert.match(submit, /Ready to submit your answers/);
	assert.deepEqual(component.handleInput("\n"), {
		type: "submit",
		answers: { scope: "small", tests: "full" },
	});
	assert.equal(
		component.getResponseValue(),
		JSON.stringify({ scope: "small", tests: "full" }),
	);
});

void test("settings and selectors share the same dialog frame and focus style", () => {
	const settings = new SettingsSelectorOverlay();
	settings.setSettings([
		{
			name: "Thinking",
			currentValue: "high",
			description: "Controls reasoning effort.",
			options: [
				{ label: "High", value: "high", current: true },
				{ label: "Low", value: "low" },
			],
		},
	]);
	settings.show();

	const themes = new ThemeSelectorOverlay();
	themes.setItems([
		{
			name: "Dark",
			description: "Low-light terminal palette",
			active: true,
		},
		{
			name: "Light",
			description: "Bright terminal palette",
			active: false,
		},
	]);
	themes.show();

	for (const lines of [settings.render(64), themes.render(64)]) {
		const output = plain(lines.join("\n"));
		assert.match(output, /^┌─/);
		assert.match(output, /❯/);
		assert.match(output, /↑↓/);
		assert.match(output, /└─/);
		// biome-ignore lint/suspicious/noControlCharactersInRegex: ANSI background escape
		assert.doesNotMatch(lines.join("\n"), /\x1b\[48;/);
		assert.ok(lines.every(line => visibleWidth(line) <= 64));
	}

	settings.handleInput("\n");
	const detail = plain(settings.render(64).join("\n"));
	assert.match(detail, /❯ High ✓/);
	assert.doesNotMatch(detail, /●/);

	const themeList = plain(themes.render(64).join("\n"));
	assert.match(themeList, /❯ Dark ✓/);
	assert.doesNotMatch(themeList, /Light ✓/);
});
