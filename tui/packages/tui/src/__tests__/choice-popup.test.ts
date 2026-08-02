import assert from "node:assert/strict";
import { test } from "node:test";
import { ChoicePopup } from "../overlays/choice-popup.ts";
import { SettingsSelectorOverlay } from "../overlays/settings-overlay.ts";
import { ThemeSelectorOverlay } from "../overlays/theme-selector.ts";
import { initTheme } from "../terminal/theme.ts";

initTheme("dark");

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
	const model = component.getInkOverlayModel();
	assert.equal(model.title, "ASK");
	assert.equal(model.items[0]?.label, "Focused fix");
	assert.match(model.items[0]?.metadata ?? "", /smallest safe change/);
	assert.equal(model.items[0]?.selected, true);
});

void test("choice popup exposes question copy to Ink", () => {
	const model = popup().getInkOverlayModel();
	assert.match(model.headerLines?.join("\n") ?? "", /next implementation\?/);
	assert.match(model.items[0]?.metadata ?? "", /current structure\./);
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
	assert.equal(component.getInkOverlayModel().headerLines?.[0], "Tests");
	component.handleInput("2");
	assert.equal(component.handleInput("\n"), null);
	const submit = component.getInkOverlayModel();
	assert.match(submit.footer, /Ready to submit/);
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
	themes.setThemes([
		{ name: "Dark", description: "Low-light terminal palette" },
		{ name: "Light", description: "Bright terminal palette" },
	]);
	themes.show();

	const settingsModel = settings.getInkOverlayModel();
	assert.equal(settingsModel.title, "Runtime Settings");
	assert.equal(settingsModel.items[0]?.selected, true);
	const themeModel = themes.getInkOverlayModel();
	assert.equal(themeModel.title, "Theme");
	assert.equal(themeModel.items.length, 2);
	assert.equal(themeModel.items[0]?.selected, true);

	settings.handleInput("\n");
	const detail = settings.getInkOverlayModel();
	assert.equal(detail.items.find((item) => item.selected)?.label, "High");
	assert.equal(detail.items.find((item) => item.current)?.label, "High");
});
