import { StatusBar } from "../footer/layout.ts";
import { createDefaultConfig } from "../footer/types.ts";

const bar = new StatusBar({
	...createDefaultConfig(),
	widgets: {
		reasoner: {
			enabled: true,
			row: 0,
			position: 1,
			align: "middle",
			fill: "none",
		},
	},
});
bar.update({
	phase: "ready",
	model: "test",
	contextTokens: 0,
	contextMaxTokens: 100000,
	reasoner: "loop-detector",
});
const lines = bar.render(300);
console.log("Line 0:", lines[0]);
console.log("Contains 'reasoner:'", lines[0].includes("reasoner:"));
console.log("Contains 'loop-detector'", lines[0].includes("loop-detector"));
