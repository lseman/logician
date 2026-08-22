import { StatusBar } from "./apps/tui/src/footer/layout.ts";
import { createDefaultConfig } from "./apps/tui/src/footer/config.ts";

const mod = await import("./apps/tui/src/__tests__/theme-setup.ts");
if (mod.setupTheme) mod.setupTheme();

const bar = new StatusBar(createDefaultConfig());
bar.update({
  phase: "ready",
  model: "test",
  contextTokens: 0,
  contextMaxTokens: 100000,
  reasoner: "loop-detector",
});
const lines = bar.render(120);
console.log("Line 0 raw:", JSON.stringify(lines[0]));

// Try with fewer widgets to reduce clipping pressure
bar.update({
  phase: "idle",
  model: "",
  contextTokens: 0,
  contextMaxTokens: 0,
  reasoner: "loop-detector",
});
const lines2 = bar.render(120);
console.log("Line 2 raw:", JSON.stringify(lines2[0]));
