// ── Browser capability ───────────────────────────────────────────────────────
// Puppeteer-based browser automation for testing, scraping, and interactive
// automation. Models OMP's browser automation pattern.

export { BrowserManager, type BrowserManagerConfig, type BrowserManagerDeps, type BrowserTab } from "./browser-manager.ts";
export { createBrowserTool, type BrowserToolDeps } from "./browser-tool.ts";
