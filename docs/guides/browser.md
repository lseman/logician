---
title: Browser Automation
description: Puppeteer-based Chromium control for testing, scraping, and interactive automation.
---

# Browser Automation

The `browser` tool drives Chromium via Puppeteer with tab lifecycle, navigation,
and element interaction. It follows OMP's browser automation pattern with stable
tab names and a single browser instance shared across tabs.

## Operations

### `open(name?, url?)`

Create or reuse a named tab. Tabs persist across calls until explicitly closed.

```
browser({ operation: "open", name: "research", url: "https://example.com" })
```

If `name` is omitted, the tab is created as `"default"`. If `url` is provided,
the tab navigates immediately after opening.

### `goto(name, url)`

Navigate an existing tab to a URL.

### `click(name, selector)`

Click an element identified by CSS selector.

### `type(name, selector, text, delay?)`

Type text into an input element. Optional `delay` sets the millisecond delay
between keystrokes.

### `fill(name, selector, value)`

Fill an input element with a value. Handles both regular inputs and
contenteditable elements.

### `press(name, key)`

Press a keyboard key. Accepts Puppeteer key names like `"Enter"`, `"Escape"`,
`"ControlLeft"`, etc.

### `screenshot(name, type_format?, quality?)`

Capture the viewport. Returns a base64-encoded image with the result string.
`type_format` is `"jpeg"` (default) or `"png"`. `quality` is 0–100.

### `evaluate(name, expression)`

Execute JavaScript in the page context. Returns the result as JSON.

### `ariaSnapshot(name)`

Get an accessibility tree snapshot of the page for debugging or validation.

### `getUrl(name)` / `getTitle(name)`

Read the tab's current URL or title.

### `waitForSelector(name, selector, timeout_ms?)`

Wait for an element to appear in the DOM. Returns a status message.

### `waitForUrl(name, url, timeout_ms?)`

Wait for the page URL to match a pattern. Supports `*` as wildcard:
`waitForUrl("t", "*example.com*")`.

### `listTabs()`

Return a string listing all open tab names.

### `close(name)` / `closeAll()`

Release resources. `close(name)` removes one tab; `closeAll()` closes all
tabs and the underlying browser process.

## Example Workflow

```
# Open a research tab
browser({ operation: "open", name: "research", url: "https://example.com" })

# Interact with the page
browser({ operation: "click", name: "research", selector: "#search-input" })
browser({ operation: "type", name: "research", selector: "#search-input", text: "Puppeteer docs" })
browser({ operation: "press", name: "research", key: "Enter" })

# Wait for results
browser({ operation: "waitForUrl", name: "research", url: "*example.com/search*" })

# Capture and inspect
browser({ operation: "screenshot", name: "research" })
browser({ operation: "evaluate", name: "research", expression: "document.title" })
browser({ operation: "ariaSnapshot", name: "research" })

# Clean up
browser({ operation: "close", name: "research" })
```

## Configuration

The browser manager accepts these options via `DefaultToolsOptions`:

```json
{
  "browserManager": {
    "headless": true,
    "navigationTimeoutMs": 30000,
    "screenshotQuality": 80
  }
}
```

- `headless` — Run Chromium headless (default: `true`)
- `navigationTimeoutMs` — Page load timeout in ms (default: `30000`)
- `screenshotQuality` — JPEG quality for screenshots, 0–100 (default: `80`)

## Notes

- A single browser instance is shared across all tabs in a session.
- Tabs are scoped by name — opening a tab with an existing name reuses it.
- The browser is shut down when `closeAll()` is called or the session ends.
- Headless mode is the default and recommended for automation.
- For interactive debugging, set `headless: false` to see the browser window.
