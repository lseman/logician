/* ============================== DATA MODEL ============================== */
const NODES = {
	home: {
		title: "LOGICIAN",
		kicker: "guest@logician:~$ ls -la subsystems/",
		summary:
			"Click any box in the diagram — or press 1–5 — to enter it and see how it works.",
		color: "#22d3ee",
		type: "diagram",
		path: "~",
		children: ["tui", "harness", "capabilities", "coding-agent", "external"],
	},

	tui: {
		title: "Terminal UI Layer",
		kicker: "drwxr-xr-x  tui/",
		path: "tui",
		color: "#8b5cf6",
		type: "grid",
		summary:
			"Everything the user sees and types. Rendering, input handling, theming, and modal overlays — the terminal-native front end (apps/tui).",
		children: ["tui-rendering", "tui-input", "tui-themes", "tui-overlays"],
	},
	"tui-rendering": {
		title: "Rendering",
		kicker: "tui/rendering",
		color: "#8b5cf6",
		type: "leaf",
		summary: "Draws the live transcript on screen.",
		bullets: [
			"Differential rendering — only the changed region of the terminal is redrawn, so nothing flickers.",
			"Renders streamed model tokens, tool calls, and thinking blocks as they arrive.",
			"Shares the same terminal frame with input and overlays.",
		],
	},
	"tui-input": {
		title: "Input",
		kicker: "tui/input",
		color: "#8b5cf6",
		type: "leaf",
		summary:
			"Captures what the user types and turns it into messages for the harness.",
		bullets: [
			"Multi-line editing with autocomplete for file paths and slash commands.",
			"Slash commands (/compact, /fork, /reset, /rewind…) are parsed here before reaching the Message Queue.",
			"Steering messages sent mid-run are queued the same way as a fresh prompt.",
		],
	},
	"tui-themes": {
		title: "Themes",
		kicker: "tui/themes",
		color: "#8b5cf6",
		type: "leaf",
		summary: "Terminal color schemes.",
		bullets: [
			"Defines the palette this very site borrows for its subsystem colors.",
			"Swappable independently of layout — the terminal chrome stays constant.",
		],
	},
	"tui-overlays": {
		title: "Overlays",
		kicker: "tui/overlays",
		color: "#8b5cf6",
		type: "leaf",
		summary: "Modal surfaces drawn on top of the transcript.",
		bullets: [
			"ask_user prompts pause the loop and wait for an answer here.",
			"Permission confirmations for acceptEdits / ask permission modes.",
			"File pickers and other transient UI.",
		],
	},

	harness: {
		title: "Harness — Agent Loop",
		kicker: "drwxr-xr-x  agent-core/",
		path: "agent-core",
		color: "#22d3ee",
		type: "pipeline",
		summary:
			"The loop that orchestrates every model call and tool execution. 10 stages, two nested loops.",
		children: ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
	},
	s1: {
		n: 1,
		stype: "pipeline",
		title: "Message Queue",
		kicker: "agent-core/message-queue",
		color: "#22d3ee",
		type: "leaf",
		summary: "Where a turn begins.",
		bullets: [
			"Buffers the user's new message together with any steering or follow-up messages sent mid-run.",
			"Hands the front of the queue to Context Assembly.",
			"Slash-command interrupts (/compact /fork /reset /rewind /bookmark) enter here too.",
		],
	},
	s2: {
		n: 2,
		stype: "pipeline",
		title: "Context Assembly",
		kicker: "agent-core/context",
		color: "#22d3ee",
		type: "leaf",
		summary: "Builds what the model will actually see this turn.",
		bullets: [
			"Combines the system prompt, loaded context files, and the session transcript so far.",
			"Pulls in cross-session context from agent memory.",
		],
	},
	s3: {
		n: 3,
		stype: "gate",
		title: "Compaction Check",
		kicker: "agent-core/compaction",
		color: "#06b6d4",
		type: "leaf",
		summary: "A gate, not a stage that always fires.",
		bullets: [
			"If compaction is enabled and the transcript is near the configured reserveTokens limit, older turns are summarized first.",
			"Otherwise this is a no-op and the turn proceeds untouched.",
		],
	},
	s4: {
		n: 4,
		stype: "pipeline",
		title: "Model Call",
		kicker: "agent-core/model-call",
		color: "#22d3ee",
		type: "leaf",
		summary: "Streams the request to the configured LLM backend.",
		bullets: [
			"Talks to any OpenAI-compatible API.",
			"Thinking tokens and partial text appear live in the TUI as they stream.",
			"Can be wrapped by an alternate step strategy instead of a single-shot call.",
		],
		related: [["Reasoners · SSR · ToT · Reflexion", "var(--c-cap)"]],
	},
	s5: {
		n: 5,
		stype: "gate",
		title: "Parse Response",
		kicker: "agent-core/parse",
		color: "#06b6d4",
		type: "leaf",
		summary: "The loop's real branch point.",
		bullets: [
			"Splits the model's response into plain text and tool calls.",
			"No tool calls → the turn is done: response streams to the TUI, harness returns to Message Queue and awaits the next input. This is the outer, turn-to-turn loop.",
			"Tool calls present → continue to Pre-Tool Hooks. This is the inner, within-turn loop back to Model Call.",
		],
	},
	s6: {
		n: 6,
		stype: "pipeline",
		title: "Pre-Tool Hooks",
		kicker: "agent-core/hooks/pre",
		color: "#22d3ee",
		type: "leaf",
		summary:
			"Before a tool actually runs, registered extensions get a chance to act.",
		bullets: [
			"Can intercept, modify, or block the call entirely.",
			"Claude-style lifecycle hooks — the same extension point skills and plugins build on.",
		],
	},
	s7: {
		n: 7,
		stype: "gate",
		title: "Permission Gate",
		kicker: "agent-core/permissions",
		color: "#06b6d4",
		type: "leaf",
		summary: "Checks the active permission mode before anything executes.",
		bullets: [
			"acceptAll — execute everything without asking.",
			"acceptEdits — ask before writes, run everything else.",
			"ask — confirm every tool call.",
			"plan — never execute, just show what would happen.",
		],
	},
	s8: {
		n: 8,
		stype: "tool",
		title: "Tool Execution",
		kicker: "agent-core/tools",
		color: "#34d399",
		type: "leaf",
		summary: "Dispatches to the tool registry.",
		bullets: [
			"File ops, search, system commands, agent primitives, and web & docs tools.",
			"Can route to an external MCP server instead of a built-in tool.",
			"spawn_agent branches into an isolated subagent here.",
		],
		related: [["Subagents · isolated worktree", "var(--c-code)"]],
	},
	s9: {
		n: 9,
		stype: "pipeline",
		title: "Post-Tool Hooks",
		kicker: "agent-core/hooks/post",
		color: "#22d3ee",
		type: "leaf",
		summary: "Formats the result before it re-enters the transcript.",
		bullets: [
			"For JS/TS/JSON edits, runs a fast syntax check before moving on (post-edit diagnostics).",
			"Last chance for an extension to reshape a tool result.",
		],
	},
	s10: {
		n: 10,
		stype: "mem",
		title: "Append to Session",
		kicker: "agent-core/session",
		color: "#22d3ee",
		type: "leaf",
		summary: "Writes the tool result into the transcript.",
		bullets: [
			"Pending writes are flushed at defined save points, not on every partial update.",
			"This is what agent memory persists across sessions.",
		],
	},

	capabilities: {
		title: "Agent Capabilities",
		kicker: "drwxr-xr-x  agent-capabilities/",
		path: "agent-capabilities",
		color: "#06b6d4",
		type: "grid",
		summary:
			"Higher-level behaviors the harness can call on: task tracking, user prompts, child agents, and structured reasoning.",
		children: [
			"cap-todo",
			"cap-ask-user",
			"cap-subagents",
			"cap-reasoners",
			"cap-eoh",
		],
	},
	"cap-todo": {
		title: "todo",
		kicker: "agent-capabilities/todo",
		color: "#06b6d4",
		type: "leaf",
		summary: "Task tracking with status transitions.",
		bullets: [
			"Lets the agent break a request into steps and report progress on each one.",
			"Visible in the TUI as the run progresses.",
		],
	},
	"cap-ask-user": {
		title: "ask-user",
		kicker: "agent-capabilities/ask-user",
		color: "#06b6d4",
		type: "leaf",
		summary: "User input prompts.",
		bullets: [
			"Pauses the loop to ask the user a direct question through a TUI overlay.",
			"Resumes the loop with the answer once given.",
		],
	},
	"cap-subagents": {
		title: "subagents",
		kicker: "agent-capabilities/subagents",
		color: "#06b6d4",
		type: "leaf",
		summary: "Child agents, isolated worktrees.",
		bullets: [
			"spawn_agent starts a child agent with a fresh context window.",
			"Runs in an isolated worktree so it can't collide with the parent's changes.",
			"Only a condensed summary returns to the parent — not the full child transcript.",
		],
	},
	"cap-reasoners": {
		title: "reasoners",
		kicker: "agent-capabilities/reasoners",
		color: "#06b6d4",
		type: "leaf",
		summary: "Structured reasoning strategies.",
		bullets: [
			"SSR, Tree of Thoughts, and Reflexion — alternatives to a single-shot model call.",
			"Plug into the Model Call stage of the harness loop.",
		],
	},
	"cap-eoh": {
		title: "eoh",
		kicker: "agent-capabilities/eoh",
		color: "#06b6d4",
		type: "leaf",
		summary: "An additional capability module exported alongside the others.",
		bullets: ["Exported from packages/agent-capabilities."],
	},

	"coding-agent": {
		title: "Coding Agent",
		kicker: "drwxr-xr-x  coding-agent/",
		path: "coding-agent",
		color: "#34d399",
		type: "grid",
		summary:
			"The full coding-agent runtime built on top of the harness: built-in tools, skills, MCP support, and sessions.",
		children: [
			"ca-tools",
			"ca-skills",
			"ca-mcp",
			"ca-context",
			"ca-prompts",
			"ca-sessions",
		],
	},
	"ca-tools": {
		title: "Tools",
		kicker: "coding-agent/tools",
		color: "#34d399",
		type: "leaf",
		summary: "Built-in tools the model can call directly.",
		bullets: [
			"File ops: read, write, edit.",
			"Search: grep-style content search across the repo.",
			"System: bash and other shell-level commands.",
			"Web & docs: web_search / web_fetch.",
		],
	},
	"ca-skills": {
		title: "Skills",
		kicker: "coding-agent/skills",
		color: "#34d399",
		type: "leaf",
		summary: "SKILL.md-driven capabilities.",
		bullets: [
			"Best-practice playbooks the agent reads before touching a given file type or task.",
			"Loaded on demand rather than kept in the system prompt at all times.",
		],
	},
	"ca-mcp": {
		title: "MCP client",
		kicker: "coding-agent/mcp",
		color: "#34d399",
		type: "leaf",
		summary: "Connects to external MCP servers.",
		bullets: [
			"Supports both stdio and streamable HTTP transports.",
			"Exposes each server's tools to the harness's Tool Execution stage as if they were local.",
		],
	},
	"ca-context": {
		title: "Context files",
		kicker: "coding-agent/context-files",
		color: "#34d399",
		type: "leaf",
		summary: "Loads repository and documentation context.",
		bullets: [
			"Feeds into Context Assembly in the harness loop alongside the system prompt.",
		],
	},
	"ca-prompts": {
		title: "Prompts + Trust",
		kicker: "coding-agent/prompts",
		color: "#34d399",
		type: "leaf",
		summary: "System prompts and permission modes.",
		bullets: [
			"Defines what a tool call is allowed to do before the Permission Gate checks it.",
			"acceptAll / acceptEdits / ask / plan are configured here.",
		],
	},
	"ca-sessions": {
		title: "Sessions",
		kicker: "coding-agent/sessions",
		color: "#34d399",
		type: "leaf",
		summary: "Session persistence, bookmarks, and rewind.",
		bullets: [
			"The same store Append to Session writes into on every turn.",
			"Bookmarks mark a point to return to; rewind restores an earlier checkpoint.",
		],
	},

	external: {
		title: "External Systems",
		kicker: "drwxr-xr-x  external/",
		path: "external",
		color: "#64748b",
		type: "grid",
		summary:
			"Systems Logician talks to outside its own process — the model, other tool servers, and the local machine.",
		children: ["ext-llm", "ext-mcp", "ext-searxng", "ext-fs"],
	},
	"ext-llm": {
		title: "LLM Backend",
		kicker: "external/llm-backend",
		color: "#64748b",
		type: "leaf",
		summary: "OpenAI-compatible API.",
		bullets: [
			"The configured model that actually generates responses and tool calls.",
			"Called from the harness's Model Call stage.",
		],
	},
	"ext-mcp": {
		title: "MCP Servers",
		kicker: "external/mcp-servers",
		color: "#64748b",
		type: "leaf",
		summary: "stdio & streamable HTTP.",
		bullets: [
			"Any MCP server reachable this way exposes extra tools to the agent.",
			"Reached through the Coding Agent's MCP client.",
		],
	},
	"ext-searxng": {
		title: "SearXNG",
		kicker: "external/searxng",
		color: "#64748b",
		type: "leaf",
		summary: "Self-hosted web search.",
		bullets: [
			"Backs the web_search / web_fetch tools.",
			"No dependency on a third-party search API.",
		],
	},
	"ext-fs": {
		title: "Filesystem + Git",
		kicker: "external/fs-git",
		color: "#64748b",
		type: "leaf",
		summary: "Direct access to the local machine.",
		bullets: [
			"bash, grep (rg), find (fd), and git.",
			"The most direct path from a tool call to the outside world.",
		],
	},
};

/* ============================== STATE ============================== */
let path = ["home"];

function node(id) {
	return NODES[id];
}
function current() {
	return node(path[path.length - 1]);
}
function parentOf(id) {
	for (const k in NODES) {
		if (NODES[k].children?.includes(id)) return k;
	}
	return null;
}

/* ============================== BOOT ============================== */
const bootEl = document.getElementById("boot");
const bootLine = document.getElementById("bootline");
const bootText = "> booting logician_";
let bootDone = false;
function finishBoot() {
	if (bootDone) return;
	bootDone = true;
	bootEl.style.transition = "opacity .35s";
	bootEl.style.opacity = "0";
	setTimeout(() => {
		bootEl.style.display = "none";
		renderView();
	}, 360);
}

// load SVG early so it's ready when home is rendered
const SVG_URL = "logician-explorer.svg";
let HOME_SVG = null;
async function loadSVG() {
	try {
		const res = await fetch(SVG_URL);
		HOME_SVG = await res.text();
	} catch (e) {
		console.warn("Failed to load SVG:", e);
		HOME_SVG = "";
	}
}
loadSVG();
(function typeBoot() {
	let i = 0;
	const t = setInterval(() => {
		i++;
		bootLine.textContent = bootText.slice(0, i);
		if (i >= bootText.length) {
			clearInterval(t);
			setTimeout(finishBoot, 700);
		}
	}, 45);
})();
bootEl.addEventListener("click", finishBoot);
window.addEventListener(
	"keydown",
	_e => {
		if (!bootDone) {
			finishBoot();
		}
	},
	{ once: false },
);

/* ============================== RENDER ============================== */
const viewEl = document.getElementById("view");
const crumbEl = document.getElementById("breadcrumb");

function fullPathParts() {
	return path.map(id => {
		const n = node(id);
		if (n.path) return n.path;
		const k = n.kicker || "";
		return k.split("/").pop().split("  ").pop();
	});
}

function renderBreadcrumb() {
	const parts = fullPathParts();
	let html = `<span class="seg" data-jump="0">guest@logician</span><span class="sep">:</span>`;
	parts.forEach((p, i) => {
		const isLast = i === parts.length - 1;
		html += `<span class="sep">/</span><span class="seg${isLast ? " current" : ""}" data-jump="${i}">${p}</span>`;
	});
	html += `<span class="sep">$</span><span class="cursor"></span>`;
	crumbEl.innerHTML = html;
	crumbEl.querySelectorAll(".seg[data-jump]").forEach(el => {
		el.addEventListener("click", () => {
			const idx = parseInt(el.getAttribute("data-jump"), 10);
			if (idx < path.length - 1) jumpTo(idx);
		});
	});
}

function _escXml(s) {
	return String(s)
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;");
}

function findEl(id) {
	return (
		viewEl.querySelector(`[data-nav="${id}"]`) ||
		viewEl.querySelector(`[data-id="${id}"]`)
	);
}

/* ---------- generic sub-architecture: hub + spoke ---------- */
function renderSubArch(n) {
	const items = n.children.map(id => ({ id, ...node(id) }));
	const nodesHTML = items
		.map(
			it => `
    <button class="subnode" data-nav="${it.id}" tabindex="0" style="border-color:${it.color}88">
      <span class="sn-title" style="color:${it.color}">${it.title}</span>
      <span class="sn-desc">${it.summary}</span>
      <span class="sn-hint" style="color:${it.color}">view &rarr;</span>
    </button>`,
		)
		.join("");
	return `
    <div class="eyebrow" style="color:${n.color}">${n.kicker}</div>
    <h1 class="title">${n.title}</h1>
    <p class="summary">${n.summary}</p>
    <div class="subarch-wrap" id="subarch-wrap">
      <div class="subarch-hub"><span class="dot" style="background:${n.color}"></span>${n.kicker}</div>
      <svg class="connector-svg"></svg>
      <div class="subarch-row">${nodesHTML}</div>
    </div>
  `;
}

function drawConnectors(color) {
	const wrap = document.getElementById("subarch-wrap");
	if (!wrap) return;
	const svg = wrap.querySelector(".connector-svg");
	const wrapRect = wrap.getBoundingClientRect();
	const dot = wrap.querySelector(".subarch-hub .dot");
	const dotRect = dot.getBoundingClientRect();
	const hubX = dotRect.left + dotRect.width / 2 - wrapRect.left;
	const hubY = dotRect.top + dotRect.height / 2 - wrapRect.top;
	const trunkY = hubY + 24;
	const boxes = wrap.querySelectorAll(".subnode");
	const centers = [];
	boxes.forEach(b => {
		const r = b.getBoundingClientRect();
		centers.push({
			cx: r.left + r.width / 2 - wrapRect.left,
			topY: r.top - wrapRect.top,
		});
	});
	let s = "";
	if (centers.length > 1) {
		s += `<line x1="${centers[0].cx}" y1="${trunkY}" x2="${centers[centers.length - 1].cx}" y2="${trunkY}" stroke="${color}" stroke-width="1.4" stroke-opacity="0.45"/>`;
	}
	centers.forEach(c => {
		s += `<line x1="${c.cx}" y1="${trunkY}" x2="${c.cx}" y2="${c.topY - 3}" stroke="${color}" stroke-width="1.4" stroke-opacity="0.45"/>`;
	});
	s += `<line x1="${hubX}" y1="${hubY + 5}" x2="${hubX}" y2="${trunkY}" stroke="${color}" stroke-width="1.4" stroke-opacity="0.45"/>`;
	svg.innerHTML = s;
}

function badgeShape(stype) {
	return stype === "gate" ? "diamond-shape" : "";
}
function stageColor(stype) {
	return stype === "tool"
		? "#34d399"
		: stype === "mem"
			? "#22d3ee"
			: stype === "gate"
				? "#06b6d4"
				: "#22d3ee";
}

function renderPipeline(n) {
	let rows = "";
	n.children.forEach((id, i) => {
		const s = node(id);
		const col = stageColor(s.stype);
		rows += `
      <div class="prow" data-id="${id}" tabindex="0" style="border-color:${col}55">
        <span class="badge ${badgeShape(s.stype)}" style="border-color:${col}; color:${col}">${s.n}</span>
        <span class="ptext">
          <div class="ptitle" style="color:${col}">${s.title}</div>
          <div class="pdesc">${s.summary}</div>
        </span>
      </div>`;
		if (i === 4) {
			rows += `<div class="loop-back-note">5 &rarr; 1 on no tool calls: turn complete, await next input (outer loop)</div>`;
		}
	});
	rows += `<div class="loop-note">10 &rarr; 4 while tool calls keep coming (inner loop) &mdash; then Message Queue waits for the next input</div>`;
	return `
    <div class="eyebrow" style="color:${n.color}">${n.kicker}</div>
    <h1 class="title">${n.title}</h1>
    <p class="summary">${n.summary}</p>
    <div class="pipeline">${rows}</div>
  `;
}

function renderHomeDiagram(n) {
	const svgContent =
		HOME_SVG ||
		`<div style="padding:40px;color:var(--dim);font-family:var(--mono)">loading diagram…</div>`;
	return `
    <div class="eyebrow" style="color:${n.color}">${n.kicker}</div>
    <h1 class="title">${n.title}</h1>
    <p class="summary">${n.summary}</p>
    <div id="diagram-wrap">${svgContent}</div>
  `;
}

function renderView() {
	const n = current();
	renderBreadcrumb();
	hideTooltip();
	currentSubarchColor = null;
	if (n.type === "diagram") {
		viewEl.innerHTML = renderHomeDiagram(n);
	} else if (n.type === "grid") {
		viewEl.innerHTML = renderSubArch(n);
	} else {
		viewEl.innerHTML = renderPipeline(n);
	}
	viewEl.scrollTop = 0;
	attachHandlers();
	if (n.type === "grid") {
		currentSubarchColor = n.color;
		requestAnimationFrame(() => drawConnectors(n.color));
		setTimeout(() => drawConnectors(n.color), 80);
	}
}

let currentSubarchColor = null;
window.addEventListener("resize", () => {
	if (currentSubarchColor) drawConnectors(currentSubarchColor);
});

function attachHandlers() {
	viewEl.querySelectorAll(".prow, .subnode").forEach(el => {
		const id = el.getAttribute("data-id") || el.getAttribute("data-nav");
		el.addEventListener("click", () => activate(id, el));
		el.addEventListener("keydown", e => {
			if (e.key === "Enter" || e.key === " ") {
				e.preventDefault();
				activate(id, el);
			}
		});
	});
	// Make tooltip related chips clickable
	ttEl.querySelectorAll(".chip").forEach(chip => {
		chip.addEventListener("click", () => jumpToRelated(chip));
		chip.addEventListener("keydown", e => {
			if (e.key === "Enter" || e.key === " ") {
				e.preventDefault();
				jumpToRelated(chip);
			}
		});
	});
	const svg = document.getElementById("home-svg");
	if (svg) {
		svg.addEventListener("click", e => {
			const el = e.target.closest("[data-nav]");
			if (el) activate(el.getAttribute("data-nav"), el);
		});
		svg.querySelectorAll("[data-nav]").forEach(el => {
			el.setAttribute("tabindex", "0");
			el.addEventListener("keydown", e => {
				if (e.key === "Enter" || e.key === " ") {
					e.preventDefault();
					activate(el.getAttribute("data-nav"), el);
				}
			});
		});
	}
}

/* ============================== TOOLTIP (innermost level) ============================== */
const ttEl = document.getElementById("tooltip");
let activeTooltip = null;

function tooltipContentHTML(n, id) {
	const bullets = (n.bullets || []).map(b => `<li>${b}</li>`).join("");
	const related = (n.related || [])
		.map(
			([label, color]) =>
				`<span class="chip" style="border-color:${color}66; color:${color}">${label}</span>`,
		)
		.join("");
	let dots = "";
	const pid = parentOf(id);
	if (pid) {
		const sibs = node(pid).children;
		dots = `<div class="tt-dots">${sibs.map(sid => `<span class="${sid === id ? "active" : ""}"></span>`).join("")}</div>`;
	}
	return `
    <div class="tt-kicker">${n.kicker}</div>
    <div class="tt-title">${n.title}</div>
    <div class="tt-summary">${n.summary}</div>
    <ul>${bullets}</ul>
    ${related ? `<div class="tt-related">${related}</div>` : ""}
    <div class="tt-nav"><span>esc to close &nbsp;\u00b7&nbsp; &larr;/&rarr; siblings</span>${dots}</div>
  `;
}

function showTooltip(id, el) {
	if (!el) return;
	const n = node(id);
	ttEl.style.visibility = "hidden";
	ttEl.classList.remove("show");
	ttEl.style.setProperty("--tt-accent", n.color);
	activeTooltip = { id, el };
	ttEl.innerHTML = tooltipContentHTML(n, id);
	ttEl.style.left = "0px";
	ttEl.style.top = "0px";
	const r = el.getBoundingClientRect();
	const tr = ttEl.getBoundingClientRect();
	let left = r.left + r.width / 2 - tr.width / 2;
	left = Math.max(12, Math.min(left, window.innerWidth - tr.width - 12));
	let top = r.bottom + 14;
	if (top + tr.height > window.innerHeight - 12) {
		top = r.top - tr.height - 14;
	}
	top = Math.max(60, top);
	ttEl.style.left = `${left}px`;
	ttEl.style.top = `${top}px`;
	ttEl.style.visibility = "visible";
	requestAnimationFrame(() => ttEl.classList.add("show"));
}

function hideTooltip() {
	if (!activeTooltip) return;
	ttEl.classList.remove("show");
	activeTooltip = null;
}

function tooltipSibling(dir) {
	if (!activeTooltip) return false;
	const pid = parentOf(activeTooltip.id);
	if (!pid) return true;
	const sibs = node(pid).children;
	const idx = sibs.indexOf(activeTooltip.id);
	const next = idx + dir;
	if (next < 0 || next >= sibs.length) return true;
	const nextId = sibs[next];
	const el = findEl(nextId);
	if (el) showTooltip(nextId, el);
	return true;
}

/* ============================== ACTIVATE (leaf -> tooltip, else -> navigate) ============================== */
function activate(id, el) {
	const n = node(id);
	if (n.type === "leaf") {
		showTooltip(id, el);
	} else {
		hideTooltip();
		enter(id, el);
	}
}

/* ============================== TRANSITIONS ============================== */
const flipEl = document.getElementById("flip");

function zoomIn(originEl, color, onDone) {
	const r = originEl.getBoundingClientRect();
	flipEl.style.transition = "none";
	flipEl.style.left = `${r.left}px`;
	flipEl.style.top = `${r.top}px`;
	flipEl.style.width = `${r.width}px`;
	flipEl.style.height = `${r.height}px`;
	flipEl.style.background = color;
	flipEl.style.borderColor = color;
	flipEl.style.opacity = "0.18";
	viewEl.style.transition = "opacity .18s";
	viewEl.style.opacity = "0";
	requestAnimationFrame(() => {
		flipEl.style.transition = "all .32s cubic-bezier(.2,.7,.3,1)";
		flipEl.style.left = "0px";
		flipEl.style.top = "52px";
		flipEl.style.width = `${window.innerWidth}px`;
		flipEl.style.height = `${window.innerHeight - 52}px`;
		flipEl.style.opacity = "0.30";
	});
	setTimeout(() => {
		onDone();
		viewEl.style.opacity = "1";
		flipEl.style.transition = "opacity .25s";
		flipEl.style.opacity = "0";
	}, 320);
}

function zoomOut(color, onDone) {
	flipEl.style.transition = "none";
	flipEl.style.left = "0px";
	flipEl.style.top = "52px";
	flipEl.style.width = `${window.innerWidth}px`;
	flipEl.style.height = `${window.innerHeight - 52}px`;
	flipEl.style.background = color;
	flipEl.style.borderColor = color;
	flipEl.style.opacity = "0.22";
	viewEl.style.transition = "opacity .16s, transform .16s";
	viewEl.style.opacity = "0";
	viewEl.style.transform = "scale(0.98)";
	requestAnimationFrame(() => {
		flipEl.style.transition = "opacity .28s";
		flipEl.style.opacity = "0";
	});
	setTimeout(() => {
		onDone();
		viewEl.style.transform = "scale(1.02)";
		requestAnimationFrame(() => {
			viewEl.style.transition = "opacity .2s, transform .2s";
			viewEl.style.opacity = "1";
			viewEl.style.transform = "scale(1)";
		});
	}, 170);
}

function _slide(dir, onDone) {
	viewEl.style.transition = "opacity .15s, transform .15s";
	viewEl.style.opacity = "0";
	viewEl.style.transform = `translateX(${dir * 18}px)`;
	setTimeout(() => {
		onDone();
		viewEl.style.transform = `translateX(${dir * -18}px)`;
		requestAnimationFrame(() => {
			viewEl.style.transition = "opacity .18s, transform .18s";
			viewEl.style.opacity = "1";
			viewEl.style.transform = "translateX(0)";
		});
	}, 160);
}

/* ============================== NAVIGATION ============================== */
function enter(id, originEl) {
	const color = node(id).color;
	const doPush = () => {
		path.push(id);
		renderView();
	};
	if (originEl) {
		zoomIn(originEl, color, doPush);
	} else {
		doPush();
	}
}

function back() {
	if (path.length <= 1) return;
	const leavingColor = current().color;
	zoomOut(leavingColor, () => {
		path.pop();
		renderView();
	});
}

function jumpTo(index) {
	if (index >= path.length - 1) return;
	const leavingColor = current().color;
	zoomOut(leavingColor, () => {
		path = path.slice(0, index + 1);
		renderView();
	});
}

function jumpDigit(n) {
	const cn = current();
	const idx = n - 1;
	if (!cn.children || idx < 0 || idx >= cn.children.length) return;
	const id = cn.children[idx];
	const el =
		findEl(id) || document.querySelector(`#home-svg [data-nav="${id}"]`);
	activate(id, el);
}

function jumpToRelated(chipEl) {
	const label = chipEl.textContent.trim();
	for (const id in NODES) {
		const n = NODES[id];
		if (n.related) {
			for (const [rLabel] of n.related) {
				if (rLabel === label) {
					const el = findEl(id);
					if (el) {
						activate(id, el);
						return;
					}
				}
			}
		}
	}
}

/* ============================== SEARCH OVERLAY ============================== */
const searchOverlay = document.createElement("div");
searchOverlay.id = "search-overlay";
searchOverlay.innerHTML = `
  <div>
    <div class="search-header">
      <span class="search-icon">\uD83D\uDD0D</span>
      <input id="search-input" type="text" placeholder="Search nodes..." autocomplete="off" spellcheck="false" />
      <kbd>ESC</kbd>
    </div>
    <div id="search-results"></div>
  </div>`;
document.body.appendChild(searchOverlay);

let searchOpen = false;
function openSearch() {
	searchOpen = true;
	searchOverlay.classList.add("show");
	const inp = document.getElementById("search-input");
	if (inp) {
		inp.value = "";
		inp.focus();
	}
	document.getElementById("search-results").innerHTML =
		"<div style='padding:20px;text-align:center;color:var(--dim);font-family:var(--mono);font-size:12px;'>start typing to search...</div>";
}
function closeSearch() {
	searchOpen = false;
	searchOverlay.classList.remove("show");
}

function doSearch(query) {
	const results = document.getElementById("search-results");
	if (!query || query.length < 2) {
		results.innerHTML =
			"<div style='padding:20px;text-align:center;color:var(--dim);font-family:var(--mono);font-size:12px;'>type at least 2 chars...</div>";
		return;
	}
	const q = query.toLowerCase();
	const matches = [];
	for (const id in NODES) {
		const n = NODES[id];
		const searchable =
			`${n.title} ${n.summary} ${n.kicker || ""} ${id}`.toLowerCase();
		if (searchable.includes(q)) {
			matches.push({
				id,
				score: n.title.toLowerCase().startsWith(q) ? 0 : 1,
				...n,
			});
		}
	}
	matches.sort((a, b) => a.score - b.score);
	if (matches.length === 0) {
		results.innerHTML =
			"<div style='padding:20px;text-align:center;color:var(--dim);font-family:var(--mono);font-size:12px;'>no matches found</div>";
		return;
	}
	results.innerHTML = matches
		.slice(0, 10)
		.map(
			m => `
    <div class="search-item" data-id="${m.id}" style="display:flex;align-items:center;gap:12px;padding:10px 14px;border-radius:8px;cursor:pointer;transition:background 0.1s;" onmouseenter="this.style.background='var(--surface-2)'" onmouseleave="this.style.background='transparent'">
      <span style="width:8px;height:8px;border-radius:50%;background:${m.color};flex:none;"></span>
      <div style="flex:1;min-width:0;">
        <div style="font-family:var(--mono);font-size:13px;font-weight:600;color:var(--text);">${m.title}</div>
        <div style="font-size:11px;color:var(--dim);margin-top:2px;">${m.path || m.kicker?.split("/").pop() || m.id}</div>
      </div>
    </div>`,
		)
		.join("");
	results.querySelectorAll(".search-item").forEach(el => {
		el.addEventListener("click", () => {
			const id = el.getAttribute("data-id");
			closeSearch();
			const el2 = findEl(id);
			activate(id, el2);
		});
	});
}

/* ============================== KEYBOARD ============================== */
const helpEl = document.getElementById("help");
window.addEventListener("keydown", e => {
	if (!bootDone) return;
	if (e.key === "?") {
		helpEl.classList.toggle("show");
		return;
	}
	if (helpEl.classList.contains("show") && e.key === "Escape") {
		helpEl.classList.remove("show");
		return;
	}
	if (e.key === "Escape" || e.key === "Backspace") {
		e.preventDefault();
		if (activeTooltip) {
			hideTooltip();
		} else {
			back();
		}
		return;
	}
	if (e.key === "ArrowRight") {
		if (!tooltipSibling(1)) return;
		return;
	}
	if (e.key === "ArrowLeft") {
		if (!tooltipSibling(-1)) return;
		return;
	}
	if (e.key === "Home") {
		hideTooltip();
		path = ["home"];
		renderView();
		return;
	}
	if (/^[1-9]$/.test(e.key)) {
		jumpDigit(parseInt(e.key, 10));
		return;
	}
	if ((e.ctrlKey || e.metaKey) && e.key === "k") {
		e.preventDefault();
		if (searchOpen) {
			closeSearch();
		} else {
			openSearch();
		}
		return;
	}
	if (e.key === "Escape" && searchOpen) {
		closeSearch();
		return;
	}
});

// Search input handler
document.addEventListener("DOMContentLoaded", () => {
	const inp = document.getElementById("search-input");
	if (inp) {
		inp.addEventListener("input", e => doSearch(e.target.value));
	}
});
