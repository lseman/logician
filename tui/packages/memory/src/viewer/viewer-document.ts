// ── @logician/memory — Dashboard HTML Document ───────────────────────────────
// Single-page app with all tabs, CSS, and client-side JS.
// Self-hosted, zero dependencies.

const HTML = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Logician Memory</title>
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <style>
    :root {
      color-scheme: dark;
      /* Midnight slate foundation with cyan primary and violet secondary. */
      --ink: #080B12;
      --surface: #0E1420;
      --surface-raised: #141C2A;
      --surface-hover: #1A2535;
      --border: rgba(148, 163, 184, 0.18);
      --border-soft: rgba(148, 163, 184, 0.09);
      --text: #F1F5F9;
      --text-dim: #A3B1C2;
      --text-faint: #697A8F;
      --accent: #67E8F9;
      --accent-dim: #22D3EE;
      --accent-contrast: #071018;
      --accent-glow: rgba(34, 211, 238, 0.16);
      --violet-glow: rgba(167, 139, 250, 0.13);
      --amber: #FBBF24;
      --coral: #FB7185;
      --sage: #4ADE80;
      --violet: #A78BFA;
      --slate: #60A5FA;
      --font-display: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      --font-ui: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      --font-mono: 'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace;
      --radius: 14px;
      --radius-sm: 10px;
      --shadow-card: 0 1px 2px rgba(0,0,0,0.28), 0 14px 36px -18px rgba(0,0,0,0.72);
      --shadow-glow: 0 0 0 1px var(--accent-glow), 0 0 28px -8px var(--accent-glow);
    }
    :root[data-theme="light"] {
      color-scheme: light;
      --ink: #F3F6FA;
      --surface: #FFFFFF;
      --surface-raised: #F8FAFC;
      --surface-hover: #EEF4FA;
      --border: #CBD5E1;
      --border-soft: #E2E8F0;
      --text: #0F172A;
      --text-dim: #475569;
      --text-faint: #718096;
      --accent: #087E8B;
      --accent-dim: #066875;
      --accent-contrast: #FFFFFF;
      --accent-glow: rgba(8, 126, 139, 0.12);
      --violet-glow: rgba(109, 40, 217, 0.09);
      --amber: #9A6700;
      --coral: #BE3455;
      --sage: #16805A;
      --violet: #6D28D9;
      --slate: #2563B8;
      --shadow-card: 0 1px 2px rgba(15,23,42,0.04), 0 12px 28px -18px rgba(15,23,42,0.24);
      --shadow-glow: 0 0 0 1px var(--accent-glow), 0 0 24px -8px var(--accent-glow);
    }
    @media (prefers-color-scheme: light) {
      :root:not([data-theme="dark"]) {
        color-scheme: light;
        --ink: #F3F6FA;
        --surface: #FFFFFF;
        --surface-raised: #F8FAFC;
        --surface-hover: #EEF4FA;
        --border: #CBD5E1;
        --border-soft: #E2E8F0;
        --text: #0F172A;
        --text-dim: #475569;
        --text-faint: #718096;
        --accent: #087E8B;
        --accent-dim: #066875;
        --accent-contrast: #FFFFFF;
        --accent-glow: rgba(8, 126, 139, 0.12);
        --violet-glow: rgba(109, 40, 217, 0.09);
        --amber: #9A6700;
        --coral: #BE3455;
        --sage: #16805A;
        --violet: #6D28D9;
        --slate: #2563B8;
        --shadow-card: 0 1px 2px rgba(15,23,42,0.04), 0 12px 28px -18px rgba(15,23,42,0.24);
        --shadow-glow: 0 0 0 1px var(--accent-glow), 0 0 24px -8px var(--accent-glow);
      }
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      font-family: var(--font-ui); background: var(--ink); color: var(--text-dim);
      line-height: 1.55; overflow: hidden; display: flex; -webkit-font-smoothing: antialiased;
    }
    ::selection { background: color-mix(in srgb, var(--accent) 28%, transparent); color: var(--text); }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-faint); }
    :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 4px; }
    @media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration: 0.001ms !important; transition-duration: 0.001ms !important; } }

    /* ── Shell ─────────────────────────────────────────────────────────── */
    .shell { display: flex; width: 100%; height: 100%; }
    .sidebar {
      width: 236px; flex: 0 0 auto;
      background: linear-gradient(180deg, color-mix(in srgb, var(--surface-raised) 72%, var(--surface)), var(--surface));
      border-right: 1px solid var(--border);
      display: flex; flex-direction: column; padding: 18px 12px;
    }
    .brand { display: flex; align-items: center; gap: 10px; padding: 4px 8px 18px; cursor: pointer; }
    .brand .mark {
      width: 28px; height: 28px; border-radius: 8px; flex: 0 0 auto;
      background: radial-gradient(circle at 30% 30%, var(--accent), var(--accent-dim));
      box-shadow: var(--shadow-glow); position: relative;
    }
    .brand .mark::after {
      content: ''; position: absolute; inset: 0; border-radius: 8px;
      background: radial-gradient(circle at 65% 70%, rgba(255,255,255,0.35), transparent 55%);
    }
    .brand-text h1 { font-family: var(--font-display); font-size: 14.5px; font-weight: 650; color: var(--text); letter-spacing: -0.01em; }
    .brand-text .tag { font-size: 10px; color: var(--text-faint); font-family: var(--font-mono); text-transform: uppercase; letter-spacing: 0.08em; }
    .pulse-strip {
      display: flex; align-items: center; gap: 8px; padding: 9px 10px; margin-bottom: 16px;
      background: var(--surface-raised); border: 1px solid var(--border-soft); border-radius: var(--radius-sm);
    }
    .pulse-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--text-faint); flex: 0 0 auto; }
    .pulse-dot.live { background: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow); animation: pulse 2s ease-in-out infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.45; } }
    .pulse-label { font-size: 11px; font-weight: 600; color: var(--text-dim); flex: 1; }
    .pulse-label.live { color: var(--accent); }

    .nav { display: flex; flex-direction: column; gap: 1px; flex: 1 1 auto; overflow-y: auto; }
    .nav-group-label { font-size: 9.5px; color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700; padding: 14px 10px 6px; }
    .nav-group-label:first-child { padding-top: 4px; }
    .nav button {
      display: flex; align-items: center; gap: 10px; width: 100%; background: none; border: none;
      color: var(--text-dim); padding: 8px 10px; font-size: 12.5px; font-weight: 500; text-align: left;
      cursor: pointer; border-radius: var(--radius-sm); font-family: var(--font-ui); transition: background 0.12s, color 0.12s;
    }
    .nav button .ic { width: 15px; text-align: center; opacity: 0.85; font-size: 13px; flex: 0 0 auto; }
    .nav button:hover { background: var(--surface-hover); color: var(--text); }
    .nav button.active {
      background: linear-gradient(90deg, var(--accent-glow), color-mix(in srgb, var(--accent-glow) 22%, transparent));
      color: var(--accent); font-weight: 650; box-shadow: inset 2px 0 0 var(--accent);
    }
    .nav button:disabled { opacity: 0.35; pointer-events: none; }
    .nav button .soon { margin-left: auto; font-size: 8.5px; color: var(--text-faint); font-family: var(--font-mono); text-transform: uppercase; letter-spacing: 0.06em; }

    .sidebar-foot { padding-top: 12px; margin-top: 8px; border-top: 1px solid var(--border-soft); display: flex; align-items: center; gap: 8px; }
    .icon-btn {
      background: none; border: 1px solid var(--border); color: var(--text-dim); width: 30px; height: 30px;
      border-radius: var(--radius-sm); cursor: pointer; display: flex; align-items: center; justify-content: center;
      font-size: 13px; transition: border-color 0.12s, color 0.12s;
    }
    .icon-btn:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-glow); }
    .dateline { font-size: 10.5px; color: var(--text-faint); font-family: var(--font-mono); flex: 1; }

    .main { flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; overflow: hidden; }
    .topbar {
      height: 56px; flex: 0 0 auto; display: flex; align-items: center; justify-content: space-between;
      padding: 0 28px; border-bottom: 1px solid var(--border);
      background: color-mix(in srgb, var(--ink) 82%, transparent); backdrop-filter: blur(14px);
    }
    .topbar h2 { font-family: var(--font-display); font-size: 16px; font-weight: 650; color: var(--text); letter-spacing: -0.01em; }
    .topbar .sub { font-size: 11.5px; color: var(--text-faint); margin-top: 1px; font-family: var(--font-mono); }
    .workspace-chip {
      font-size: 11px; color: var(--text-dim); font-family: var(--font-mono); background: var(--surface-raised);
      border: 1px solid var(--border-soft); padding: 5px 11px; border-radius: 999px; max-width: 420px;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .view { display: none; flex: 1 1 auto; min-height: 0; overflow-y: auto; padding: 24px 28px 40px; }
    .view.active { display: block; }

    /* ── Primitives ────────────────────────────────────────────────────── */
    .stats-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 20px; }
    .stat-card {
      background: linear-gradient(145deg, var(--surface-raised), var(--surface)); border: 1px solid var(--border-soft); border-radius: var(--radius);
      padding: 16px 18px; box-shadow: var(--shadow-card); position: relative; overflow: hidden;
    }
    .stat-card .label { font-size: 10px; color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px; font-weight: 700; }
    .stat-card .value { font-size: 30px; font-weight: 650; color: var(--text); font-family: var(--font-display); line-height: 1; font-variant-numeric: tabular-nums; }
    .stat-card .sub { font-size: 11px; color: var(--text-faint); margin-top: 5px; font-family: var(--font-mono); }
    .stat-card.accent { border-color: color-mix(in srgb, var(--accent) 42%, var(--border-soft)); box-shadow: var(--shadow-glow); }
    .stat-card.accent .value { color: var(--accent); }

    .card {
      background: var(--surface); border: 1px solid var(--border-soft); border-radius: var(--radius);
      padding: 18px 20px; margin-bottom: 14px; box-shadow: var(--shadow-card);
    }
    .card-title {
      font-size: 11.5px; font-weight: 700; color: var(--text); margin-bottom: 14px; letter-spacing: 0.02em;
      display: flex; align-items: center; justify-content: space-between;
    }
    .card-title .hint { font-size: 10.5px; color: var(--text-faint); font-weight: 500; font-family: var(--font-mono); }

    .health-bar { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
    .gauge-bar { flex: 1; height: 6px; background: var(--border-soft); overflow: hidden; border-radius: 6px; }
    .gauge-fill { height: 100%; border-radius: 6px; transition: width 0.6s cubic-bezier(0.16,1,0.3,1); }
    .gauge-label { width: 78px; font-size: 10.5px; color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
    .gauge-value { width: 64px; font-size: 11px; color: var(--text-dim); text-align: right; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }

    .badge {
      display: inline-flex; align-items: center; gap: 4px; font-size: 10px; padding: 3px 9px; font-weight: 600;
      border: 1px solid transparent; border-radius: 999px; letter-spacing: 0.02em; line-height: 1.4; white-space: nowrap;
    }
    .badge-teal { background: color-mix(in srgb, var(--accent) 14%, transparent); border-color: color-mix(in srgb, var(--accent) 25%, transparent); color: var(--accent); }
    .badge-sage { background: color-mix(in srgb, var(--sage) 14%, transparent); border-color: color-mix(in srgb, var(--sage) 24%, transparent); color: var(--sage); }
    .badge-amber { background: color-mix(in srgb, var(--amber) 14%, transparent); border-color: color-mix(in srgb, var(--amber) 24%, transparent); color: var(--amber); }
    .badge-coral { background: color-mix(in srgb, var(--coral) 14%, transparent); border-color: color-mix(in srgb, var(--coral) 24%, transparent); color: var(--coral); }
    .badge-violet { background: color-mix(in srgb, var(--violet) 14%, transparent); border-color: color-mix(in srgb, var(--violet) 24%, transparent); color: var(--violet); }
    .badge-slate { background: color-mix(in srgb, var(--slate) 14%, transparent); border-color: color-mix(in srgb, var(--slate) 24%, transparent); color: var(--slate); }
    .badge-muted { background: var(--border-soft); color: var(--text-faint); }

    table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
    th {
      text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); color: var(--text-faint);
      font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; position: sticky; top: 0; background: var(--surface);
    }
    td { padding: 10px 12px; border-bottom: 1px solid var(--border-soft); vertical-align: top; color: var(--text-dim); }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: var(--surface-hover); }

    .toolbar { display: flex; gap: 8px; margin-bottom: 18px; align-items: center; flex-wrap: wrap; }
    .toolbar input, .toolbar select {
      background: var(--surface-raised); border: 1px solid var(--border); color: var(--text); padding: 8px 12px;
      font-size: 12.5px; outline: none; font-family: var(--font-ui); border-radius: var(--radius-sm); transition: border-color 0.12s;
    }
    .toolbar input:focus, .toolbar select:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow); }
    .toolbar input { flex: 1; min-width: 200px; }
    .toolbar input::placeholder { color: var(--text-faint); }
    .btn {
      background: var(--surface); border: 1px solid var(--border); color: var(--text-dim); padding: 8px 14px;
      font-size: 11.5px; cursor: pointer; font-family: var(--font-ui); font-weight: 600; border-radius: var(--radius-sm);
      transition: border-color 0.12s, color 0.12s, background 0.12s;
    }
    .btn:hover { border-color: var(--accent); color: var(--text); background: var(--surface-hover); }
    .btn-primary { background: var(--accent); color: var(--accent-contrast); border-color: var(--accent); box-shadow: 0 6px 18px -9px var(--accent); }
    .btn-primary:hover { background: var(--accent-dim); border-color: var(--accent-dim); color: var(--accent-contrast); }
    .btn-danger { border-color: color-mix(in srgb, var(--coral) 55%, var(--border)); color: var(--coral); }

    .entry-card {
      background: var(--surface); border: 1px solid var(--border-soft); border-radius: var(--radius);
      padding: 15px 18px; margin-bottom: 10px; border-left: 3px solid var(--border);
      transition: border-color 0.12s, box-shadow 0.12s;
    }
    .entry-card:hover { border-color: var(--border); background: var(--surface-raised); box-shadow: var(--shadow-card); }
    .entry-card.expandable-entry { cursor: pointer; }
    .entry-card.expandable-entry:focus-visible { border-color: var(--accent); box-shadow: var(--shadow-glow); }
    .entry-full { display: none; }
    .entry-card.expanded .entry-preview { display: none; }
    .entry-card.expanded .entry-full { display: block; }
    .entry-expand-hint {
      display: flex; align-items: center; gap: 6px; margin-top: 10px; color: var(--accent);
      font: 600 10px/1.4 var(--font-mono); letter-spacing: 0.02em;
    }
    .entry-expand-hint::before { content: '›'; font-size: 14px; line-height: 1; transition: transform 0.15s ease; }
    .entry-card.expanded .entry-expand-hint::before { transform: rotate(90deg); }
    .entry-card.expanded .entry-expand-hint .expand-collapsed,
    .entry-card:not(.expanded) .entry-expand-hint .expand-expanded { display: none; }
    .entry-card.imp-high { border-left-color: var(--coral); }
    .entry-card.imp-med { border-left-color: var(--amber); }
    .entry-card.imp-low { border-left-color: var(--sage); }
    .entry-head { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .entry-title { font-size: 13.5px; font-weight: 600; color: var(--text); margin-top: 8px; letter-spacing: -0.005em; }
    .entry-meta { font-size: 10.5px; color: var(--text-faint); font-family: var(--font-mono); margin-top: 3px; }
    .entry-body { font-size: 12.5px; color: var(--text-dim); margin-top: 8px; word-break: break-word; white-space: pre-wrap; }

    .session-item {
      background: var(--surface); border: 1px solid var(--border-soft); border-radius: var(--radius);
      padding: 13px 16px; cursor: pointer; margin-bottom: 6px; transition: border-color 0.12s, background 0.12s;
    }
    .session-item:hover { background: var(--surface-hover); }
    .session-item.selected { border-color: var(--accent); box-shadow: var(--shadow-glow); }
    .session-item .session-project { font-weight: 600; color: var(--text); font-size: 13px; }
    .session-item .session-meta { font-size: 10.5px; color: var(--text-faint); font-family: var(--font-mono); margin-top: 3px; }

    .detail-panel { background: var(--surface); border: 1px solid var(--border-soft); border-radius: var(--radius); padding: 22px; margin-top: 0; box-shadow: var(--shadow-card); }
    .detail-panel h3 { font-size: 13.5px; font-weight: 700; color: var(--text); margin-bottom: 14px; letter-spacing: 0.01em; padding-bottom: 12px; border-bottom: 1px solid var(--border-soft); }
    .detail-row { display: flex; padding: 7px 0; font-size: 12.5px; border-bottom: 1px solid var(--border-soft); gap: 12px; }
    .detail-row:last-child { border-bottom: none; }
    .detail-row .dl { color: var(--text-faint); width: 116px; flex-shrink: 0; font-size: 10px; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 700; padding-top: 1px; }
    .detail-row .dv { color: var(--text); }

    .bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; font-size: 12px; }
    .bar-label { width: 112px; color: var(--text-dim); font-size: 11.5px; text-transform: capitalize; }
    .bar-track { flex: 1; height: 7px; background: var(--border-soft); overflow: hidden; border-radius: 6px; }
    .bar-fill { height: 100%; border-radius: 6px; transition: width 0.6s cubic-bezier(0.16,1,0.3,1); }
    .bar-value { width: 26px; text-align: right; color: var(--text-faint); font-size: 11px; font-family: var(--font-mono); font-variant-numeric: tabular-nums; }

    .tag {
      font-size: 10px; padding: 2px 8px; border-radius: 999px; background: var(--surface-raised); color: var(--text-dim);
      border: 1px solid var(--border-soft);
      font-family: var(--font-mono); display: inline-block; margin: 2px 3px 0 0;
    }
    .empty-state { text-align: center; padding: 64px 20px; color: var(--text-faint); }
    .empty-state .empty-icon { font-size: 30px; margin-bottom: 12px; opacity: 0.5; }
    .empty-state p { font-size: 13px; }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start; }
    @media (max-width: 860px) { .two-col { grid-template-columns: 1fr; } }
    .placeholder { text-align: center; padding: 90px 20px; color: var(--text-faint); }
    .placeholder .placeholder-icon { font-size: 40px; opacity: 0.4; margin-bottom: 18px; }
    .placeholder h3 { font-size: 16px; color: var(--text-dim); margin-bottom: 8px; font-family: var(--font-display); font-weight: 650; }
    .muted { color: var(--text-faint); font-size: 11.5px; }

    .activity-item { display: flex; gap: 12px; padding: 13px 0; border-bottom: 1px solid var(--border-soft); }
    .activity-item:last-child { border-bottom: none; }
    .activity-dot { width: 7px; height: 7px; border-radius: 50%; margin-top: 6px; flex-shrink: 0; }
    .activity-body { flex: 1; min-width: 0; }
    .activity-title { font-size: 13px; color: var(--text); font-weight: 500; }
    .activity-meta { font-size: 10.5px; color: var(--text-faint); font-family: var(--font-mono); margin-top: 3px; }
    .activity-body p { font-size: 12px; color: var(--text-dim); margin-top: 6px; }

    /* ── Live-arrival flash for new entries ───────────────────────────── */
    @keyframes arrive { from { background: var(--accent-glow); } to { background: transparent; } }
    .arrived { animation: arrive 1.4s ease-out; }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <div class="brand" onclick="switchTab('dashboard')">
        <div class="mark"></div>
        <div class="brand-text">
          <h1>logician</h1>
          <div class="tag">memory</div>
        </div>
      </div>
      <div class="pulse-strip">
        <div class="pulse-dot" id="pulse-dot"></div>
        <div class="pulse-label" id="pulse-label">connecting</div>
      </div>
      <nav class="nav" id="nav">
        <div class="nav-group-label">Overview</div>
        <button data-tab="dashboard" class="active"><span class="ic">&#9673;</span>Dashboard</button>
        <button data-tab="activity"><span class="ic">&#9679;</span>Live Activity</button>
        <div class="nav-group-label">Recall</div>
        <button data-tab="memories"><span class="ic">&#9670;</span>Memories</button>
        <button data-tab="observations"><span class="ic">&#9633;</span>Observations</button>
        <button data-tab="timeline"><span class="ic">&#9702;</span>Timeline</button>
        <div class="nav-group-label">Sessions</div>
        <button data-tab="sessions"><span class="ic">&#9635;</span>Sessions</button>
        <button data-tab="profile"><span class="ic">&#9636;</span>Profile</button>
        <div class="nav-group-label">System</div>
        <button data-tab="working-memory"><span class="ic">&#9642;</span>Working Set</button>
        <button data-tab="audit"><span class="ic">&#9776;</span>Audit Log</button>
        <button data-tab="graph" disabled><span class="ic">&#8982;</span>Graph<span class="soon">soon</span></button>
        <button data-tab="replay" disabled><span class="ic">&#9654;</span>Replay<span class="soon">soon</span></button>
      </nav>
      <div class="sidebar-foot">
        <span class="dateline" id="dateline"></span>
        <button class="icon-btn" id="theme-toggle" title="Toggle theme">&#9789;</button>
      </div>
    </aside>
    <div class="main">
      <div class="topbar">
        <div>
          <h2 id="view-heading">Dashboard</h2>
          <div class="sub" id="view-subheading">Live overview of agent memory</div>
        </div>
        <div class="workspace-chip" id="workspace-chip">—</div>
      </div>

      <div id="view-dashboard" class="view active">
        <div class="stats-row" id="stats-grid">
          <div class="stat-card accent"><div class="label">Sessions</div><div class="value" id="stat-sessions">—</div></div>
          <div class="stat-card accent"><div class="label">Memories</div><div class="value" id="stat-memories">—</div></div>
          <div class="stat-card"><div class="label">Observations</div><div class="value" id="stat-observations">—</div></div>
          <div class="stat-card"><div class="label">Today</div><div class="value" id="stat-observations-today">—</div></div>
        </div>
        <div class="two-col">
          <div>
            <div class="card"><div class="card-title">Process Health</div><div id="health-bars"></div></div>
            <div class="card"><div class="card-title">Memories by Type</div><div id="memories-by-type"></div></div>
          </div>
          <div>
            <div class="card"><div class="card-title">Sessions by Status</div><div id="sessions-by-status"></div></div>
            <div class="card"><div class="card-title">Recent Activity<span class="hint">last 5</span></div><div id="recent-activity"></div></div>
          </div>
        </div>
      </div>

      <div id="view-observations" class="view">
        <div class="toolbar">
          <input id="observation-search" type="text" placeholder="Search observations in this folder…">
          <select id="observation-type"><option value="">All types</option><option value="conversation">Conversation</option><option value="file_read">File read</option><option value="file_write">File write</option><option value="file_edit">File edit</option><option value="command_run">Command</option><option value="search">Search</option><option value="web_fetch">Web fetch</option><option value="error">Error</option><option value="other">Other</option></select>
          <select id="observation-min-importance"><option value="0">Any importance</option><option value="5">5+</option><option value="7">7+</option></select>
          <button class="btn" id="btn-refresh-observations">Refresh</button>
        </div>
        <div id="observations-list"></div>
      </div>

      <div id="view-memories" class="view">
        <div class="toolbar">
          <input id="memory-search" type="text" placeholder="Search memories…">
          <select id="memory-type"><option value="">All types</option><option value="pattern">Pattern</option><option value="preference">Preference</option><option value="architecture">Architecture</option><option value="bug">Bug</option><option value="workflow">Workflow</option><option value="fact">Fact</option></select>
          <select id="memory-min-strength"><option value="">Any strength</option><option value="3">3+</option><option value="5">5+</option><option value="7">7+</option></select>
          <button class="btn" id="btn-refresh-memories">Refresh</button>
        </div>
        <div id="memories-list"></div>
      </div>

      <div id="view-timeline" class="view">
        <div class="toolbar">
          <select id="timeline-session"><option value="">All sessions</option></select>
          <select id="timeline-min-importance"><option value="0">All</option><option value="3">3+</option><option value="5">5+</option><option value="7">7+</option></select>
          <button class="btn" id="btn-refresh-timeline">Refresh</button>
        </div>
        <div id="timeline-container"></div>
      </div>

      <div id="view-sessions" class="view">
        <div class="two-col">
          <div id="sessions-list" style="max-width:380px;"></div>
          <div id="session-detail"></div>
        </div>
      </div>

      <div id="view-audit" class="view">
        <div class="toolbar"><select id="audit-filter"><option value="">All operations</option><option value="create">Create</option><option value="forget">Forget</option></select><button class="btn" id="btn-refresh-audit">Refresh</button></div>
        <div class="card" style="padding:0;overflow:hidden;">
          <table id="audit-table"><thead><tr><th>Time</th><th>Operation</th><th>Resource</th><th>Type</th><th>Strength</th></tr></thead><tbody id="audit-body"></tbody></table>
        </div>
      </div>

      <div id="view-activity" class="view">
        <div class="toolbar"><button class="btn" id="btn-refresh-activity">Refresh</button></div>
        <div class="card"><div id="activity-feed"></div></div>
      </div>

      <div id="view-profile" class="view"><div id="profile-content"></div></div>

      <div id="view-working-memory" class="view">
        <div class="toolbar">
          <select id="wm-tier-filter"><option value="">All tiers</option><option value="hot">Hot</option><option value="warm">Warm</option><option value="cold">Cold</option><option value="archived">Archived</option></select>
          <button class="btn btn-primary" id="btn-auto-tier">Auto-tier now</button>
          <button class="btn" id="btn-refresh-wm">Refresh</button>
        </div>
        <div class="card" style="padding:0;overflow:hidden;">
          <table id="wm-table"><thead><tr><th>Tier</th><th>Type</th><th>Strength</th><th>Content</th></tr></thead><tbody id="wm-body"></tbody></table>
        </div>
      </div>

      <div id="view-graph" class="view">
        <div class="placeholder"><div class="placeholder-icon">&#8982;</div><h3>Knowledge Graph</h3><p>Graph visualization coming soon.</p></div>
      </div>
      <div id="view-replay" class="view">
        <div class="placeholder"><div class="placeholder-icon">&#9654;</div><h3>Session Replay</h3><p>Replay feature coming soon.</p></div>
      </div>
    </div>
  </div>
  <script nonce="__NONCE__">
  const state = { activeTab: 'dashboard', currentSessionId: null, ws: null, refreshInterval: null, expandedEntries: new Set() };
  const VIEW_META = {
    dashboard: ['Dashboard', 'Live overview of agent memory'],
    activity: ['Live Activity', 'Streaming feed of everything the agent observes'],
    memories: ['Memories', 'Distilled, long-term knowledge'],
    observations: ['Observations', 'Raw captured events for this folder'],
    timeline: ['Timeline', 'Chronological view across sessions'],
    sessions: ['Sessions', 'Every agent run in this workspace'],
    profile: ['Profile', 'Aggregate stats per project'],
    'working-memory': ['Working Set', 'Recency-tiered memory (hot / warm / cold / archived)'],
    audit: ['Audit Log', 'Every memory write, in order'],
    graph: ['Graph', 'Knowledge graph'],
    replay: ['Replay', 'Session replay'],
  };
  async function api(path, opts) {
    const headers = { 'Cache-Control': 'no-cache' };
    return fetch('/api' + path, { ...opts, headers: { ...headers, ...(opts?.headers || {}) } });
  }
  function esc(s) { if (!s) return ''; return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;'); }
  const TABS = Object.keys(VIEW_META);
  function switchTab(tabId) {
    if (!TABS.includes(tabId)) return;
    state.activeTab = tabId;
    document.querySelectorAll('.nav button').forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    const view = document.getElementById('view-' + tabId);
    if (view) view.classList.add('active');
    const meta = VIEW_META[tabId];
    if (meta) {
      document.getElementById('view-heading').textContent = meta[0];
      document.getElementById('view-subheading').textContent = meta[1];
    }
    if (tabId === 'dashboard') loadDashboard();
    else if (tabId === 'observations') loadObservations();
    else if (tabId === 'memories') loadMemories();
    else if (tabId === 'timeline') loadTimeline();
    else if (tabId === 'sessions') loadSessions();
    else if (tabId === 'audit') loadAudit();
    else if (tabId === 'activity') loadActivity();
    else if (tabId === 'profile') loadProfile();
    else if (tabId === 'working-memory') loadWorkingMemory();
  }
  function isDark() { return document.documentElement.dataset.theme !== 'light'; }
  document.getElementById('theme-toggle').addEventListener('click', () => {
    const next = isDark() ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('logician-theme', next);
  });
  const saved = localStorage.getItem('logician-theme');
  if (saved === 'light') document.documentElement.dataset.theme = 'light';
  else if (saved === 'dark') document.documentElement.dataset.theme = 'dark';
  const dateEl = document.getElementById('dateline');
  if (dateEl) dateEl.textContent = new Date().toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
  const TYPE_COLORS = { file_read:'slate', file_write:'sage', file_edit:'amber', command_run:'violet', search:'teal', web_fetch:'teal', conversation:'muted', error:'coral', decision:'violet', other:'muted' };
  const MEM_COLORS = { pattern:'violet', preference:'slate', architecture:'teal', bug:'coral', workflow:'sage', fact:'amber' };
  function impBadge(imp) { return imp >= 7 ? 'coral' : imp >= 4 ? 'amber' : 'sage'; }
  function impClass(imp) { return imp >= 7 ? 'imp-high' : imp >= 4 ? 'imp-med' : 'imp-low'; }
  function expandHint(noun) {
    return '<div class="entry-expand-hint"><span class="expand-collapsed">Show full '+noun+'</span><span class="expand-expanded">Collapse '+noun+'</span></div>';
  }
  function expandableAttrs(key, expandable) {
    if (!expandable) return '';
    const expanded = state.expandedEntries.has(key);
    return ' expandable-entry'+(expanded ? ' expanded' : '')+'" data-entry-key="'+esc(key)+'" role="button" tabindex="0" aria-expanded="'+expanded;
  }
  function toggleExpandedEntry(card) {
    const key = card?.dataset.entryKey;
    if (!key) return;
    const expanded = !card.classList.contains('expanded');
    card.classList.toggle('expanded', expanded);
    card.setAttribute('aria-expanded', String(expanded));
    if (expanded) state.expandedEntries.add(key); else state.expandedEntries.delete(key);
  }
  async function loadDashboard() {
    try {
      const res = await api('/stats');
      const data = await res.json();
      const s = data.stats || {}, h = data.health || {};
      document.getElementById('stat-sessions').textContent = s.sessions || 0;
      document.getElementById('stat-memories').textContent = s.memories || 0;
      document.getElementById('stat-observations').textContent = s.observations || 0;
      document.getElementById('stat-observations-today').textContent = s.observationsToday || 0;
      const chip = document.getElementById('workspace-chip');
      if (chip) chip.textContent = s.workspace || 'Unknown workspace';
      const rssMB = (h.rss / 1024 / 1024).toFixed(0);
      const heapMB = (h.heapUsed / 1024 / 1024).toFixed(0);
      const maxMem = 512;
      const gauge = (label, value, mb) => {
        const pct = Math.min(100, value / maxMem / 1024 / 1024 * 100);
        const color = pct > 80 ? 'var(--coral)' : 'var(--accent)';
        return '<div class="health-bar"><span class="gauge-label">'+label+'</span><div class="gauge-bar"><div class="gauge-fill" style="width:'+pct+'%;background:'+color+'"></div></div><span class="gauge-value">'+mb+' MB</span></div>';
      };
      document.getElementById('health-bars').innerHTML = gauge('RSS', h.rss, rssMB) + gauge('Heap', h.heapUsed, heapMB);
      const mType = s.memoriesByType || {};
      const mMax = Math.max(1, ...Object.values(mType));
      document.getElementById('memories-by-type').innerHTML = Object.entries(mType).map(([t, c]) =>
        '<div class="bar-row"><span class="bar-label">'+esc(t)+'</span><div class="bar-track"><div class="bar-fill" style="width:'+(c/mMax*100)+'%;background:var(--'+(MEM_COLORS[t]||'slate')+')"></div></div><span class="bar-value">'+c+'</span></div>'
      ).join('') || '<div class="muted">No memories yet</div>';
      const sStatus = s.sessionsByStatus || {};
      const sMax = Math.max(1, ...Object.values(sStatus));
      const sColors = { active:'sage', completed:'slate', abandoned:'coral' };
      document.getElementById('sessions-by-status').innerHTML = Object.entries(sStatus).map(([t, c]) =>
        '<div class="bar-row"><span class="bar-label">'+esc(t)+'</span><div class="bar-track"><div class="bar-fill" style="width:'+(c/sMax*100)+'%;background:var(--'+(sColors[t]||'slate')+')"></div></div><span class="bar-value">'+c+'</span></div>'
      ).join('') || '<div class="muted">No sessions yet</div>';
      const actRes = await api('/activity?limit=5');
      const actData = await actRes.json();
      const actHtml = (actData || []).slice(0, 5).map(a => {
        const o = a.observation || {};
        return '<div class="entry-card ' + impClass(o.importance) + '" style="margin-bottom:8px;">' +
          '<div class="entry-head"><span class="badge badge-' + impBadge(o.importance) + '">'+o.importance+'/10</span>' +
          '<span style="font-size:12.5px;color:var(--text);font-weight:500;">' + esc(o.title) + '</span></div>' +
          '<div class="entry-meta">' + esc(o.type) + ' · ' + esc(a.sessionProject || '') + '</div></div>';
      }).join('');
      document.getElementById('recent-activity').innerHTML = actHtml || '<div class="empty-state"><div class="empty-icon">&#9679;</div><p>No recent activity</p></div>';
    } catch (e) { console.error('[dashboard] error:', e); }
  }
  async function loadObservations() {
    const search = document.getElementById('observation-search').value.trim();
    const type = document.getElementById('observation-type').value;
    const minImportance = document.getElementById('observation-min-importance').value;
    const params = new URLSearchParams({ limit: '250', minImportance });
    if (search) params.set('search', search);
    if (type) params.set('type', type);
    try {
      const [obsRes, statsRes] = await Promise.all([api('/observations?' + params), api('/stats')]);
      const observations = await obsRes.json();
      const statsData = await statsRes.json();
      document.getElementById('workspace-chip').textContent = statsData.stats?.workspace || 'Unknown workspace';
      const html = observations.length === 0
        ? '<div class="empty-state"><div class="empty-icon">&#9633;</div><p>No observations in this folder</p></div>'
        : observations.map((o, index) => {
          const narrative = String(o.narrative || '');
          const expandable = narrative.length > 500;
          const key = 'observation:' + String(o.id || index);
          return '<div class="entry-card ' + impClass(o.importance) + expandableAttrs(key, expandable) + '">' +
          '<div class="entry-head">' +
          '<span class="badge badge-muted">#' + (index + 1) + '</span>' +
          '<span class="badge badge-' + (TYPE_COLORS[o.type] || 'slate') + '">' + esc(o.type) + '</span>' +
          '<span class="badge badge-' + impBadge(o.importance) + '">' + o.importance + '/10</span>' +
          '<span class="entry-meta" style="margin-left:auto;">' + esc(o.timestamp?.slice(0,19)) + '</span></div>' +
          '<div class="entry-title">' + esc(o.title) + '</div>' +
          '<div class="entry-meta">' + esc(o.id) + '</div>' +
          (narrative ? '<div class="entry-body entry-preview">' + esc(narrative.slice(0,500)) + (expandable ? '…' : '') + '</div>' : '') +
          (expandable ? '<div class="entry-body entry-full">' + esc(narrative) + '</div>' + expandHint('observation') : '') +
          '</div>';
        }).join('');
      document.getElementById('observations-list').innerHTML = html;
    } catch (e) { console.error('[observations] error:', e); }
  }
  async function loadMemories() {
    const search = document.getElementById('memory-search').value;
    const type = document.getElementById('memory-type').value;
    const minStr = document.getElementById('memory-min-strength').value;
    const params = new URLSearchParams();
    if (search) params.set('search', search);
    if (type) params.set('type', type);
    if (minStr) params.set('minStrength', minStr);
    try {
      const res = await api('/memories?' + params.toString());
      const memories = await res.json();
      const html = memories.length === 0
        ? '<div class="empty-state"><div class="empty-icon">&#9670;</div><p>No memories found</p></div>'
        : memories.map((m, index) => {
          const content = String(m.content || '');
          const expandable = content.length > 140;
          const key = 'memory:' + String(m.id || index);
          return '<div class="entry-card' + expandableAttrs(key, expandable) + '">' +
          '<div class="entry-head"><span class="badge badge-' + (m.strength >= 8 ? 'coral' : m.strength >= 6 ? 'amber' : 'sage') + '">' + m.strength + '/10</span>' +
          '<span class="badge badge-' + (MEM_COLORS[m.type] || 'slate') + '">' + esc(m.type) + '</span></div>' +
          '<div class="entry-title entry-preview">' + esc(content.slice(0, 140)) + (expandable ? '…' : '') + '</div>' +
          (expandable ? '<div class="entry-body entry-full">' + esc(content) + '</div>' : '') +
          '<div class="entry-meta">created ' + esc(m.createdAt?.slice(0, 10)) + '</div>' +
          (m.concepts?.length ? '<div style="margin-top:8px;">' + m.concepts.map(c => '<span class="tag">'+esc(c)+'</span>').join('') + '</div>' : '') +
          (expandable ? expandHint('memory') : '') +
          '</div>';
        }).join('');
      document.getElementById('memories-list').innerHTML = html;
    } catch (e) { console.error('[memories] error:', e); }
  }
  async function loadTimeline() {
    const sessionId = document.getElementById('timeline-session').value;
    const minImp = document.getElementById('timeline-min-importance').value;
    const params = new URLSearchParams();
    if (sessionId) params.set('sessionId', sessionId);
    params.set('minImportance', minImp);
    params.set('limit', '100');
    try {
      const res = await api('/observations?' + params.toString());
      const obs = await res.json();
      const html = obs.length === 0
        ? '<div class="empty-state"><div class="empty-icon">&#9702;</div><p>No observations</p></div>'
        : obs.map(o => '<div class="entry-card ' + impClass(o.importance) + '">' +
          '<div class="entry-head">' +
          '<span class="badge badge-'+(TYPE_COLORS[o.type]||'slate')+'">'+esc(o.type)+'</span>' +
          '<span class="badge badge-'+impBadge(o.importance)+'">'+o.importance+'/10</span>' +
          '<span class="entry-meta" style="margin-left:auto;">'+esc(o.timestamp?.slice(0, 16))+'</span>' +
          '</div>' +
          '<div class="entry-title">'+esc(o.title)+'</div>' +
          '<div class="entry-body">'+esc(o.narrative?.slice(0, 300))+'</div>' +
          (o.concepts?.length ? '<div style="margin-top:8px;">' + o.concepts.map(c => '<span class="tag">'+esc(c)+'</span>').join('') + '</div>' : '') +
          '</div>').join('');
      document.getElementById('timeline-container').innerHTML = html;
    } catch (e) { console.error('[timeline] error:', e); }
  }
  async function loadSessions() {
    try {
      const res = await api('/sessions');
      const sessions = await res.json();
      const sel = document.getElementById('timeline-session');
      const curVal = sel.value;
      sel.innerHTML = '<option value="">All sessions</option>' + sessions.map(s => '<option value="'+esc(s.id)+'">'+esc((s.name||s.project||'Untitled').slice(0,40))+'</option>').join('');
      sel.value = curVal;
      const html = sessions.length === 0
        ? '<div class="empty-state"><div class="empty-icon">&#9635;</div><p>No sessions</p></div>'
        : sessions.map(s => '<div class="session-item'+(state.currentSessionId === s.id ? ' selected' : '')+'" data-session-id="'+esc(s.id)+'">' +
          '<div style="display:flex;justify-content:space-between;align-items:center;gap:10px;">' +
          '<span class="session-project">'+esc(s.name||s.project||'Untitled')+'</span>' +
          '<span class="badge badge-'+(s.status==='active'?'sage':'slate')+'">'+esc(s.status)+'</span>' +
          '</div><div class="session-meta">'+s.observationCount+' obs · '+esc(s.startedAt?.slice(0,16))+'</div></div>').join('');
      const listEl = document.getElementById('sessions-list');
      listEl.innerHTML = html;
      listEl.querySelectorAll('[data-session-id]').forEach(el => {
        el.addEventListener('click', () => selectSession(el.getAttribute('data-session-id')));
      });
    } catch (e) { console.error('[sessions] error:', e); }
  }
  async function selectSession(sid) {
    state.currentSessionId = sid;
    await loadSessions();
    try {
      const res = await api('/sessions/' + sid);
      const session = await res.json();
      const obsRes = await api('/observations?sessionId='+sid+'&limit=50');
      const obs = await obsRes.json();
      const oHtml = obs.slice(0, 20).map(o =>
        '<div class="entry-card '+impClass(o.importance)+'">' +
        '<div class="entry-head"><span class="badge badge-'+impBadge(o.importance)+'">'+o.importance+'/10</span></div>' +
        '<div class="entry-title">'+esc(o.title)+'</div>' +
        '<div class="entry-meta">'+esc(o.type)+' · '+esc(o.timestamp?.slice(0,16))+'</div>' +
        '<div class="entry-body">'+esc(o.narrative?.slice(0,200))+'</div></div>').join('');
      document.getElementById('session-detail').innerHTML =
        '<div class="detail-panel"><h3>'+esc(session.name||session.project||'Untitled')+'</h3>' +
        '<div class="detail-row"><span class="dl">ID</span><span class="dv" style="font-family:var(--font-mono);font-size:11px;">'+esc(session.id)+'</span></div>' +
        '<div class="detail-row"><span class="dl">Status</span><span class="dv"><span class="badge badge-'+(session.status==='active'?'sage':'slate')+'">'+esc(session.status)+'</span></span></div>' +
        '<div class="detail-row"><span class="dl">Observations</span><span class="dv">'+session.observationCount+'</span></div>' +
        '<div class="detail-row"><span class="dl">Started</span><span class="dv">'+esc(session.startedAt)+'</span></div>' +
        (session.summary ? '<div class="detail-row"><span class="dl">Summary</span><span class="dv">'+esc(session.summary)+'</span></div>' : '') +
        (session.model ? '<div class="detail-row"><span class="dl">Model</span><span class="dv" style="font-family:var(--font-mono);font-size:11px;">'+esc(session.model)+'</span></div>' : '') +
        '</div>' + (obs.length > 0 ? '<h3 style="margin:20px 0 12px;font-size:12px;color:var(--text-faint);text-transform:uppercase;letter-spacing:0.08em;">Recent observations ('+obs.length+')</h3>' + oHtml : '');
    } catch (e) { console.error('[session] error:', e); }
  }
  async function loadAudit() {
    try {
      const res = await api('/audit?limit=100');
      const entries = await res.json();
      const opColors = { create:'sage', delete:'coral', forget:'coral', consolidate:'amber', update:'slate' };
      const html = entries.length === 0
        ? '<tr><td colspan="5"><div class="empty-state"><div class="empty-icon">&#9776;</div><p>No audit entries</p></div></td></tr>'
        : entries.map(e => '<tr><td style="font-family:var(--font-mono);font-size:11px;">'+esc(e.timestamp?.slice(0,16))+'</td>' +
          '<td><span class="badge badge-'+(opColors[e.operation]||'muted')+'">'+esc(e.operation)+'</span></td>' +
          '<td>'+esc(e.resource)+'</td><td>'+esc(e.type||'')+'</td>' +
          '<td>'+(e.strength != null ? '<span class="badge badge-'+(e.strength >= 7 ? 'coral':'sage')+'">'+e.strength+'/10</span>' : '—')+'</td></tr>').join('');
      document.getElementById('audit-body').innerHTML = html;
    } catch (e) { console.error('[audit] error:', e); }
  }
  async function loadActivity() {
    try {
      const res = await api('/activity?limit=50');
      const activity = await res.json();
      const html = activity.length === 0
        ? '<div class="empty-state"><div class="empty-icon">&#9679;</div><p>No activity</p></div>'
        : activity.slice(0, 50).map(a => {
          const o = a.observation || {};
          const c = 'var(--' + (TYPE_COLORS[o.type] === 'muted' ? 'text-faint' : (TYPE_COLORS[o.type] || 'slate')) + ')';
          return '<div class="activity-item">' +
            '<div class="activity-dot" style="background:'+c+'"></div>' +
            '<div class="activity-body">' +
            '<div class="activity-title"><span class="badge badge-'+impBadge(o.importance)+'">'+o.importance+'/10</span> '+esc(o.title)+'</div>' +
            '<div class="activity-meta">'+esc(o.type)+' · '+esc(a.sessionProject||'Unknown')+' · '+esc(o.timestamp?.slice(0,16))+'</div>' +
            (o.narrative ? '<p>'+esc(o.narrative.slice(0,150))+'</p>' : '') +
            '</div></div>';
        }).join('');
      document.getElementById('activity-feed').innerHTML = html;
    } catch (e) { console.error('[activity] error:', e); }
  }
  async function loadProfile() {
    try {
      const sRes = await api('/sessions');
      const sessions = await sRes.json();
      const mRes = await api('/memories?limit=1000');
      const memories = await mRes.json();
      const oRes = await api('/observations?limit=1000');
      const observations = await oRes.json();
      const stats = sessions.map(s => {
        const sObs = observations.filter(o => o.sessionId === s.id);
        return { project: s.project, status: s.status, observations: sObs.length, strength: sObs.reduce((a, o) => a + o.importance, 0) / Math.max(1, sObs.length), memories: memories.filter(m => m.sessionIds?.includes(s.id)).length };
      });
      const html = '<div class="card" style="padding:0;overflow:hidden;"><div style="padding:18px 20px 0;"><div class="card-title" style="margin-bottom:14px;">Project overview</div></div>' +
        '<table><thead><tr><th>Project</th><th>Status</th><th>Obs</th><th>Avg importance</th><th>Memories</th></tr></thead><tbody>' +
        stats.map(s => '<tr><td>'+esc(s.project)+'</td><td><span class="badge badge-'+(s.status==='active'?'sage':'slate')+'">'+s.status+'</span></td><td>'+s.observations+'</td><td>'+s.strength.toFixed(1)+'/10</td><td>'+s.memories+'</td></tr>').join('') +
        '</tbody></table></div>';
      document.getElementById('profile-content').innerHTML = html;
    } catch (e) { console.error('[profile] error:', e); }
  }
  async function loadWorkingMemory() {
    try {
      const res = await api('/working-memory');
      const tiered = await res.json();
      const rows = Object.entries(tiered).map(([id, info]) => ({ id, ...info })).slice(0, 200);
      const filter = document.getElementById('wm-tier-filter').value;
      const filtered = filter ? rows.filter(r => r.tier === filter) : rows;
      const tierColors = { hot:'coral', warm:'amber', cold:'sage', archived:'muted' };
      const html = filtered.length === 0
        ? '<tr><td colspan="4"><div class="empty-state"><div class="empty-icon">&#9642;</div><p>No memories to display</p></div></td></tr>'
        : filtered.map(r => '<tr><td><span class="badge badge-'+tierColors[r.tier]+'">'+esc(r.tier)+'</span></td>' +
          '<td>'+esc(r.type)+'</td>' +
          '<td><span class="badge badge-'+(r.strength >= 7 ? 'coral':'sage')+'">'+r.strength+'/10</span></td>' +
          '<td>'+esc(r.content?.slice(0, 100))+'</td></tr>').join('');
      document.getElementById('wm-body').innerHTML = html;
    } catch (e) { console.error('[wm] error:', e); }
  }
  function connectWebSocket() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    state.ws = new WebSocket(proto + '//' + window.location.host + '/ws');
    state.ws.onopen = () => {
      document.getElementById('pulse-dot').classList.add('live');
      document.getElementById('pulse-label').classList.add('live');
      document.getElementById('pulse-label').textContent = 'agent live';
      state.ws.send(JSON.stringify({ type: 'subscribe' }));
    };
    state.ws.onclose = () => {
      document.getElementById('pulse-dot').classList.remove('live');
      document.getElementById('pulse-label').classList.remove('live');
      document.getElementById('pulse-label').textContent = 'reconnecting…';
      setTimeout(connectWebSocket, 3000);
    };
    state.ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === 'observation' && state.activeTab === 'dashboard') loadDashboard();
        if (msg.type === 'observation' && state.activeTab === 'observations') loadObservations();
        if (msg.type === 'observation' && state.activeTab === 'activity') loadActivity();
      } catch {}
    };
  }
  document.querySelectorAll('.nav button').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });
  document.getElementById('memory-search').addEventListener('input', debounce(loadMemories, 300));
  document.getElementById('observation-search').addEventListener('input', debounce(loadObservations, 300));
  document.getElementById('observation-type').addEventListener('change', loadObservations);
  document.getElementById('observation-min-importance').addEventListener('change', loadObservations);
  document.getElementById('btn-refresh-observations').addEventListener('click', loadObservations);
  document.getElementById('memory-type').addEventListener('change', loadMemories);
  document.getElementById('memory-min-strength').addEventListener('change', loadMemories);
  document.getElementById('btn-refresh-memories').addEventListener('click', loadMemories);
  for (const listId of ['observations-list', 'memories-list']) {
    const list = document.getElementById(listId);
    list.addEventListener('click', event => {
      const card = event.target.closest('.expandable-entry');
      if (card && list.contains(card)) toggleExpandedEntry(card);
    });
    list.addEventListener('keydown', event => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      const card = event.target.closest('.expandable-entry');
      if (!card || !list.contains(card)) return;
      event.preventDefault();
      toggleExpandedEntry(card);
    });
  }
  document.getElementById('timeline-session').addEventListener('change', loadTimeline);
  document.getElementById('timeline-min-importance').addEventListener('change', loadTimeline);
  document.getElementById('btn-refresh-timeline').addEventListener('click', loadTimeline);
  document.getElementById('btn-refresh-audit').addEventListener('click', loadAudit);
  document.getElementById('btn-refresh-activity').addEventListener('click', loadActivity);
  document.getElementById('btn-refresh-wm').addEventListener('click', loadWorkingMemory);
  document.getElementById('wm-tier-filter').addEventListener('change', loadWorkingMemory);
  document.getElementById('btn-auto-tier').addEventListener('click', async () => {
    try {
      await api('/auto-tier', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
      loadWorkingMemory();
    } catch (e) { console.error('[auto-tier] error:', e); }
  });
  function debounce(fn, ms) { let t; return function() { clearTimeout(t); t = setTimeout(fn, ms); };
  loadDashboard();
  state.refreshInterval = setInterval(() => {
    if (state.activeTab === 'observations') loadObservations();
    else if (state.activeTab === 'dashboard') loadDashboard();
    else if (state.activeTab === 'activity') loadActivity();
  }, 5000);
  connectWebSocket();
  </script>
</body>
</html>`;

export default HTML;
