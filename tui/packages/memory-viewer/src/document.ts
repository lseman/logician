// ── @logician/memory-viewer — Dashboard HTML Document ────────────────────────
// Single-page app with all tabs, CSS, and client-side JS.
// Self-hosted, zero dependencies.

const HTML = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Logician Memory Viewer</title>
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <style>
    :root {
      --bg: #F9F9F7; --bg-alt: #F0F0EC; --bg-subtle: #F4F4F0;
      --bg-inset: #E8E8E3; --border: #111111; --border-light: #D4D4CF;
      --border-heavy: #111111; --ink: #111111; --ink-secondary: #333333;
      --ink-muted: #666666; --ink-faint: #999999;
      --accent: #CC0000; --green: #2D6A4F; --blue: #1D4E89;
      --yellow: #B8860B; --red: #CC0000; --purple: #6B3FA0;
      --orange: #C2410C; --cyan: #0E7490; --pink: #EA76CB;
      --font-display: Georgia, serif;
      --font-body: Georgia, serif;
      --font-ui: -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }
    html[data-theme="dark"] {
      --bg: #1a1a1e; --bg-alt: #232328; --bg-subtle: #1f1f24;
      --bg-inset: #2a2a30; --border: #444; --border-light: #3a3a42;
      --border-heavy: #ccc; --ink: #eee; --ink-secondary: #ccc;
      --ink-muted: #999; --ink-faint: #777;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: var(--font-body); background: var(--bg); color: var(--ink-secondary); line-height: 1.6; overflow: hidden; height: 100vh; display: flex; flex-direction: column; }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border-light); }
    .app-header { padding: 10px 24px; border-bottom: 4px solid var(--border-heavy); display: flex; align-items: center; justify-content: space-between; background: var(--bg); flex: 0 0 auto; }
    .app-header .brand { display: flex; align-items: baseline; gap: 10px; cursor: pointer; }
    .app-header .brand h1 { font-size: 22px; color: var(--ink); font-weight: 900; font-family: var(--font-display); letter-spacing: -0.02em; }
    .app-header .brand .version { font-size: 10px; color: var(--ink-faint); font-family: var(--font-mono); text-transform: uppercase; }
    .header-right { display: flex; align-items: center; gap: 12px; }
    .ws-status { font-size: 10px; padding: 3px 10px; display: flex; align-items: center; gap: 5px; font-family: var(--font-ui); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; border: 1px solid var(--border-light); }
    .ws-status::before { content: ''; width: 6px; height: 6px; display: inline-block; border-radius: 50%; }
    .ws-status.connected { border-color: var(--green); color: var(--green); }
    .ws-status.connected::before { background: var(--green); }
    .ws-status.disconnected { border-color: var(--ink-faint); color: var(--ink-faint); }
    .ws-status.disconnected::before { background: var(--ink-faint); }
    .tab-bar { display: flex; height: 48px; flex-shrink: 0; border-bottom: 1px solid var(--border-light); background: var(--bg); overflow-x: auto; flex: 0 0 auto; }
    .tab-bar button { background: none; border: none; color: var(--ink-muted); padding: 10px 20px; font-size: 11px; cursor: pointer; border-bottom: 2px solid transparent; white-space: nowrap; font-family: var(--font-ui); text-transform: uppercase; letter-spacing: 0.12em; font-weight: 600; transition: color 0.15s, border-color 0.15s; }
    .tab-bar button:hover { color: var(--ink); }
    .tab-bar button.active { color: var(--ink); border-bottom-color: var(--accent); }
    .tab-bar button:disabled { opacity: 0.4; pointer-events: none; }
    .view { display: none; flex: 1 1 auto; min-height: 0; overflow-y: auto; padding: 24px; }
    .view.active { display: block; }
    .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0; margin-bottom: 24px; border: 1px solid var(--border); }
    .stat-card { background: var(--bg); padding: 16px 20px; border-right: 1px solid var(--border-light); border-bottom: 1px solid var(--border-light); }
    .stat-card:last-child { border-right: none; }
    .stat-card .label { font-size: 9px; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 4px; font-family: var(--font-ui); font-weight: 600; }
    .stat-card .value { font-size: 32px; font-weight: 900; color: var(--ink); font-family: var(--font-display); line-height: 1.1; }
    .stat-card .sub { font-size: 11px; color: var(--ink-faint); margin-top: 2px; font-family: var(--font-ui); }
    .card { background: var(--bg); border: 1px solid var(--border); padding: 20px; margin-bottom: 16px; }
    .card-title { font-size: 13px; font-weight: 700; color: var(--ink); margin-bottom: 12px; font-family: var(--font-display); text-transform: uppercase; letter-spacing: 0.06em; padding-bottom: 8px; border-bottom: 1px solid var(--border-light); }
    .health-bar { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
    .gauge-bar { flex: 1; height: 6px; background: var(--bg-inset); overflow: hidden; }
    .gauge-fill { height: 100%; transition: width 0.5s; }
    .gauge-label { width: 90px; font-size: 10px; color: var(--ink-muted); font-family: var(--font-ui); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
    .gauge-value { width: 70px; font-size: 11px; color: var(--ink); text-align: right; font-family: var(--font-mono); }
    .badge { display: inline-block; font-size: 9px; padding: 2px 8px; font-weight: 600; font-family: var(--font-ui); text-transform: uppercase; letter-spacing: 0.08em; border: 1px solid; }
    .badge-blue { border-color: var(--blue); color: var(--blue); }
    .badge-green { border-color: var(--green); color: var(--green); }
    .badge-yellow { border-color: var(--yellow); color: var(--yellow); }
    .badge-red { border-color: var(--red); color: var(--red); }
    .badge-purple { border-color: var(--purple); color: var(--purple); }
    .badge-muted { border-color: var(--border-light); color: var(--ink-muted); }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th { text-align: left; padding: 8px 12px; border-bottom: 2px solid var(--border); color: var(--ink); font-size: 9px; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 600; font-family: var(--font-ui); }
    td { padding: 8px 12px; border-bottom: 1px solid var(--border-light); vertical-align: top; }
    tr:hover td { background: var(--bg-alt); }
    .toolbar { display: flex; gap: 10px; margin-bottom: 20px; align-items: center; flex-wrap: wrap; }
    .toolbar input, .toolbar select { background: var(--bg); border: 1px solid var(--border); color: var(--ink); padding: 7px 12px; font-size: 13px; outline: none; font-family: var(--font-ui); }
    .toolbar input:focus, .toolbar select:focus { border-color: var(--ink); box-shadow: 2px 2px 0px 0px var(--border); }
    .toolbar input { flex: 1; min-width: 200px; }
    .btn { background: var(--bg); border: 1px solid var(--border); color: var(--ink); padding: 7px 16px; font-size: 11px; cursor: pointer; font-family: var(--font-ui); font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
    .btn:hover { box-shadow: 3px 3px 0px 0px var(--border); }
    .btn-primary { background: var(--ink); color: var(--bg); border-color: var(--ink); }
    .btn-danger { border-color: var(--red); color: var(--red); }
    .obs-card { background: var(--bg); border: 1px solid var(--border-light); padding: 16px 20px; margin-bottom: 12px; border-left: 3px solid var(--border-light); }
    .obs-card.imp-high { border-left-color: var(--red); }
    .obs-card.imp-med { border-left-color: var(--yellow); }
    .obs-card.imp-low { border-left-color: var(--green); }
    .obs-card .obs-title { font-size: 14px; font-weight: 700; color: var(--ink); font-family: var(--font-display); }
    .obs-card .obs-meta { font-size: 10px; color: var(--ink-faint); font-family: var(--font-mono); margin-top: 4px; }
    .obs-card .obs-narrative { font-size: 13px; color: var(--ink-muted); margin-top: 8px; word-break: break-word; }
    .session-item { background: var(--bg); border: 1px solid var(--border-light); padding: 14px 20px; cursor: pointer; margin-bottom: 4px; }
    .session-item:hover { background: var(--bg-alt); }
    .session-item.selected { background: var(--bg-alt); border-left: 3px solid var(--accent); }
    .session-item .session-project { font-weight: 700; color: var(--ink); font-size: 14px; font-family: var(--font-display); }
    .session-item .session-meta { font-size: 11px; color: var(--ink-muted); font-family: var(--font-mono); }
    .detail-panel { background: var(--bg); border: 1px solid var(--border); padding: 24px; margin-top: 20px; }
    .detail-panel h3 { font-size: 15px; font-weight: 700; color: var(--ink); margin-bottom: 16px; font-family: var(--font-display); text-transform: uppercase; padding-bottom: 8px; border-bottom: 2px solid var(--border); }
    .detail-row { display: flex; padding: 6px 0; font-size: 13px; border-bottom: 1px solid var(--bg-inset); }
    .detail-row .dl { color: var(--ink-muted); width: 140px; flex-shrink: 0; font-family: var(--font-ui); font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }
    .detail-row .dv { color: var(--ink); }
    .bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 12px; }
    .bar-label { width: 120px; color: var(--ink-muted); font-family: var(--font-mono); font-size: 11px; }
    .bar-track { flex: 1; height: 6px; background: var(--bg-inset); overflow: hidden; }
    .bar-fill { height: 100%; transition: width 0.3s; }
    .bar-value { width: 30px; text-align: right; color: var(--ink-muted); font-size: 11px; font-family: var(--font-mono); }
    .tag { font-size: 10px; padding: 1px 6px; border: 1px solid var(--blue); color: var(--blue); font-family: var(--font-mono); font-weight: 500; display: inline-block; margin: 1px 2px; }
    .tag-green { border-color: var(--green); color: var(--green); }
    .empty-state { text-align: center; padding: 60px 20px; color: var(--ink-faint); }
    .empty-state .empty-icon { font-size: 36px; margin-bottom: 10px; opacity: 0.4; }
    .empty-state p { font-size: 14px; font-style: italic; }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 768px) { .two-col { grid-template-columns: 1fr; } }
    .placeholder { text-align: center; padding: 80px 20px; color: var(--ink-faint); }
    .placeholder .placeholder-icon { font-size: 48px; opacity: 0.3; margin-bottom: 16px; }
    .placeholder h3 { font-size: 18px; color: var(--ink-muted); margin-bottom: 8px; font-family: var(--font-display); }
    .muted { color: var(--ink-muted); font-size: 11px; font-style: italic; }
    .activity-item { display: flex; gap: 12px; padding: 12px 0; border-bottom: 1px solid var(--border-light); }
    .activity-dot { width: 8px; height: 8px; border-radius: 50%; margin-top: 6px; flex-shrink: 0; }
    .activity-body { flex: 1; }
    .activity-title { font-size: 13px; color: var(--ink); font-family: var(--font-display); }
    .activity-meta { font-size: 11px; color: var(--ink-faint); font-family: var(--font-mono); margin-top: 2px; }
    .activity-body p { font-size: 12px; color: var(--ink-muted); margin-top: 6px; }
  </style>
</head>
<body>
  <div class="app-header">
    <div class="brand" onclick="switchTab('dashboard')">
      <h1>logician memory</h1>
      <span class="version">viewer</span>
    </div>
    <div class="header-right">
      <span id="dateline"></span>
      <button id="theme-toggle" class="btn" style="font-size:9px;padding:3px 10px;letter-spacing:0.1em;">DARK</button>
      <span id="ws-status" class="ws-status disconnected">offline</span>
    </div>
  </div>
  <div class="tab-bar" id="tab-bar">
    <button class="active" data-tab="dashboard">Dashboard</button>
    <button data-tab="observations">Observations</button>
    <button data-tab="memories">Memories</button>
    <button data-tab="timeline">Timeline</button>
    <button data-tab="sessions">Sessions</button>
    <button data-tab="audit">Audit</button>
    <button data-tab="activity">Activity</button>
    <button data-tab="profile">Profile</button>
    <button data-tab="working-memory">Working Memory</button>
    <button data-tab="graph" disabled>Graph</button>
    <button data-tab="replay" disabled>Replay</button>
  </div>
  <div id="view-dashboard" class="view active">
    <div class="stats-grid" id="stats-grid">
      <div class="stat-card"><div class="label">Sessions</div><div class="value" id="stat-sessions">—</div></div>
      <div class="stat-card"><div class="label">Memories</div><div class="value" id="stat-memories">—</div></div>
      <div class="stat-card"><div class="label">Observations</div><div class="value" id="stat-observations">—</div></div>
      <div class="stat-card"><div class="label">Today</div><div class="value" id="stat-observations-today">—</div></div>
    </div>
    <div class="two-col">
      <div>
        <div class="card"><div class="card-title">Health</div><div id="health-bars"></div></div>
        <div class="card"><div class="card-title">Memories by Type</div><div id="memories-by-type"></div></div>
      </div>
      <div>
        <div class="card"><div class="card-title">Sessions by Status</div><div id="sessions-by-status"></div></div>
        <div class="card"><div class="card-title">Recent Activity</div><div id="recent-activity"></div></div>
      </div>
    </div>
  </div>
  <div id="view-observations" class="view">
    <div class="card" style="padding:12px 16px;">
      <div class="card-title" style="margin-bottom:0;">Current folder</div>
      <div id="observations-workspace" style="font-family:var(--font-mono);font-size:12px;color:var(--ink-muted);word-break:break-all;">—</div>
    </div>
    <div class="toolbar">
      <input id="observation-search" type="text" placeholder="Search observations in this folder...">
      <select id="observation-type"><option value="">All Types</option><option value="conversation">Conversation</option><option value="file_read">File Read</option><option value="file_write">File Write</option><option value="file_edit">File Edit</option><option value="command_run">Command</option><option value="search">Search</option><option value="web_fetch">Web Fetch</option><option value="error">Error</option><option value="other">Other</option></select>
      <select id="observation-min-importance"><option value="0">Any Importance</option><option value="5">5+</option><option value="7">7+</option></select>
      <button class="btn" id="btn-refresh-observations">Refresh</button>
    </div>
    <div id="observations-list"></div>
  </div>
  <div id="view-memories" class="view">
    <div class="toolbar">
      <input id="memory-search" type="text" placeholder="Search memories...">
      <select id="memory-type"><option value="">All Types</option><option value="pattern">Pattern</option><option value="preference">Preference</option><option value="architecture">Architecture</option><option value="bug">Bug</option><option value="workflow">Workflow</option><option value="fact">Fact</option></select>
      <select id="memory-min-strength"><option value="">Any Strength</option><option value="3">3+</option><option value="5">5+</option><option value="7">7+</option></select>
      <button class="btn" id="btn-refresh-memories">Refresh</button>
    </div>
    <div id="memories-list"></div>
  </div>
  <div id="view-timeline" class="view">
    <div class="toolbar">
      <select id="timeline-session"><option value="">All Sessions</option></select>
      <select id="timeline-min-importance"><option value="0">All</option><option value="3">3+</option><option value="5">5+</option><option value="7">7+</option></select>
      <button class="btn" id="btn-refresh-timeline">Refresh</button>
    </div>
    <div id="timeline-container"></div>
  </div>
  <div id="view-sessions" class="view">
    <div class="two-col">
      <div id="sessions-list" style="max-width:400px;"></div>
      <div id="session-detail"></div>
    </div>
  </div>
  <div id="view-audit" class="view">
    <div class="toolbar"><select id="audit-filter"><option value="">All Operations</option><option value="create">Create</option><option value="forget">Forget</option></select><button class="btn" id="btn-refresh-audit">Refresh</button></div>
    <table id="audit-table"><thead><tr><th>Time</th><th>Operation</th><th>Resource</th><th>Type</th><th>Strength</th></tr></thead><tbody id="audit-body"></tbody></table>
  </div>
  <div id="view-activity" class="view">
    <div class="toolbar"><button class="btn" id="btn-refresh-activity">Refresh</button></div>
    <div id="activity-feed"></div>
  </div>
  <div id="view-profile" class="view"><div id="profile-content"></div></div>
  <div id="view-working-memory" class="view">
    <div class="toolbar">
      <select id="wm-tier-filter"><option value="">All Tiers</option><option value="hot">Hot</option><option value="warm">Warm</option><option value="cold">Cold</option><option value="archived">Archived</option></select>
      <button class="btn btn-primary" id="btn-auto-tier">Auto-Tier</button>
      <button class="btn" id="btn-refresh-wm">Refresh</button>
    </div>
    <table id="wm-table"><thead><tr><th>Tier</th><th>Type</th><th>Strength</th><th>Content</th></tr></thead><tbody id="wm-body"></tbody></table>
  </div>
  <div id="view-graph" class="view">
    <div class="placeholder"><div class="placeholder-icon">&#128306;</div><h3>Knowledge Graph</h3><p>Graph visualization coming soon.</p></div>
  </div>
  <div id="view-replay" class="view">
    <div class="placeholder"><div class="placeholder-icon">&#9194;</div><h3>Session Replay</h3><p>Replay feature coming soon.</p></div>
  </div>
  <script nonce="__NONCE__">
  const state = { activeTab: 'dashboard', currentSessionId: null, ws: null, refreshInterval: null };
  async function api(path, opts) {
    const headers = { 'Cache-Control': 'no-cache' };
    return fetch('/api' + path, { ...opts, headers: { ...headers, ...(opts?.headers || {}) } });
  }
  function esc(s) { if (!s) return ''; return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;'); }
  const TABS = ['dashboard','observations','memories','timeline','sessions','audit','activity','profile','working-memory','graph','replay'];
  function switchTab(tabId) {
    if (!TABS.includes(tabId)) return;
    state.activeTab = tabId;
    document.querySelectorAll('.tab-bar button').forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    const view = document.getElementById('view-' + tabId);
    if (view) view.classList.add('active');
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
  function isDark() { return document.documentElement.dataset.theme === 'dark'; }
  document.getElementById('theme-toggle').addEventListener('click', () => {
    document.documentElement.dataset.theme = isDark() ? '' : 'dark';
    document.getElementById('theme-toggle').textContent = isDark() ? 'DARK' : 'LIGHT';
    localStorage.setItem('logician-theme', isDark() ? 'light' : 'dark');
  });
  const saved = localStorage.getItem('logician-theme');
  if (saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    document.documentElement.dataset.theme = 'dark';
    document.getElementById('theme-toggle').textContent = 'LIGHT';
  }
  const dateEl = document.getElementById('dateline');
  if (dateEl) dateEl.textContent = new Date().toLocaleDateString('en-US', { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' });
  async function loadDashboard() {
    try {
      const res = await api('/stats');
      const data = await res.json();
      const s = data.stats || {}, h = data.health || {};
      const statEls = document.querySelectorAll('.stat-card .value');
      statEls[0].textContent = s.sessions || 0;
      statEls[1].textContent = s.memories || 0;
      statEls[2].textContent = s.observations || 0;
      statEls[3].textContent = s.observationsToday || 0;
      const workspaceEl = document.getElementById('observations-workspace');
      if (workspaceEl) workspaceEl.textContent = s.workspace || 'Unknown workspace';
      // Health
      const rssMB = (h.rss / 1024 / 1024).toFixed(0);
      const heapMB = (h.heapUsed / 1024 / 1024).toFixed(0);
      const maxMem = 512;
      document.getElementById('health-bars').innerHTML =
        '<div class="health-bar"><span class="gauge-label">RSS</span><div class="gauge-bar"><div class="gauge-fill" style="width:'+(h.rss/maxMem/1024/1024*100)+'%;background:'+(h.rss/maxMem/1024/1024>0.8?'var(--red)':'var(--green)')+'"></div></div><span class="gauge-value">'+rssMB+'MB</span></div>' +
        '<div class="health-bar"><span class="gauge-label">Heap</span><div class="gauge-bar"><div class="gauge-fill" style="width:'+(h.heapUsed/maxMem/1024/1024*100)+'%;background:'+(h.heapUsed/maxMem/1024/1024>0.8?'var(--red)':'var(--green)')+'"></div></div><span class="gauge-value">'+heapMB+'MB</span></div>';
      // Memories by type
      const mType = s.memoriesByType || {};
      const mColors = { pattern:'var(--purple)', preference:'var(--blue)', architecture:'var(--cyan)', bug:'var(--red)', workflow:'var(--green)', fact:'var(--yellow)' };
      document.getElementById('memories-by-type').innerHTML = Object.entries(mType).map(([t, c]) =>
        '<div class="bar-row"><span class="bar-label">'+esc(t)+'</span><div class="bar-track"><div class="bar-fill" style="width:'+c+'%;background:'+mColors[t]+'"></div></div><span class="bar-value">'+c+'</span></div>'
      ).join('') || '<div class="muted">No memories</div>';
      // Sessions by status
      const sStatus = s.sessionsByStatus || {};
      const sColors = { active:'var(--green)', completed:'var(--blue)', abandoned:'var(--red)' };
      document.getElementById('sessions-by-status').innerHTML = Object.entries(sStatus).map(([t, c]) =>
        '<div class="bar-row"><span class="bar-label">'+esc(t)+'</span><div class="bar-track"><div class="bar-fill" style="width:'+c+'%;background:'+sColors[t]+'"></div></div><span class="bar-value">'+c+'</span></div>'
      ).join('') || '<div class="muted">No sessions</div>';
      // Recent activity
      const actRes = await api('/activity?limit=5');
      const actData = await actRes.json();
      const actHtml = (actData || []).slice(0, 5).map(a => {
        const o = a.observation || {};
        return '<div class="obs-card imp-' + (o.importance >= 7 ? 'high' : o.importance >= 4 ? 'med' : 'low') + '">' +
          '<div class="obs-title"><span class="badge badge-' + (o.importance >= 7 ? 'red' : o.importance >= 4 ? 'yellow' : 'green') + '">[' + o.importance + '/10]</span> ' + esc(o.title) + '</div>' +
          '<div class="obs-meta">' + esc(o.type) + ' · ' + esc(a.sessionProject || '') + '</div></div>';
      }).join('');
      document.getElementById('recent-activity').innerHTML = actHtml || '<div class="empty-state"><div class="empty-icon">&#128231;</div><p>No recent activity</p></div>';
    } catch (e) { console.error('[dashboard] error:', e); }
  }
  // Observations for the store's active working directory
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
      document.getElementById('observations-workspace').textContent = statsData.stats?.workspace || 'Unknown workspace';
      const colors = { file_read:'var(--blue)', file_write:'var(--green)', file_edit:'var(--yellow)', command_run:'var(--orange)', search:'var(--purple)', web_fetch:'var(--cyan)', conversation:'var(--ink-muted)', error:'var(--red)', decision:'var(--purple)', other:'var(--ink-muted)' };
      const html = observations.length === 0
        ? '<div class="empty-state"><div class="empty-icon">&#128203;</div><p>No observations in this folder</p></div>'
        : observations.map((o, index) => '<div class="obs-card imp-' + (o.importance >= 7 ? 'high' : o.importance >= 4 ? 'med' : 'low') + '">' +
          '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">' +
          '<span class="badge badge-blue">#' + (index + 1) + '</span>' +
          '<span class="badge" style="border-color:' + (colors[o.type] || 'var(--border-light)') + ';color:' + (colors[o.type] || 'var(--ink-muted)') + '">' + esc(o.type) + '</span>' +
          '<span class="badge badge-' + (o.importance >= 7 ? 'red' : o.importance >= 4 ? 'yellow' : 'green') + '">importance ' + o.importance + '/10</span>' +
          '<span class="obs-meta" style="margin-left:auto;">' + esc(o.timestamp?.slice(0,19)) + '</span></div>' +
          '<div class="obs-title" style="margin-top:8px;">' + esc(o.title) + '</div>' +
          '<div class="obs-meta">' + esc(o.id) + '</div>' +
          (o.narrative ? '<div class="obs-narrative">' + esc(o.narrative.slice(0,500)) + '</div>' : '') +
          '</div>').join('');
      document.getElementById('observations-list').innerHTML = html;
    } catch (e) { console.error('[observations] error:', e); }
  }
  // Memories
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
      const colors = { pattern:'var(--purple)', preference:'var(--blue)', architecture:'var(--cyan)', bug:'var(--red)', workflow:'var(--green)', fact:'var(--yellow)' };
      const html = memories.length === 0
        ? '<div class="empty-state"><div class="empty-icon">&#129505;</div><p>No memories found</p></div>'
        : memories.map(m => '<div class="obs-card">' +
          '<div class="obs-title"><span class="badge badge-' + (m.strength >= 8 ? 'red' : m.strength >= 6 ? 'yellow' : 'green') + '">[' + m.strength + '/10]</span> ' + esc(m.content?.slice(0, 120)) + '</div>' +
          '<div class="obs-meta">' + esc(m.type) + ' · ' + esc(m.createdAt?.slice(0, 10)) + '</div>' +
          (m.concepts?.length ? '<div style="margin-top:6px;">' + m.concepts.map(c => '<span class="tag">'+esc(c)+'</span>').join('') + '</div>' : '') +
          '</div>').join('');
      document.getElementById('memories-list').innerHTML = html;
    } catch (e) { console.error('[memories] error:', e); }
  }
  // Timeline
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
      const oColors = { file_read:'var(--blue)', file_write:'var(--green)', file_edit:'var(--yellow)', command_run:'var(--orange)', search:'var(--purple)', web_fetch:'var(--cyan)', conversation:'var(--ink-muted)', error:'var(--red)', decision:'var(--purple)', other:'var(--ink-muted)' };
      const html = obs.length === 0
        ? '<div class="empty-state"><div class="empty-icon">&#128203;</div><p>No observations</p></div>'
        : obs.map(o => '<div class="obs-card imp-' + (o.importance >= 7 ? 'high' : o.importance >= 4 ? 'med' : 'low') + '">' +
          '<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">' +
          '<span class="badge" style="border-color:'+(oColors[o.type]||'var(--border-light)')+';color:'+(oColors[o.type]||'var(--ink-muted)')+'">'+esc(o.type)+'</span>' +
          '<span class="badge badge-'+(o.importance >= 7 ? 'red' : o.importance >= 4 ? 'yellow' : 'green')+'">['+o.importance+'/10]</span>' +
          '<span style="font-family:var(--font-mono);font-size:11px;color:var(--ink-faint);margin-left:auto;">'+esc(o.timestamp?.slice(0, 16))+'</span>' +
          '</div>' +
          '<div style="font-family:var(--font-display);font-size:14px;color:var(--ink);margin-bottom:4px;">'+esc(o.title)+'</div>' +
          '<div style="font-size:12px;color:var(--ink-muted);white-space:pre-wrap;">'+esc(o.narrative?.slice(0, 300))+'</div>' +
          (o.concepts?.length ? '<div style="margin-top:6px;">' + o.concepts.map(c => '<span class="tag">'+esc(c)+'</span>').join('') + '</div>' : '') +
          '</div>').join('');
      document.getElementById('timeline-container').innerHTML = html;
    } catch (e) { console.error('[timeline] error:', e); }
  }
  // Sessions
  async function loadSessions() {
    try {
      const res = await api('/sessions');
      const sessions = await res.json();
      // Populate session dropdown
      const sel = document.getElementById('timeline-session');
      const curVal = sel.value;
      sel.innerHTML = '<option value="">All Sessions</option>' + sessions.map(s => '<option value="'+esc(s.id)+'">'+esc((s.name||s.project||'Untitled').slice(0,40))+'</option>').join('');
      sel.value = curVal;
      // Render list
      const html = sessions.length === 0
        ? '<div class="empty-state"><div class="empty-icon">&#128193;</div><p>No sessions</p></div>'
        : sessions.map(s => '<div class="session-item'+(state.currentSessionId === s.id ? ' selected' : '')+'" data-session-id="'+esc(s.id)+'">' +
          '<div style="display:flex;justify-content:space-between;align-items:center;">' +
          '<span class="session-project">'+esc(s.name||s.project||'Untitled')+'</span>' +
          '<span class="session-meta">'+esc(s.status)+' · '+s.observationCount+' obs</span>' +
          '</div><div class="session-meta">'+esc(s.startedAt?.slice(0,16))+'</div></div>').join('');
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
        '<div class="obs-card imp-'+(o.importance >= 7 ? 'high' : o.importance >= 4 ? 'med' : 'low')+'">' +
        '<div class="obs-title">['+o.importance+'/10] '+esc(o.title)+'</div>' +
        '<div class="obs-meta">'+esc(o.type)+' · '+esc(o.timestamp?.slice(0,16))+'</div>' +
        '<div class="obs-narrative">'+esc(o.narrative?.slice(0,200))+'</div></div>').join('');
      document.getElementById('session-detail').innerHTML =
        '<div class="detail-panel"><h3>Session: '+esc(session.name||session.project||'Untitled')+'</h3>' +
        '<div class="detail-row"><span class="dl">ID</span><span class="dv" style="font-family:var(--font-mono);font-size:11px;">'+esc(session.id)+'</span></div>' +
        '<div class="detail-row"><span class="dl">Status</span><span class="dv"><span class="badge badge-'+(session.status==='active'?'green':'blue')+'">'+esc(session.status)+'</span></span></div>' +
        '<div class="detail-row"><span class="dl">Observations</span><span class="dv">'+session.observationCount+'</span></div>' +
        '<div class="detail-row"><span class="dl">Started</span><span class="dv">'+esc(session.startedAt)+'</span></div>' +
        (session.summary ? '<div class="detail-row"><span class="dl">Summary</span><span class="dv">'+esc(session.summary)+'</span></div>' : '') +
        (session.model ? '<div class="detail-row"><span class="dl">Model</span><span class="dv" style="font-family:var(--font-mono);font-size:11px;">'+esc(session.model)+'</span></div>' : '') +
        '</div>' + (obs.length > 0 ? '<h3 style="margin-top:20px;font-family:var(--font-display);font-size:16px;">Recent Observations ('+obs.length+')</h3>' + oHtml : '');
    } catch (e) { console.error('[session] error:', e); }
  }
  // Audit
  async function loadAudit() {
    try {
      const res = await api('/audit?limit=100');
      const entries = await res.json();
      const opColors = { create:'badge-green', delete:'badge-red', forget:'badge-red', consolidate:'badge-yellow', update:'badge-blue' };
      const html = entries.length === 0
        ? '<div class="empty-state"><div class="empty-icon">&#128221;</div><p>No audit entries</p></div>'
        : entries.map(e => '<tr><td style="font-family:var(--font-mono);font-size:11px;">'+esc(e.timestamp?.slice(0,16))+'</td>' +
          '<td><span class="badge '+(opColors[e.operation]||'badge-muted')+'">'+esc(e.operation)+'</span></td>' +
          '<td>'+esc(e.resource)+'</td><td>'+esc(e.type||'')+'</td>' +
          '<td>'+(e.strength != null ? '<span class="badge badge-'+(e.strength >= 7 ? 'red':'green')+'">'+e.strength+'/10</span>' : '—')+'</td></tr>').join('');
      document.getElementById('audit-body').innerHTML = html;
    } catch (e) { console.error('[audit] error:', e); }
  }
  // Activity
  async function loadActivity() {
    try {
      const res = await api('/activity?limit=50');
      const activity = await res.json();
      const oColors = { file_read:'var(--blue)', file_write:'var(--green)', file_edit:'var(--yellow)', command_run:'var(--orange)', search:'var(--purple)', web_fetch:'var(--cyan)', conversation:'var(--ink-muted)', error:'var(--red)', decision:'var(--purple)', other:'var(--ink-muted)' };
      const html = activity.length === 0
        ? '<div class="empty-state"><div class="empty-icon">&#128245;</div><p>No activity</p></div>'
        : activity.slice(0, 50).map(a => {
          const o = a.observation || {};
          const c = oColors[o.type] || 'var(--border-light)';
          return '<div class="activity-item">' +
            '<div class="activity-dot" style="background:'+c+'"></div>' +
            '<div class="activity-body">' +
            '<div class="activity-title"><span class="badge badge-'+(o.importance >= 7 ? 'red' : o.importance >= 4 ? 'yellow' : 'green')+'">['+o.importance+'/10]</span> '+esc(o.title)+'</div>' +
            '<div class="activity-meta">'+esc(o.type)+' · '+esc(a.sessionProject||'Unknown')+' · '+esc(o.timestamp?.slice(0,16))+'</div>' +
            (o.narrative ? '<p>'+esc(o.narrative.slice(0,150))+'</p>' : '') +
            '</div></div>';
        }).join('');
      document.getElementById('activity-feed').innerHTML = html;
    } catch (e) { console.error('[activity] error:', e); }
  }
  // Profile
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
      const html = '<div class="card"><div class="card-title">Project Overview</div>' +
        '<table><thead><tr><th>Project</th><th>Status</th><th>Obs</th><th>Avg Import</th><th>Memories</th></tr></thead><tbody>' +
        stats.map(s => '<tr><td>'+esc(s.project)+'</td><td><span class="badge badge-'+(s.status==='active'?'green':'blue')+'">'+s.status+'</span></td><td>'+s.observations+'</td><td>'+s.strength.toFixed(1)+'/10</td><td>'+s.memories+'</td></tr>').join('') +
        '</tbody></table></div>';
      document.getElementById('profile-content').innerHTML = html;
    } catch (e) { console.error('[profile] error:', e); }
  }
  // Working Memory
  async function loadWorkingMemory() {
    try {
      const res = await api('/working-memory');
      const tiered = await res.json();
      const rows = Object.entries(tiered).map(([id, info]) => ({ id, ...info })).slice(0, 200);
      const filter = document.getElementById('wm-tier-filter').value;
      const filtered = filter ? rows.filter(r => r.tier === filter) : rows;
      const tierColors = { hot:'badge-red', warm:'badge-yellow', cold:'badge-green', archived:'badge-muted' };
      const html = filtered.length === 0
        ? '<div class="empty-state"><div class="empty-icon">&#129524;</div><p>No memories to display</p></div>'
        : filtered.map(r => '<tr><td><span class="badge '+tierColors[r.tier]+'">'+esc(r.tier)+'</span></td>' +
          '<td>'+esc(r.type)+'</td>' +
          '<td><span class="badge badge-'+(r.strength >= 7 ? 'red':'green')+'">'+r.strength+'/10</span></td>' +
          '<td>'+esc(r.content?.slice(0, 100))+'</td></tr>').join('');
      document.getElementById('wm-body').innerHTML = html;
    } catch (e) { console.error('[wm] error:', e); }
  }
  // WebSocket
  function connectWebSocket() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    state.ws = new WebSocket(proto + '//' + window.location.host + '/ws');
    state.ws.onopen = () => {
      document.getElementById('ws-status').classList.remove('disconnected');
      document.getElementById('ws-status').classList.add('connected');
      document.getElementById('ws-status').textContent = 'live';
      state.ws.send(JSON.stringify({ type: 'subscribe' }));
    };
    state.ws.onclose = () => {
      document.getElementById('ws-status').classList.remove('connected');
      document.getElementById('ws-status').classList.add('disconnected');
      document.getElementById('ws-status').textContent = 'offline';
      setTimeout(connectWebSocket, 3000);
    };
    state.ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === 'observation' && state.activeTab === 'dashboard') loadDashboard();
        if (msg.type === 'observation' && state.activeTab === 'observations') loadObservations();
      } catch {}
    };
  }
  // Event bindings
  document.querySelectorAll('.tab-bar button').forEach(btn => {
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
  function debounce(fn, ms) { let t; return function() { clearTimeout(t); t = setTimeout(fn, ms); }; }
  // Init
  loadDashboard();
  state.refreshInterval = setInterval(() => {
    if (state.activeTab === 'observations') loadObservations();
    else if (state.activeTab === 'dashboard') loadDashboard();
  }, 5000);
  connectWebSocket();
  </script>
</body>
</html>`;

export default HTML;
