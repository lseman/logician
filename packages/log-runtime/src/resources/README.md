# Internal links

Internal links use `scheme://target`. Schemes are case-insensitive and contain
at least two characters so Windows drive paths stay filesystem paths. Resource
names and targets are case-sensitive. `parse.ts` owns recognition and parsing.
Do not use the browser `URL` parser for resource names: `skill://plugin:name`
contains a name, not a host and port.

The parsed link has five fields:

- `scheme`: lowercase scheme.
- `target`: everything after `://`, unchanged.
- `host`: the first target segment, before `/`, `?` or `#`, including colons.
- `pathname`: the remainder if it begins with `/`, otherwise `/`. This is a
  resource path, not a decoded browser URL pathname.
- `href`: lowercase scheme plus the original target. No slash is added.

Protocols interpret targets themselves. MCP preserves nested URIs verbatim;
SSH parses `[user@]host[:port]` locally, with bracketed IPv6 hosts. The shared
link type has no browser URL aliases, credentials or placeholder query fields.

Recognize links before checking handler availability. `read` routes all
recognized links through `InternalUrlRouter`; unsupported schemes return an
error instead of becoming filesystem paths. Register handlers with that router;
do not maintain another scheme list in callers. Prefix literal filenames that
contain `://` with `./`.

`xd://` is a registered tool-device protocol. `read` reads device documentation;
`write` dispatches devices through its own `resolveCall` path (see "Reading and
device dispatch" below). Other schemes go through the generic write dispatch
described in "Mutability and write dispatch"; a scheme with no `write()`
handler is rejected.

Use `mcp://<server>/<resource-uri>` to select a server explicitly. For example,
`mcp://catalog/resource://items/0?format=json#summary` forwards
`resource://items/0?format=json#summary` unchanged to `catalog`. Missing or unknown
servers return errors; there is no fallback to another client. The old
`mcp://<resource-uri>` form is no longer supported. Bare `mcp://` lists servers.

`skill://<name>` reads an exact loaded name, with an optional trailing slash.
Extra paths, queries and fragments are rejected. It does not read relative files.
Pass user-provided skill URLs unchanged to `read`;
`read_skill` remains the tool for formatted skill invocation from the catalog.

`pr://owner/repo/1428` reads a GitHub pull request via the GitHub MCP server.
Supported paths: `pr://owner/repo` (list open PRs),
`pr://owner/repo/<number>` (PR details),
`pr://owner/repo/<number>/files` (changed files),
`pr://owner/repo/<number>/comments` (comments),
`pr://owner/repo/<number>/reviews` (reviews),
`pr://owner/repo/<number>/commits` (commits).

`issue://owner/repo/42` reads a GitHub issue via the GitHub MCP server.
Supported paths: `issue://owner/repo` (list open issues),
`issue://owner/repo/<number>` (issue details),
`issue://owner/repo/<number>/comments` (comments),
`issue://owner/repo/<number>/sub-issues` (sub-issues).

Both require a configured GitHub MCP server; they return a diagnostic message
when unavailable.

The prompt source is `context/system-prompt.md`. After editing it,
regenerate its embedded TypeScript export with `node scripts/embed-md.mjs` as
shown in `apps/tui/Makefile`.

## Reading and device dispatch

`createReadTool(router)` is a generic text-resource reader; its public tool name
is `read`. `readResource` resolves filesystem paths
or delegates recognized URLs to the injected router. `formatResourceRead` then
applies the same numbered lines, positive 1-based `offset`/`limit`, and line/byte
caps to files, directory listings, device documentation, and protocol content.
Only direct filesystem reads grant hashline edit anchors and update read tracking.
A protocol's `sourcePath` is provenance, not permission to edit its backing file.
Read results are not cached because files and resource catalogs can change.

Each `ToolRouter` owns its handler table and `XdDeviceRegistry`. Standalone
`createDefaultTools()` calls also get their own instances. Register a
`ProtocolHandler` on that router to support another resource scheme; the read
tool needs no scheme-specific branch. Handlers receive cwd, allowed paths,
loaded skills/rules/memory, and an abort signal. They must enforce their backing
store's access boundaries and cooperate with cancellation. Existing backend
services (such as agent output and artifact storage) retain their own lifecycles;
a separate handler table does not itself isolate those services.

`XdDeviceRegistry` implements the `xd` read protocol and owns device discovery,
documentation, and JSON-object decoding. Mount/unmount changes are reflected in
subsequent reads. The session refreshes its catalog when optional or MCP tools
are added or toggled. `tools.xdev=false` leaves the catalog empty. Device
addresses supplement the existing tool inventory; this does not change which
schemas are advertised to providers.

Device execution belongs to the core `ToolRegistry`, not to protocol handlers.
The write tool's `resolveCall` maps `write(path="xd://name", content="{...}")`
to the registered target during preparation, before hooks, permissions,
scheduling, and execution. The original call ID is preserved; execution events
identify the target tool. Target argument preparation, context, cancellation,
timeouts, updates, and structured results work exactly as for a direct call.
A catalog entry cannot execute a tool absent from the active execution registry.
Calling `write.execute` directly is intentionally insufficient for device
execution; use the normal registry/harness path. Append and non-object JSON
payloads are rejected.

## Mutability and write dispatch

`ProtocolHandler.immutable` is required on every handler — a stance on
whether its resources can be edited, independent of whether a `write()` hook
actually exists yet (e.g. `ssh://` currently declares `immutable: false` while
still lacking `write()`). `InternalUrlRouter.resolve()` stamps
`resource.immutable ?? handler.immutable` onto every resolved resource; a
handler that sets its own per-resource `immutable` wins.

A handler may implement optional `write(url, content, context?)`. The write
tool checks `xd://` first (unchanged, dispatched via `resolveCall` through
`ToolRegistry` as above); for every other recognized scheme it calls
`InternalUrlRouter.write(path, content, ctx)`, which looks up the handler and
either dispatches to `write()` or throws `"<scheme>:// is read-only for
write"`. `append` is not supported for any protocol scheme.

`local://` (session artifacts under `.logician/artifacts/`) is the first
mutable protocol, sandboxed the same way its reads are. Protocol writes do
**not** get the file tool's read-before-overwrite staleness protection —
resource reads are not tracked by `read-tracker.ts` the way direct filesystem
reads are (see "Reading and device dispatch" above: only `kind: "file"` reads
call `recordRead()`). Extending read-tracking to resource reads, so protocol
writes get the same guarantee, is a known gap and would need to apply to
every protocol at once, not just `local://`.

`rag://` (Retrieval-Augmented Generation) operations on the vector store.
`read rag://` shows usage help; `read rag://list` lists indexed document IDs.
`write` dispatches JSON payloads: `write rag://search` with `{"query":"...","k?:number"}`
returns search results as JSON; `write rag://ingest` with `{"path":"...","docId?:string"}`
ingests a document; `write rag://delete` with `{"docId":"..."}` removes it.
All operations require `cwd` from the write context and access the shared RAG pipeline singleton.

## Completion and path-only resolution

`ProtocolHandler.complete?(query, context?)` returns autocomplete candidates
for the host/path portion of `scheme://<query>`; implementations must be fast
and local. `InternalUrlRouter.completionSchemes()` lists which registered
schemes support it, and `router.complete(scheme, query, context?)` dispatches
to that handler, returning `null` (not an error) for a scheme with none.
`ArtifactProtocolHandler` and `SshProtocolHandler` implement it today; no
caller in `apps/tui` consumes it yet (the only `://` autocomplete UI,
`overlays/skill-popup.ts`, is a hardcoded skill-only popup with no router
dependency) — this capability is available and tested, not yet wired to a UI.

`ResolveContext.pathOnly` tells a handler that the caller only needs
`sourcePath`/shape info, not materialized content — useful when content would
be expensive to read (a large file, a remote listing). `local://` honors it;
every other handler ignores it since they have no real backing path to substitute for content. No caller in logician sets it yet (`ToolContext` in `packages/log-core` has no `pathOnly` field) — like `complete()` before this, it's forward-compatible handler support without a live caller.

## Known limitation: single session per process

`ArtifactRegistry.instance()` is a process-global singleton; its `init()` is
called once from `ToolRouter`'s constructor. This is safe today because
`ToolRouter` is constructed exactly once per `AgentBridge`. If logician ever
hosts multiple sessions in one process, a second `init()` call would silently
repoint `local://` artifact-ID resolution for every session to the new session's
directory. Fixing this would mean a per-`ToolRouter` `ArtifactManager` (or an
`ArtifactRegistry` keyed by session id), not threading session identity
through every `ProtocolHandler`'s `ResolveContext` the way a multi-session
host like oh-my-pi's does — that's more machinery than logician's current
single-session-per-process model justifies.
