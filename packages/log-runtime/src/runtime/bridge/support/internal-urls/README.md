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

Recognize links before checking handler availability. `read_file` routes all
recognized links through `InternalUrlRouter`; unsupported schemes return an
error instead of becoming filesystem paths. Register handlers with that router;
do not maintain another scheme list in callers. Prefix literal filenames that
contain `://` with `./`.

`xd://` is a separate tool-device protocol. `read_file` reads device documentation;
`write_file` dispatches devices. Other links are rejected by `write_file`.

Use `mcp://<server>/<resource-uri>` to select a server explicitly. For example,
`mcp://catalog/resource://items/0?format=json#summary` forwards
`resource://items/0?format=json#summary` unchanged to `catalog`. Missing or unknown
servers return errors; there is no fallback to another client. The old
`mcp://<resource-uri>` form is no longer supported. Bare `mcp://` lists servers.

`skill://<name>` and `rule://<name>` read an exact loaded name, with an optional
trailing slash. Extra paths, queries and fragments are rejected. They do not read
relative files. Pass user-provided skill URLs unchanged to `read_file`;
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

The prompt source is `runtime/context/system-prompt.md`. After editing it,
regenerate its embedded TypeScript export with `node scripts/embed-md.mjs` as
shown in `apps/tui/Makefile`.
