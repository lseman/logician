// ── cfg:// Protocol Handler ───────────────────────────────────────────────────
// The agent's view of Logician settings, driven by the settings schema
// registry. Ported from oh-my-pi's cfg:// protocol.
//
// Read forms (`/` and `.` both separate segments):
//   cfg://                 every setting as a YAML-ish tree
//   cfg://<namespace>      one namespace, e.g. cfg://ttsr
//   cfg://<setting>        one setting: value, type, default, source, description
//
// Write forms (content is the new value). Every write needs the user's
// approval, so only the interactive main session registers this handler —
// subagents and headless runs never see the scheme:
//   cfg://<setting>        apply to the running session (live-applicable keys only)
//   cfg://<setting>/save   persist to ~/.logician/settings.json (and apply live when possible)
//
// Credential-like values are always redacted.

import { validateConfig } from "../../../configuration/config.ts";
import type { ConfigProvenance } from "../../../configuration/config-provenance.ts";
import {
	SETTINGS_SCHEMA,
	type SettingSpec,
} from "../../../configuration/settings-schema.ts";
import type { RuntimeSettingsPatch } from "../../types.ts";
import type {
	InternalResource,
	InternalUrl,
	ProtocolHandler,
	SchemeHost,
	UrlCompletion,
} from "./types.ts";

export const CFG_URL_PREFIX = "cfg://";
const SAVE_SEGMENT = "save";
const REDACTED = "<redacted>";
/** Credential-shaped key names (not token *counts* like maxTokens). */
const CREDENTIAL_RE =
	/(?:api[_-]?key|secret|passw(?:or)?d|credential|(?:auth|access|bearer|refresh|api)[_-]?token|^token$)/i;

/** Where an effective value came from, lowest precedence first. */
export type CfgSource = "default" | "global" | "project" | "env" | "session";

const SOURCE_LABELS: Record<CfgSource, string> = {
	default: "default",
	global: "global config (~/.logician/settings.json)",
	project: "project config (.logician.json)",
	env: "environment variable",
	session: "session override",
};

/** A settings change awaiting the user's decision (values display-formatted). */
export interface CfgChangeRequest {
	path: string;
	previous: string;
	value: string;
	/** Persist to the global config instead of scoping the change to the session. */
	save: boolean;
	/** Layer that will keep a saved value from taking effect here, if any. */
	shadowedBy?: string | undefined;
}

/**
 * `once` applies this change, `session` also approves later writes for the
 * rest of the session (a grant from a save prompt covers saves too), `deny`
 * declines.
 */
export type CfgApproval = "once" | "session" | "deny";

/** Runtime services the handler borrows from its session. */
export interface CfgHost {
	/** Effective config and its provenance, freshly resolved from disk + env. */
	resolve(): {
		config: Record<string, unknown>;
		provenance: ConfigProvenance;
	};
	/** Apply a runtime patch to the live session. */
	applyLive(patch: RuntimeSettingsPatch): void;
	/** Persist a dotted config path to the global config; false on failure. */
	save(path: string, value: unknown): boolean;
	/** Ask the user to approve a change. Dismissal must resolve `deny`. */
	approve(request: CfgChangeRequest): Promise<CfgApproval>;
}

/**
 * Settings the running session can pick up without a restart, and the
 * runtime patch each maps to. Everything else is read once at startup, so a
 * session-only write would be a lie — those keys must be saved.
 */
const LIVE_SETTINGS: Readonly<
	Partial<Record<string, (value: unknown) => RuntimeSettingsPatch>>
> = {
	temperature: value => ({ temperature: value as number }),
	maxTokens: value => ({ maxTokens: value as number }),
	maxIterations: value => ({ maxIterations: value as number }),
	thinkingLevel: value => ({
		thinkingLevel: value as RuntimeSettingsPatch["thinkingLevel"],
	}),
	inferenceMode: value => ({
		inferenceMode: value as RuntimeSettingsPatch["inferenceMode"],
	}),
	executionProfile: value => ({
		executionProfile: value as RuntimeSettingsPatch["executionProfile"],
	}),
	guardsEnabled: value => ({ guardMode: value ? "on" : "off" }),
	duplicateGuardEnabled: value => ({ duplicateGuardEnabled: value as boolean }),
	failureGuardEnabled: value => ({ failureGuardEnabled: value as boolean }),
	progressStopEnabled: value => ({ progressStopEnabled: value as boolean }),
	continuationEnabled: value => ({ continuationEnabled: value as boolean }),
	autoRetryEnabled: value => ({ autoRetryEnabled: value as boolean }),
	rtkProxyEnabled: value => ({ rtkProxyEnabled: value as boolean }),
	graphicianEnabled: value => ({ graphicianEnabled: value as boolean }),
	fffgrepEnabled: value => ({ fffgrepEnabled: value as boolean }),
	steeringInterrupt: value => ({ steeringInterrupt: value as boolean }),
	postEditDiagnostics: value => ({ postEditDiagnostics: value as boolean }),
	"compaction.enabled": value => ({
		proactiveCompactionEnabled: value as boolean,
	}),
	reasoner: value => ({ reasonerId: value as string }),
};

/** Settings addressable through cfg://: every schema leaf, namespaces excluded. */
function allSettings(): string[] {
	const keys = Object.keys(SETTINGS_SCHEMA);
	return keys
		.filter(key => {
			const spec = SETTINGS_SCHEMA[key];
			if (!spec) return false;
			// A registered object with registered children is a namespace.
			return !(
				spec.type === "object" &&
				keys.some(other => other.startsWith(`${key}.`))
			);
		})
		.sort((a, b) => a.localeCompare(b, "en", { sensitivity: "base" }));
}

function getPath(config: Record<string, unknown>, path: string): unknown {
	let node: unknown = config;
	for (const segment of path.split(".")) {
		if (!node || typeof node !== "object" || Array.isArray(node))
			return undefined;
		node = (node as Record<string, unknown>)[segment];
	}
	return node;
}

function setPath(
	target: Record<string, unknown>,
	path: string,
	value: unknown,
): void {
	const segments = path.split(".");
	const leaf = segments.pop() as string;
	let node = target;
	for (const segment of segments) {
		const next: Record<string, unknown> = {};
		node[segment] = next;
		node = next;
	}
	node[leaf] = value;
}

function isCredential(key: string): boolean {
	return key.split(".").some(segment => CREDENTIAL_RE.test(segment));
}

function formatValue(key: string, value: unknown): string {
	if (value === undefined || value === null) return "unset";
	if (isCredential(key) && value !== "") return REDACTED;
	if (typeof value === "boolean" || typeof value === "number")
		return String(value);
	return JSON.stringify(value);
}

function describeType(spec: SettingSpec): string {
	if (spec.type === "number") {
		const low =
			spec.min === undefined
				? ""
				: `${spec.minExclusive ? ">" : ">="}${spec.min}`;
		const high = spec.max === undefined ? "" : `<=${spec.max}`;
		const range = [low, high].filter(Boolean).join(", ");
		return range ? `number (${range})` : "number";
	}
	return spec.type;
}

/**
 * Canonicalize URL segments against the schema, case-insensitively.
 * Returns the leaf setting (if the path names one) and every setting beneath it.
 */
function resolveSegments(segments: readonly string[]): {
	path: string;
	leaf: string | undefined;
	members: string[];
} {
	const settings = allSettings();
	if (segments.length === 0)
		return { path: "", leaf: undefined, members: settings };
	const lower = segments.join(".").toLowerCase();
	const leaf = settings.find(key => key.toLowerCase() === lower);
	const members = settings.filter(key =>
		key.toLowerCase().startsWith(`${lower}.`),
	);
	const path = leaf ?? members[0]?.slice(0, lower.length);
	if (path === undefined) {
		const needle = segments.at(-1)?.toLowerCase() ?? "";
		const similar = settings
			.filter(key => key.toLowerCase().includes(needle))
			.slice(0, 8);
		const hint = similar.length > 0 ? `\nSimilar: ${similar.join(", ")}` : "";
		throw new Error(
			`Unknown setting: ${segments.join(".")}${hint}\nRead ${CFG_URL_PREFIX} for the full tree.`,
		);
	}
	return { path, leaf, members };
}

/** Parse `cfg://a/b[/save]` (dots and slashes both separate segments). */
function parseCfgTarget(url: InternalUrl): {
	segments: string[];
	save: boolean;
} {
	const segments = url.target
		.split(/[/.]/)
		.map(segment => segment.trim())
		.filter(Boolean);
	const save = segments.at(-1)?.toLowerCase() === SAVE_SEGMENT;
	if (save) segments.pop();
	return { segments, save };
}

/** Parse written content into a raw value for `spec` (validation happens after). */
function parseContent(
	key: string,
	spec: SettingSpec,
	content: string,
): unknown {
	const text = content.trim();
	switch (spec.type) {
		case "boolean": {
			const lower = text.toLowerCase();
			if (["true", "on", "yes", "1"].includes(lower)) return true;
			if (["false", "off", "no", "0"].includes(lower)) return false;
			throw new Error(`${key} is a boolean; write true or false.`);
		}
		case "number": {
			const value = Number(text);
			if (text === "" || !Number.isFinite(value)) {
				throw new Error(`${key} is a number; got ${JSON.stringify(text)}.`);
			}
			return value;
		}
		case "enum":
			if (!spec.enum?.includes(text)) {
				throw new Error(`${key} must be one of: ${spec.enum?.join(", ")}.`);
			}
			return text;
		case "string":
		case "url":
			// Accept a JSON-quoted string as well as the bare value.
			if (text.startsWith('"') && text.endsWith('"')) {
				try {
					return JSON.parse(text) as string;
				} catch {
					return text;
				}
			}
			return text;
		default:
			try {
				return JSON.parse(text) as unknown;
			} catch {
				throw new Error(`${key} is a ${spec.type}; write it as JSON.`);
			}
	}
}

/**
 * Validate one value exactly as a config file would be validated, and
 * return the accepted value. Rejects anything the validator warns about or
 * drops, so cfg:// can never write a value the config loader would refuse.
 */
function validateValue(key: string, value: unknown): unknown {
	const raw: Record<string, unknown> = {};
	setPath(raw, key, value);
	const warnings: string[] = [];
	const validated = validateConfig(raw, warnings) as unknown as Record<
		string,
		unknown
	>;
	if (warnings.length > 0) throw new Error(warnings.join(" "));
	const accepted = getPath(validated, key);
	if (accepted === undefined) {
		throw new Error(`${key}: value ${formatValue(key, value)} was rejected.`);
	}
	return accepted;
}

export class CfgProtocolHandler implements ProtocolHandler {
	readonly scheme = "cfg";
	readonly immutable = false;
	readonly spec = { backing: "virtual" as const, selectors: "lines" as const };

	readonly #host: CfgHost;
	/** Session-scoped overrides written through cfg:// (effective until restart). */
	readonly #overrides = new Map<string, unknown>();
	/** Tail of the approval chain: concurrent writes prompt one at a time. */
	#approvalQueue: Promise<unknown> = Promise.resolve();
	/** "Allow for this session" grant; `save` extends it to persisted writes. */
	#grant: { save: boolean } | undefined;

	constructor(host: CfgHost) {
		this.#host = host;
	}

	promptDoc(_host: SchemeHost): string | undefined {
		return [
			"## cfg:// — Logician settings",
			"",
			"Read `cfg://` for every setting, `cfg://<namespace>` (e.g. `cfg://ttsr`) for one group, or `cfg://<setting>` for one setting's value, type, default, source, and description.",
			"Write `cfg://<setting>` to change it for this session (live-applicable settings only) or `cfg://<setting>/save` to persist it. Every write asks the user for approval; only change settings the user asked for or clearly needs.",
		].join("\n");
	}

	async resolve(url: InternalUrl): Promise<InternalResource> {
		const { segments } = parseCfgTarget(url);
		const { path, leaf, members } = resolveSegments(segments);
		const view = this.#view();
		const sections: string[] = [];
		if (leaf) sections.push(this.#renderLeaf(leaf, view));
		if (members.length > 0)
			sections.push(this.#renderTree(members, path, view));
		const content = sections.join("\n\n");
		return {
			url: url.href,
			content,
			contentType: "text/plain",
			size: Buffer.byteLength(content, "utf-8"),
		};
	}

	async write(url: InternalUrl, content: string): Promise<string> {
		const { segments, save } = parseCfgTarget(url);
		const { path, leaf } = resolveSegments(segments);
		if (!leaf) {
			const example = `${CFG_URL_PREFIX}${path ? `${path.replaceAll(".", "/")}/` : ""}<key>`;
			throw new Error(
				`${path || CFG_URL_PREFIX} is a namespace; write a single setting, e.g. ${example}.`,
			);
		}
		const spec = SETTINGS_SCHEMA[leaf] as SettingSpec;
		if (spec.type === "models") {
			throw new Error(
				`${leaf} holds model definitions; ask the user to edit it in their config file.`,
			);
		}
		if (isCredential(leaf)) {
			throw new Error(
				`${leaf} holds a credential; ask the user to set it themselves.`,
			);
		}
		const value = validateValue(leaf, parseContent(leaf, spec, content));
		const live = LIVE_SETTINGS[leaf];
		if (!save && !live) {
			throw new Error(
				`${leaf} is read at startup, so a session-only change would not take effect. Write ${CFG_URL_PREFIX}${leaf.replaceAll(".", "/")}/${SAVE_SEGMENT} to persist it for the next session.`,
			);
		}

		const view = this.#view();
		const previous = view.value(leaf);
		if (!save && Bun.deepEquals(previous, value)) {
			return `${leaf} is already ${formatValue(leaf, value)}; nothing changed.`;
		}
		const source = view.source(leaf);
		const shadowedBy =
			save && (source === "project" || source === "env")
				? SOURCE_LABELS[source]
				: undefined;
		const request: CfgChangeRequest = {
			path: leaf,
			previous: formatValue(leaf, previous),
			value: formatValue(leaf, value),
			save,
			shadowedBy,
		};
		const decision = this.#approvalQueue.then(() => this.#decide(request));
		this.#approvalQueue = decision.catch(() => undefined);
		if ((await decision) === "deny") {
			return `The user declined changing ${leaf} (${request.previous} → ${request.value}). It is unchanged; don't retry unless they ask.`;
		}

		if (save && !this.#host.save(leaf, value)) {
			throw new Error(`Failed to save ${leaf} to the global config.`);
		}
		if (live) {
			this.#host.applyLive(live(value));
			this.#overrides.set(leaf, value);
		}
		const parts = [
			save
				? `Saved ${leaf} = ${request.value} to the global config (was ${request.previous}).`
				: `Set ${leaf} = ${request.value} for this session (was ${request.previous}).`,
		];
		if (save) {
			parts.push(
				live
					? "It is also applied to the running session."
					: "It takes effect in the next session.",
			);
		}
		if (shadowedBy) {
			parts.push(
				`Note: the ${shadowedBy} sets ${leaf} too and wins over the saved global value in this project.`,
			);
		}
		return parts.join(" ");
	}

	async complete(query: string): Promise<UrlCompletion[]> {
		const needle = query.toLowerCase().replaceAll("/", ".");
		return allSettings()
			.filter(key => key.toLowerCase().includes(needle))
			.map(key => {
				const description = SETTINGS_SCHEMA[key]?.ui?.description;
				return {
					value: key.replaceAll(".", "/"),
					...(description ? { description } : {}),
				};
			});
	}

	// ── Internals ────────────────────────────────────────────────────────────

	async #decide(request: CfgChangeRequest): Promise<CfgApproval> {
		const grant = this.#grant;
		if (grant && (grant.save || !request.save)) return "once";
		const answer = await this.#host.approve(request);
		if (answer === "session") {
			this.#grant = { save: request.save || (grant?.save ?? false) };
		}
		return answer;
	}

	/** Effective values and sources: resolved config, then session overrides. */
	#view(): {
		value: (key: string) => unknown;
		source: (key: string) => CfgSource;
	} {
		const { config, provenance } = this.#host.resolve();
		const layers = new Map(provenance.map(entry => [entry.key, entry.layer]));
		return {
			value: key =>
				this.#overrides.has(key)
					? this.#overrides.get(key)
					: (getPath(config, key) ?? SETTINGS_SCHEMA[key]?.default),
			source: key => {
				if (this.#overrides.has(key)) return "session";
				// Arrays and passthrough objects are attributed at their top-level key.
				const segments = key.split(".");
				for (let depth = segments.length; depth > 0; depth--) {
					const layer = layers.get(segments.slice(0, depth).join("."));
					if (layer) return layer;
				}
				return "default";
			},
		};
	}

	#renderLeaf(
		key: string,
		view: {
			value: (key: string) => unknown;
			source: (key: string) => CfgSource;
		},
	): string {
		const spec = SETTINGS_SCHEMA[key] as SettingSpec;
		const lines = [
			`${key}: ${formatValue(key, view.value(key))}`,
			`type: ${describeType(spec)}`,
			`default: ${formatValue(key, spec.default)}`,
			`source: ${SOURCE_LABELS[view.source(key)]}`,
			`applies: ${LIVE_SETTINGS[key] ? "live (session writes allowed)" : "at startup (write /save)"}`,
		];
		if (spec.enum) lines.push(`values: [${spec.enum.join(", ")}]`);
		if (spec.ui?.description) lines.push(`description: ${spec.ui.description}`);
		return lines.join("\n");
	}

	#renderTree(
		members: readonly string[],
		prefix: string,
		view: {
			value: (key: string) => unknown;
			source: (key: string) => CfgSource;
		},
	): string {
		const lines: string[] = [];
		const opened: string[] = [];
		const strip = prefix ? prefix.length + 1 : 0;
		for (const key of members) {
			const segments = key.slice(strip).split(".");
			let shared = 0;
			while (
				shared < opened.length &&
				shared < segments.length - 1 &&
				opened[shared] === segments[shared]
			)
				shared++;
			opened.length = shared;
			for (let depth = shared; depth < segments.length - 1; depth++) {
				lines.push(`${"  ".repeat(depth)}${segments[depth]}:`);
				opened.push(segments[depth] as string);
			}
			const spec = SETTINGS_SCHEMA[key] as SettingSpec;
			const value = view.value(key);
			const notes: string[] = [];
			if (spec.enum) notes.push(spec.enum.join("|"));
			const source = view.source(key);
			if (source !== "default") notes.push(source);
			if (spec.ui?.description) notes.push(spec.ui.description);
			const comment = notes.length > 0 ? `  # ${notes.join(" · ")}` : "";
			lines.push(
				`${"  ".repeat(segments.length - 1)}${segments.at(-1)}: ${formatValue(key, value)}${comment}`,
			);
		}
		return lines.join("\n");
	}
}
