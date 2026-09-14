/**
 * Archive member read/write for the zip family (.zip, .jar, .war, .ear,
 * .apk) and the tar family (.tar, .tar.gz, .tgz), addressed via
 * `archive.ext:path/inside/archive`. Other archive formats (.rar, .7z,
 * .iso, ...) are not supported — logician has no native binding layer to
 * lean on for those, unlike oh-my-pi's Rust-backed archive reader.
 */

import { randomUUID } from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";
import * as zlib from "node:zlib";
import AdmZip from "adm-zip";
import * as tar from "tar-stream";
import { formatSize } from "./utils/truncate.ts";

export type ArchiveFamily = "zip" | "tar";

const ZIP_EXTENSIONS = [".zip", ".jar", ".war", ".ear", ".apk"];
const TAR_GZ_EXTENSIONS = [".tar.gz", ".tgz"];

export const MAX_ARCHIVE_MEMBER_BYTES = 8 * 1024 * 1024; // matches artifact-protocol.ts's MAX_ARTIFACT_BYTES

export function archiveFamilyFromPath(absolutePath: string): ArchiveFamily | undefined {
	const lower = absolutePath.toLowerCase();
	if (ZIP_EXTENSIONS.some(ext => lower.endsWith(ext))) return "zip";
	if (TAR_GZ_EXTENSIONS.some(ext => lower.endsWith(ext)) || lower.endsWith(".tar"))
		return "tar";
	return undefined;
}

function isGzipped(absolutePath: string): boolean {
	return TAR_GZ_EXTENSIONS.some(ext => absolutePath.toLowerCase().endsWith(ext));
}

function assertSafeMemberPath(memberPath: string): void {
	if (!memberPath || memberPath.endsWith("/")) {
		throw new Error(`Invalid archive member path: ${memberPath || "(empty)"}`);
	}
	const segments = memberPath.split("/");
	if (segments.some(segment => segment === "..")) {
		throw new Error(`Archive member path may not contain "..": ${memberPath}`);
	}
}

function tempPathFor(absolutePath: string): string {
	return path.join(
		path.dirname(absolutePath),
		`.${path.basename(absolutePath)}.logician-${process.pid}-${randomUUID()}.tmp`,
	);
}

async function renameOver(tempPath: string, absolutePath: string): Promise<void> {
	await fs.promises.mkdir(path.dirname(absolutePath), { recursive: true });
	await fs.promises.rename(tempPath, absolutePath);
}

// ── zip family ──────────────────────────────────────────────────────────────

function openZip(absolutePath: string): AdmZip {
	return fs.existsSync(absolutePath) ? new AdmZip(absolutePath) : new AdmZip();
}

function listZipEntries(absolutePath: string): string {
	const zip = openZip(absolutePath);
	const names = zip
		.getEntries()
		.map(entry => entry.entryName)
		.sort();
	return names.join("\n");
}

function readZipMember(absolutePath: string, memberPath: string): { content: Buffer; size: number } {
	const zip = new AdmZip(absolutePath);
	const entry = zip.getEntry(memberPath);
	if (!entry) throw new Error(`No such archive member: ${memberPath}`);
	if (entry.header.size > MAX_ARCHIVE_MEMBER_BYTES) {
		throw new Error(
			`Archive member ${memberPath} is ${formatSize(entry.header.size)}, exceeding the ${formatSize(MAX_ARCHIVE_MEMBER_BYTES)} limit.`,
		);
	}
	const content = zip.readFile(entry);
	if (!content) throw new Error(`Failed to read archive member: ${memberPath}`);
	return { content, size: content.length };
}

async function writeZipMember(
	absolutePath: string,
	memberPath: string,
	content: string,
): Promise<{ created: boolean }> {
	const containerExists = fs.existsSync(absolutePath);
	const zip = openZip(absolutePath);
	const existingEntry = zip.getEntry(memberPath);
	const buffer = Buffer.from(content, "utf-8");
	if (existingEntry) zip.updateFile(existingEntry, buffer);
	else zip.addFile(memberPath, buffer);

	// adm-zip's async writeZipPromise()/toBufferPromise() has a reproducible bug
	// re-serializing a zip that mixes disk-loaded entries with newly added ones
	// ("Invalid LOC header (bad signature)", crashes the process under Node).
	// The synchronous toBuffer() takes a different, unaffected code path.
	const tempPath = tempPathFor(absolutePath);
	await fs.promises.writeFile(tempPath, zip.toBuffer());
	await renameOver(tempPath, absolutePath);
	return { created: !containerExists || !existingEntry };
}

// ── tar family ──────────────────────────────────────────────────────────────

interface TarEntry {
	header: tar.Header;
	data: Buffer;
}

/**
 * `.pipe()` does not forward source-stream errors to the destination, so a
 * corrupt file or invalid gzip data would otherwise emit an unhandled
 * 'error' event and crash the process. Every stream in the chain forwards
 * its errors into `extractStream.destroy()` so a `for await` consumer gets
 * a clean rejection instead.
 */
function openTarExtractStream(absolutePath: string): tar.Extract {
	const extractStream = tar.extract();
	const fileStream = fs.createReadStream(absolutePath);
	fileStream.on("error", err => extractStream.destroy(err));
	if (isGzipped(absolutePath)) {
		const gunzip = zlib.createGunzip();
		gunzip.on("error", err => extractStream.destroy(err));
		fileStream.pipe(gunzip).pipe(extractStream);
	} else {
		fileStream.pipe(extractStream);
	}
	return extractStream;
}

/** Header-only pass — discards entry data without buffering it, for listing. */
async function listTarEntries(absolutePath: string): Promise<string> {
	const names: string[] = [];
	for await (const entry of openTarExtractStream(absolutePath)) {
		if (entry.header.type === "file") names.push(entry.header.name);
		entry.resume();
	}
	return names.sort().join("\n");
}

/** Full pass buffering every entry's data — needed to rebuild the archive on write, or to read one member. */
async function readTarEntries(absolutePath: string): Promise<TarEntry[]> {
	const entries: TarEntry[] = [];
	for await (const entry of openTarExtractStream(absolutePath)) {
		const chunks: Buffer[] = [];
		for await (const chunk of entry) chunks.push(chunk as Buffer);
		entries.push({ header: entry.header, data: Buffer.concat(chunks) });
	}
	return entries;
}

function readTarMember(entries: TarEntry[], memberPath: string): { content: Buffer; size: number } {
	const entry = entries.find(e => e.header.name === memberPath && e.header.type === "file");
	if (!entry) throw new Error(`No such archive member: ${memberPath}`);
	if (entry.data.length > MAX_ARCHIVE_MEMBER_BYTES) {
		throw new Error(
			`Archive member ${memberPath} is ${formatSize(entry.data.length)}, exceeding the ${formatSize(MAX_ARCHIVE_MEMBER_BYTES)} limit.`,
		);
	}
	return { content: entry.data, size: entry.data.length };
}

async function writeTarMember(
	absolutePath: string,
	memberPath: string,
	content: string,
): Promise<{ created: boolean }> {
	const containerExists = fs.existsSync(absolutePath);
	const entries = containerExists ? await readTarEntries(absolutePath) : [];
	const buffer = Buffer.from(content, "utf-8");
	const existingIndex = entries.findIndex(
		e => e.header.name === memberPath && e.header.type === "file",
	);
	const created = existingIndex === -1;
	const newEntry: TarEntry = {
		header: { name: memberPath, size: buffer.length, type: "file", mode: 0o644 } as tar.Header,
		data: buffer,
	};
	if (existingIndex === -1) entries.push(newEntry);
	else entries[existingIndex] = newEntry;

	const tempPath = tempPathFor(absolutePath);
	// As in openTarExtractStream: .pipe() doesn't forward errors, so every
	// stream in the chain reports into `reject`, and completion is signaled
	// by the actual file stream's "finish" (data flushed to disk), not an
	// upstream transform's.
	await new Promise<void>((resolve, reject) => {
		const packStream = tar.pack();
		const fileStream = fs.createWriteStream(tempPath);
		fileStream.on("error", reject);
		fileStream.on("finish", resolve);
		packStream.on("error", reject);

		if (isGzipped(absolutePath)) {
			const gzip = zlib.createGzip();
			gzip.on("error", reject);
			packStream.pipe(gzip).pipe(fileStream);
		} else {
			packStream.pipe(fileStream);
		}

		for (const entry of entries) {
			packStream.entry({ name: entry.header.name, size: entry.data.length, mode: entry.header.mode ?? 0o644 }, entry.data);
		}
		packStream.finalize();
	});
	await renameOver(tempPath, absolutePath);
	return { created };
}

// ── public API ──────────────────────────────────────────────────────────────

export async function listArchiveEntries(
	absolutePath: string,
	family: ArchiveFamily,
): Promise<string> {
	if (family === "zip") return listZipEntries(absolutePath);
	return listTarEntries(absolutePath);
}

export async function readArchiveMember(
	absolutePath: string,
	family: ArchiveFamily,
	memberPath: string,
): Promise<{ content: Buffer; size: number }> {
	assertSafeMemberPath(memberPath);
	if (family === "zip") return readZipMember(absolutePath, memberPath);
	const entries = await readTarEntries(absolutePath);
	return readTarMember(entries, memberPath);
}

export async function writeArchiveMember(
	absolutePath: string,
	family: ArchiveFamily,
	memberPath: string,
	content: string,
): Promise<{ created: boolean }> {
	assertSafeMemberPath(memberPath);
	if (family === "zip") return writeZipMember(absolutePath, memberPath, content);
	return writeTarMember(absolutePath, memberPath, content);
}
