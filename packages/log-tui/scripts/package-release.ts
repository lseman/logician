import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
	cpSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const version = (process.env.RELEASE_VERSION ?? "").replace(/^v/, "");
const platform = process.env.RELEASE_PLATFORM ?? "";
if (!/^\d+\.\d+\.\d+(?:[-+][\w.-]+)?$/.test(version))
	throw new Error("Set RELEASE_VERSION to a release version");
if (
	!["linux-x86_64", "linux-arm64", "darwin-x86_64", "darwin-arm64"].includes(
		platform,
	)
)
	throw new Error("Invalid RELEASE_PLATFORM");
const dist = resolve(import.meta.dir, "../dist");
const temp = mkdtempSync(join(tmpdir(), "logician-release-"));
try {
	const name = `logician-${version}`;
	const app = join(temp, name);
	mkdirSync(app);
	cpSync(join(dist, "logician"), join(app, "logician"));
	cpSync(join(dist, "themes"), join(app, "themes"), { recursive: true });
	const archive = join(dist, `logician-${platform}.tar.gz`);
	const tar = spawnSync("tar", ["-czf", archive, "-C", temp, name], {
		stdio: "inherit",
	});
	if (tar.status !== 0) throw new Error("Archive creation failed");
	const unpack = join(temp, "unpacked");
	const home = join(temp, "home");
	mkdirSync(unpack);
	mkdirSync(home);
	const extracted = spawnSync("tar", ["-xzf", archive, "-C", unpack]);
	if (extracted.status !== 0) throw new Error("Archive extraction failed");
	const result = spawnSync(
		join(unpack, name, "logician"),
		["doctor", "--json"],
		{
			cwd: home,
			env: {
				PATH: process.env.PATH,
				HOME: home,
				LOGICIAN_THEME: "github-dark",
			},
			encoding: "utf8",
			timeout: 30000,
		},
	);
	if (result.status !== 0)
		throw new Error(
			`Packaged doctor failed: ${result.error ?? result.stderr}\n${result.stdout}`,
		);
	const report = JSON.parse(result.stdout);
	if (
		!report.native?.healthy ||
		!report.installation?.theme?.valid ||
		!report.installation?.revision
	)
		throw new Error("Packaged installation is incomplete");
	if (
		report.installation.theme.path !==
		join(unpack, name, "themes", "github-dark.json")
	)
		throw new Error("Packaged theme resolved outside the archive");
	const digest = createHash("sha256")
		.update(readFileSync(archive))
		.digest("hex");
	writeFileSync(
		`${archive}.sha256`,
		`${digest}  logician-${platform}.tar.gz\n`,
	);
	console.log(
		`Verified ${archive}: native capabilities, themes, build revision`,
	);
} finally {
	rmSync(temp, { recursive: true, force: true });
}
