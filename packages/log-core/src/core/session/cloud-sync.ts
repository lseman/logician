import { execFile } from "node:child_process";
import { existsSync, writeFileSync } from "node:fs";
import path from "node:path";

/** Best-effort marker preventing ephemeral session state from cloud syncing. */
export function markPathIgnoredByCloudSync(dirPath: string): void {
	try {
		if (process.platform === "darwin") {
			execFile("xattr", [
				"-w",
				"com.apple.metadata:com_apple_backup_excludeItem",
				"com.apple.backupd",
				dirPath,
			]);
		}
		const marker = path.join(dirPath, ".noindex");
		if (!existsSync(marker)) writeFileSync(marker, "");
	} catch {
		// Cloud-sync exclusion is advisory and must never break persistence.
	}
}
