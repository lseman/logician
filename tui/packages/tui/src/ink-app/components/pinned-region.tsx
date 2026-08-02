import React from "react";
import type { InkTextRow } from "../../terminal/core.ts";
import { InkTextRows } from "./status-region.tsx";

/** Native Ink stack for notifications, tasks, work surface, and steer queue. */
export function PinnedRegion({ rows }: { rows: readonly InkTextRow[] }): React.ReactElement | null {
	return <InkTextRows rows={rows} />;
}
