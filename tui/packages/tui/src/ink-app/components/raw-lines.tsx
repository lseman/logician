import { Text } from "ink";
import React from "react";
import { isImageLine } from "../../terminal/core.ts";

/**
 * Bridge for the ~16k lines of existing Component.render(width): string[]
 * implementations (transcript, overlays, status bar, ...). Each already
 * returns fully ANSI-styled, width-clamped rows via visibleWidth/
 * clampLineToWidth (terminal/core.ts), so Ink only needs to place them —
 * it does not need to re-flow or re-measure them.
 *
 * Components rewritten as native Ink JSX skip this and render directly,
 * gaining Ink's own layout/diffing instead of relying on their own
 * render(width) string output.
 */
export function RawLines({ lines }: { lines: readonly string[] }): React.ReactElement {
	return (
		<>
			{lines.map((line, index) => (
				<Text
					// biome-ignore lint/suspicious/noArrayIndexKey: rows are positional, not identity-bearing
					key={index}
					wrap={isImageLine(line) ? "truncate-end" : "wrap"}
				>
					{line || " "}
				</Text>
			))}
		</>
	);
}
