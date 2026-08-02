import { Text } from "ink";
import React from "react";

/**
 * Bridge for the ~16k lines of existing Component.render(width): string[]
 * implementations (transcript, overlays, status bar, ...). Each already
 * returns fully ANSI-styled, width-clamped rows via the same
 * visibleWidth/clampLineToWidth primitives the legacy renderer uses, so Ink
 * only needs to place them — it does not need to re-flow or re-measure them.
 *
 * This is a migration shim: components rewritten as native Ink JSX skip this
 * and render directly, gaining Ink's own layout/diffing instead of relying
 * on their own render(width) string output.
 */
export function RawLines({ lines }: { lines: readonly string[] }): React.ReactElement {
	return (
		<>
			{lines.map((line, index) => (
				// biome-ignore lint/suspicious/noArrayIndexKey: rows are positional, not identity-bearing
				<Text key={index}>{line || " "}</Text>
			))}
		</>
	);
}
