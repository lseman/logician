// ── Ink TUI — File Mention Popup ──────────────────────────────────────────────
// Inline @-mention autocomplete. `files` is pre-filtered by the caller; this
// component ranks and renders them and owns list navigation.

import React, { useMemo } from "react";
import { Box, Text } from "ink";
import { getCurrentTheme } from "../theme";
import { useOverlayInput } from "../hooks/useOverlayInput";

interface FileMentionPopupProps {
	files: string[];
	query: string;
	isActive: boolean;
	onSelect: (path: string) => void;
	onClose: () => void;
	maxVisible?: number;
}

function score(query: string, path: string): number {
	const p = path.toLowerCase();
	const base = p.slice(p.lastIndexOf("/") + 1);
	if (!query) return 1;
	if (base.startsWith(query)) return 2500 - (base.length - query.length);
	if (p.startsWith(query)) return 2200;
	if (base.includes(query)) return 2000 - base.indexOf(query) * 8;
	if (p.includes(query)) return 1500 - p.indexOf(query) * 4;
	return -1;
}

export const FileMentionPopup: React.FC<FileMentionPopupProps> = ({
	files,
	query,
	isActive,
	onSelect,
	onClose,
	maxVisible = 8,
}) => {
	const theme = getCurrentTheme();
	const q = query.toLowerCase();

	const matches = useMemo(
		() =>
			files
				.map(path => ({ path, s: score(q, path) }))
				.filter(m => m.s > 0)
				.sort((a, b) => b.s - a.s)
				.slice(0, maxVisible)
				.map(m => m.path),
		[files, q, maxVisible],
	);

	const { index } = useOverlayInput({
		isActive,
		count: matches.length,
		onSelect: i => {
			const p = matches[i];
			if (p) onSelect(p);
		},
		onClose,
	});

	if (matches.length === 0) return null;

	return (
		<Box
			borderColor={theme.fg.accent as string}
			borderStyle="round"
			paddingX={1}
			flexDirection="column"
			minWidth={40}
		>
			<Text color={theme.fg.accent as string} bold>
				{`@${query}`}
			</Text>
			{matches.map((path, i) => (
				<Text
					key={path}
					color={
						i === index ? (theme.fg.selected as string) : (theme.fg.primary as string)
					}
					bold={i === index}
				>
					{`${i === index ? "▸ " : "  "}${path}`}
				</Text>
			))}
		</Box>
	);
};
