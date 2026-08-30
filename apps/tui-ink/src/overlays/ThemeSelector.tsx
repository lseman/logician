// ── Ink TUI — Theme Selector ──────────────────────────────────────────────────

import React from "react";
import { Box, Text } from "ink";
import { getCurrentTheme, getAvailableThemes, setCurrentTheme } from "../theme";
import { useOverlayInput } from "../hooks/useOverlayInput";

interface ThemeSelectorProps {
	currentTheme: string;
	isActive: boolean;
	onSelect: (themeName: string) => void;
	onClose: () => void;
}

export const ThemeSelector: React.FC<ThemeSelectorProps> = ({
	currentTheme,
	isActive,
	onSelect,
	onClose,
}) => {
	const theme = getCurrentTheme();
	const names = getAvailableThemes();

	const { index } = useOverlayInput({
		isActive,
		count: names.length,
		onSelect: i => {
			const name = names[i];
			if (name && setCurrentTheme(name)) onSelect(name);
		},
		onClose,
	});

	return (
		<Box
			borderColor={theme.fg.border}
			borderStyle="round"
			paddingX={1}
			flexDirection="column"
			minWidth={36}
		>
			<Text color={theme.fg.header} bold>
				Select Theme
			</Text>
			{names.map((name, i) => (
				<Text
					key={name}
					color={
						i === index ? (theme.fg.selected as string) : (theme.fg.text as string)
					}
					bold={i === index}
				>
					{`${i === index ? "▸ " : "  "}${name}${name === currentTheme ? "  ✓" : ""}`}
				</Text>
			))}
		</Box>
	);
};
