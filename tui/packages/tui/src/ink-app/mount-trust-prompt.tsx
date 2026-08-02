import { Box, render, useApp, useInput, useStdout } from "ink";
import React, { useState } from "react";
import type { TrustChoice } from "../overlays/trust-prompt-overlay.ts";
import { TrustPromptOverlay } from "../overlays/trust-prompt-overlay.ts";
import { ListOverlay } from "./components/list-overlay.tsx";

function TrustPrompt({ overlay, onChoice }: {
	overlay: TrustPromptOverlay;
	onChoice: (choice: TrustChoice) => void;
}): React.ReactElement {
	const { exit } = useApp();
	const { stdout } = useStdout();
	const [, setTick] = useState(0);
	useInput((input, key) => {
		const sequence = key.escape ? "\x1b"
			: key.return ? "\r"
				: key.upArrow ? "\x1b[A"
					: key.downArrow ? "\x1b[B" : input;
		const action = overlay.handleInput(sequence);
		if (action?.type === "trust-choice") {
			onChoice(action.choice);
			exit();
			return;
		}
		setTick((tick) => tick + 1);
	});
	const width = Math.max(24, Math.min(stdout.columns ?? 80, 78));
	return (
		<Box
			width={stdout.columns ?? 80}
			height={stdout.rows ?? 24}
			alignItems="center"
			justifyContent="flex-end"
			paddingBottom={2}
		>
			<ListOverlay model={overlay.getInkOverlayModel()} width={width} />
		</Box>
	);
}

export async function showTrustOverlayInk(cwd: string, paths: string[]): Promise<TrustChoice> {
	const overlay = new TrustPromptOverlay();
	overlay.setOptions({ cwd, paths });
	overlay.show();
	return new Promise((resolve) => {
		render(<TrustPrompt overlay={overlay} onChoice={resolve} />, {
			alternateScreen: true,
		});
	});
}
