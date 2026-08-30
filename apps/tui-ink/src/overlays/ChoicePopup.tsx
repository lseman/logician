// ── Ink TUI — Choice Popup (question_request) ─────────────────────────────────

import React, { useState } from "react";
import { Box, Text } from "ink";
import { getCurrentTheme } from "../theme";
import { useOverlayInput } from "../hooks/useOverlayInput";

interface Choice {
	value: string;
	label: string;
	description?: string;
}

interface ChoiceQuestion {
	id: string;
	header?: string;
	question: string;
	choices: Choice[];
}

interface ChoicePopupProps {
	questionId: string;
	questions: ChoiceQuestion[];
	isActive: boolean;
	onSubmit: (questionId: string, answer: string) => void;
	onClose: () => void;
}

export const ChoicePopup: React.FC<ChoicePopupProps> = ({
	questionId,
	questions,
	isActive,
	onSubmit,
	onClose,
}) => {
	const theme = getCurrentTheme();
	const [step, setStep] = useState(0);
	const [answers, setAnswers] = useState<Record<string, string>>({});

	const question = questions[step];
	const choices = question?.choices ?? [];

	const { index } = useOverlayInput({
		isActive,
		count: choices.length,
		onClose,
		onSelect: i => {
			const choice = choices[i];
			if (!question || !choice) return;
			const next = { ...answers, [question.id]: choice.value };
			if (step + 1 < questions.length) {
				setAnswers(next);
				setStep(step + 1);
				return;
			}
			const answer =
				questions.length === 1
					? (next[questions[0]!.id] ?? "")
					: JSON.stringify(next);
			onSubmit(questionId, answer);
		},
	});

	if (!question) return null;

	return (
		<Box
			borderColor={theme.fg.accent as string}
			borderStyle="round"
			paddingX={1}
			flexDirection="column"
			minWidth={48}
		>
			<Text color={theme.fg.accent as string} bold>
				{question.header ?? "Choose"}
				{questions.length > 1 ? `  (${step + 1}/${questions.length})` : ""}
			</Text>
			<Text color={theme.fg.primary as string} wrap="wrap">
				{question.question}
			</Text>
			<Box flexDirection="column" marginTop={1}>
				{choices.map((choice, i) => (
					<Text
						key={choice.value}
						color={
							i === index
								? (theme.fg.selected as string)
								: (theme.fg.primary as string)
						}
						bold={i === index}
					>
						{`${i === index ? "▸ " : "  "}${choice.label}`}
						{choice.description ? ` — ${choice.description}` : ""}
					</Text>
				))}
			</Box>
		</Box>
	);
};
