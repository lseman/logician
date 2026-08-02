#!/usr/bin/env node
// ── Ink renderer proof-of-concept ───────────────────────────────────────────
// Standalone vertical slice: alt-screen, resize-safe fixed dock (input bar +
// status bar) above a scrollable transcript, all laid out by Ink instead of
// manual row-slicing + absolute cursor addressing. Uses the real
// TranscriptDisplay/InputBar/StatusBar components with synthetic content —
// not yet wired to LogicianTUI's bridge/overlays/slash-command orchestration.
// Run with: npx tsx src/ink-app/demo-main.tsx

import { render, useApp, useInput } from "ink";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { initTheme } from "../terminal/theme.ts";
import { TranscriptDisplay } from "../rendering/transcript/display.ts";
import { InputBar } from "../input/input-bar.ts";
import { StatusBar } from "../status/status-bar.ts";
import type { Turn } from "@logician/coding-agent/sessions";
import { AppShell } from "./app-shell.tsx";

initTheme();

function buildDemoTurns(): Turn[] {
	const turns: Turn[] = [];
	for (let i = 0; i < 30; i++) {
		turns.push({
			id: `turn-${i}`,
			userMessage: { type: "user", content: `Demo question #${i + 1} 🚀 — 日本語テスト` },
			assistantMessage: {
				type: "assistant",
				isComplete: true,
				chunks: [
					{
						seq: 0,
						type: "content",
						contentText: `This is a demo response for turn ${i + 1}. It includes some emoji 🎉✨🔥 and wide characters 中文 to exercise the Unicode-width fix, plus enough text to wrap across multiple lines when the terminal is narrow.`,
						isComplete: true,
					},
				],
			},
			isComplete: true,
		});
	}
	return turns;
}

function DemoApp(): React.ReactElement {
	const { exit } = useApp();
	const transcript = useMemo(() => {
		const t = new TranscriptDisplay({ thinkingMode: "collapsed" });
		t.setTurns(buildDemoTurns());
		return t;
	}, []);
	const inputBar = useMemo(() => new InputBar(), []);
	const statusBar = useMemo(() => {
		const s = new StatusBar();
		s.update({ phase: "ready", model: "demo", cwd: process.cwd() });
		return s;
	}, []);

	const [renderTick, setRenderTick] = useState(0);
	const bump = () => setRenderTick((n) => n + 1);

	const mouseBuffer = useRef("");

	useInput((input, key) => {
		if (key.escape || (key.ctrl && input === "c")) {
			exit();
			return;
		}
		if (key.pageUp) {
			transcript.scroll(20);
			bump();
			return;
		}
		if (key.pageDown) {
			transcript.scroll(-20);
			bump();
			return;
		}
		if (key.upArrow) {
			transcript.scroll(1);
			bump();
			return;
		}
		if (key.downArrow) {
			transcript.scroll(-1);
			bump();
			return;
		}
		if (key.return) {
			inputBar.handleInput("\r");
			bump();
			return;
		}
		if (key.backspace || key.delete) {
			inputBar.handleInput("\x7f");
			bump();
			return;
		}
		if (input) {
			inputBar.handleInput(input);
			bump();
		}
	});

	useEffect(() => {
		transcript.setOnAnimationTick(bump);
		statusBar.setOnInvalidate(bump);
		transcript.startAnimation();
		return () => {
			transcript.stopAnimation();
		};
	}, [transcript, statusBar]);

	return (
		<AppShell
			transcript={transcript}
			inputBar={inputBar}
			statusBar={statusBar}
			renderTick={renderTick}
		/>
	);
}

const instance = render(<DemoApp />, { alternateScreen: true });
await instance.waitUntilExit();
