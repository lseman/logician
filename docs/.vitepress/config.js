import { defineConfig } from "vitepress";
import { withMermaid } from "vitepress-plugin-mermaid";

export default withMermaid(
	defineConfig({
		title: "Logician",
		description: "Local-first coding agent with streaming terminal UI",
		base: "/docs/",
		appearance: "force-dark",
		cleanUrls: true,
		outDir: "../site/docs",
		head: [["link", { rel: "icon", href: "/logo.svg" }]],
		themeConfig: {
			logo: "/logo.svg",
			nav: [
				{
					text: "Logician v0.2.0",
					items: [
						{ text: "Getting Started", link: "/getting-started" },
						{ text: "Guides", link: "/guides/overview" },
						{ text: "API", link: "/reference/api" },
						{ text: "GitHub", link: "https://github.com/lseman/logician" },
					],
				},
			],
			sidebar: [
				{
					text: "Start",
					items: [
						{ text: "Home", link: "/" },
						{ text: "Overview", link: "/overview" },
						{ text: "Getting Started", link: "/getting-started" },
					],
				},
				{
					text: "Guides",
					items: [
						{ text: "Overview", link: "/guides/overview" },
						{ text: "Terminal UI", link: "/guides/terminal-ui" },
						{ text: "Streaming Mode", link: "/guides/streaming" },
						{ text: "Skills", link: "/guides/skills" },
						{ text: "Plugins & Hooks", link: "/guides/plugins" },
						{ text: "Session Management", link: "/guides/sessions" },
						{ text: "MCP Servers", link: "/guides/mcp" },
						{ text: "Reasoning & Inference", link: "/guides/reasoning" },
						{ text: "Subagents", link: "/guides/subagents" },
						{ text: "Agent Evaluation", link: "/guides/agent-evaluation" },
						{ text: "Trust & Safety", link: "/guides/trust" },
						{ text: "Configuration", link: "/guides/configuration" },
						{ text: "Troubleshooting", link: "/guides/troubleshooting" },
					],
				},
				{
					text: "Tutorials",
					items: [
						{ text: "First Agent Session", link: "/tutorials/first-session" },
						{ text: "Custom Skills", link: "/tutorials/custom-skills" },
						{ text: "Headless Mode", link: "/tutorials/headless" },
					],
				},
				{
					text: "Architecture",
					items: [
						{ text: "System Overview", link: "/architecture/overview" },
						{ text: "Agent Loop", link: "/architecture/agent-loop" },
						{ text: "Hook System", link: "/architecture/hooks" },
						{ text: "Session Persistence", link: "/architecture/sessions" },
						{ text: "Durability & Recovery", link: "/architecture/run-kernel" },
						{ text: "Evolving Memory", link: "/architecture/evolving-memory" },
						{ text: "Runtime Decisions", link: "/architecture/modernization" },
					],
				},
				{
					text: "Reference",
					items: [
						{ text: "API", link: "/reference/api" },
						{ text: "Config Schema", link: "/reference/config" },
						{ text: "Skills API", link: "/reference/skills" },
						{ text: "Hook API", link: "/reference/hooks" },
					],
				},
			],
			socialLinks: [
				{ icon: "github", link: "https://github.com/lseman/logician" },
			],
			editLink: {
				pattern: "https://github.com/lseman/logician/edit/main/docs/:path",
				text: "Edit this page on GitHub",
			},
			footer: {
				message: "MIT License",
				copyright: "Copyright © 2026 Logician",
			},
			search: {
				provider: "local",
			},
		},
		mermaid: {
			theme: "dark",
			securityLevel: "strict",
		},
	}),
);
