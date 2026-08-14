/**
 * EoH skills — Evolution of Heuristics (EoH) skill catalog.
 *
 * Sub-skills:
 * - eoh: Main skill — overview of the EoH framework
 * - eoh-evolve: Running evolution loops (init, run, monitor)
 * - eoh-problem: Defining and configuring evolution problems
 * - eoh-dashboard: Monitoring, visualization, and result analysis
 */

export const eohSkills = [
	{
		name: "eoh",
		path: "./SKILL.md",
		description:
			"Evolution of Heuristics (EoH) — arxiv 2401.02051. Autonomous heuristic evolution via LLM-driven operators.",
	},
	{
		name: "eoh-evolve",
		path: "./evolve/SKILL.md",
		description:
			"Run EoH evolution loops — initialize, run generations, and monitor heuristic evolution.",
	},
	{
		name: "eoh-problem",
		path: "./problem/SKILL.md",
		description:
			"Define and configure EoH evolution problems — function signatures, evaluation functions, and problem definitions.",
	},
	{
		name: "eoh-dashboard",
		path: "./dashboard/SKILL.md",
		description:
			"Monitor and visualize EoH evolution progress — status, dashboard export, and result analysis.",
	},
] as const;

export type EohSkillName = (typeof eohSkills)[number]["name"];
