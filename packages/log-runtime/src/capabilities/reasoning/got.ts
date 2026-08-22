// ── GoT: Graph of Thoughts ──────────────────────────────────────────────────────
// Adapted from "Graph of Thoughts: Solving Elaborate Problems with Large Language Models" (Mialon et al., 2023).
//
// GoT extends Tree of Thoughts by allowing reasoning paths to merge/diverge via a graph structure.
// Nodes represent partial solutions; edges represent reasoning steps.
// Supports:
// - Divergence: Generate multiple thoughts from a node
// - Convergence: Merge multiple thoughts into a consolidated solution
// - Pruning: Remove low-scoring nodes

import { BaseReasoner, type ReasoningTrace } from "./base.ts";

interface GoTNode {
	id: string;
	content: string;
	score: number;
	children: string[];
	parents: string[];
}

interface GoTConfig {
	beamWidth?: number;
	maxDepth?: number;
	branchFactor?: number;
	mergeThreshold?: number;
	temperature?: number;
	maxTokens?: number;
	[key: string]: unknown;
}

export class GoTReasoner extends BaseReasoner {
	config: GoTConfig;

	constructor(
		llm: import("@logician/log-core").LLMBackend,
		config: GoTConfig = {},
	) {
		super(llm, config);
		this.config = config;
	}

	async solve(
		query: string,
		initialSolution?: string | undefined | null,
	): Promise<ReasoningTrace> {
		const beamWidth = this.config.beamWidth ?? 6;
		const maxDepth = this.config.maxDepth ?? 10;
		const branchFactor = this.config.branchFactor ?? 3;
		const mergeThreshold = this.config.mergeThreshold ?? 0.7;

		// Graph state: id -> node
		const graph = new Map<string, GoTNode>();
		let nodeIdCounter = 0;

		// Create root node
		const rootId = `node_${++nodeIdCounter}`;
		const rootNode: GoTNode = {
			id: rootId,
			content: initialSolution || "",
			score: initialSolution ? await this._score(query, initialSolution) : 0.0,
			children: [],
			parents: [],
		};
		graph.set(rootId, rootNode);

		let frontier = [rootId];
		let bestNode: GoTNode | null = null;
		let bestScore = -Infinity;

		for (let depth = 0; depth < maxDepth; depth++) {
			const nextFrontier: string[] = [];
			const mergeCandidates: Map<string, string[]> = new Map();

			for (const nodeId of frontier) {
				const node = graph.get(nodeId);
				if (!node) continue;

				// Check if node is a complete solution
				if (/final answer/i.test(node.content) || node.score >= 0.9) {
					if (node.score > bestScore) {
						bestScore = node.score;
						bestNode = node;
					}
					continue;
				}

				// Divergence: generate branchFactor thoughts
				const prompt = `${query}\n\nCurrent reasoning:\n${node.content || "(empty)"}\n\nGenerate ${branchFactor} different continuations or perspectives. Each should be a distinct reasoning step. If done, end with 'Final answer: ...'.`;

				for (let b = 0; b < branchFactor; b++) {
					const cont = await this._chat([{ role: "user", content: prompt }], {
						temperature: 0.9,
						maxTokens: 512,
					});
					const full = `${node.content}\n${cont}`.trim();
					const score = await this._score(query, full);
					const childId = `node_${++nodeIdCounter}`;

					const childNode: GoTNode = {
						id: childId,
						content: full,
						score,
						children: [],
						parents: [nodeId],
					};
					graph.set(childId, childNode);
					node.children.push(childId);

					if (/final answer/i.test(full) && score > bestScore) {
						bestScore = score;
						bestNode = childNode;
					}
					nextFrontier.push(childId);

					// Track for potential merging
					const scoreBucket = Math.floor(score * 10);
					if (!mergeCandidates.has(String(scoreBucket))) {
						mergeCandidates.set(String(scoreBucket), []);
					}
					mergeCandidates.get(String(scoreBucket))?.push(childId);
				}
			}

			// Convergence: merge similar nodes
			for (const [_bucket, nodeIds] of mergeCandidates.entries()) {
				if (nodeIds.length < 2) continue;

				// Find pairs with similar content/score and merge
				for (let i = 0; i < nodeIds.length; i++) {
					for (let j = i + 1; j < nodeIds.length; j++) {
						const n1 = graph.get(nodeIds[i]);
						const n2 = graph.get(nodeIds[j]);
						if (!n1 || !n2) continue;

						if (Math.abs(n1.score - n2.score) < 0.15) {
							// Merge: create consolidated node
							const mergedContent = await this._mergeThoughts(
								query,
								n1.content,
								n2.content,
							);
							const mergedScore = await this._score(query, mergedContent);

							if (mergedScore >= mergeThreshold) {
								const mergedId = `node_${++nodeIdCounter}`;
								const mergedNode: GoTNode = {
									id: mergedId,
									content: mergedContent,
									score: mergedScore,
									children: [],
									parents: [...n1.parents, ...n2.parents],
								};
								graph.set(mergedId, mergedNode);

								// Link parents to merged node
								for (const parentId of [...n1.parents, ...n2.parents]) {
									const parent = graph.get(parentId);
									if (parent) {
										parent.children = parent.children.filter(
											id => id !== n1.id && id !== n2.id,
										);
										parent.children.push(mergedId);
									}
								}

								if (mergedScore > bestScore) {
									bestScore = mergedScore;
									bestNode = mergedNode;
								}
								nextFrontier.push(mergedId);
							}
						}
					}
				}
			}

			// Prune: keep only top beamWidth nodes
			if (nextFrontier.length > beamWidth) {
				const scored = nextFrontier
					.map(id => ({ id, score: graph.get(id)?.score || 0 }))
					.sort((a, b) => b.score - a.score);
				frontier = scored.slice(0, beamWidth).map(s => s.id);
			} else {
				frontier = nextFrontier;
			}

			if (frontier.length === 0) break;
		}

		// Fallback to best frontier node if no complete solution found
		if (!bestNode && frontier.length > 0) {
			const bestId = frontier.reduce((best, id) => {
				const node = graph.get(id);
				return node && node.score > (graph.get(best)?.score || -Infinity)
					? id
					: best;
			}, frontier[0]);
			bestNode = graph.get(bestId) || null;
		}

		const finalContent = bestNode?.content || "";
		const [reasoning, answer] = this._split(finalContent);

		return {
			reasoning,
			answer,
			metadata: {
				method: "got",
				graphSize: graph.size,
				frontierSize: frontier.length,
				bestScore,
			},
		};
	}

	private async _mergeThoughts(
		query: string,
		thought1: string,
		thought2: string,
	): Promise<string> {
		const prompt = `[Problem]\n${query}\n\n[Thought 1]\n${thought1}\n\n[Thought 2]\n${thought2}\n\n\nMerge these into a single coherent solution. Resolve conflicts and keep the best parts. End with 'Final answer: ...'.`;
		const resp = await this._chat([{ role: "user", content: prompt }], {
			temperature: 0.3,
			maxTokens: 512,
		});
		return resp;
	}

	private async _score(query: string, reasoning: string): Promise<number> {
		const prompt = `[Problem]\n${query}\n\n[Partial solution]\n${reasoning}\n\nRate promise (0-1). Output only a number.`;
		const raw = await this._chat([{ role: "user", content: prompt }], {
			temperature: 0.0,
			maxTokens: 16,
		});
		const match = raw.trim().match(/[0-1](?:\.\d+)?/);
		if (match) {
			const v = parseFloat(match[0]);
			if (!Number.isNaN(v)) return v;
		}
		const length = Math.min(reasoning.length / 1000, 1.0);
		const bonus = /final answer/i.test(reasoning) ? 0.2 : 0.0;
		return length + bonus;
	}
}
