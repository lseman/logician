import type {
	Observation,
	PersistedKnowledgeGraph,
	Reflection,
} from "../types.ts";

export type KnowledgeNode = PersistedKnowledgeGraph["nodes"][number];
export type KnowledgeEdge = PersistedKnowledgeGraph["edges"][number];
export type KnowledgeGraph = PersistedKnowledgeGraph;

/**
 * Read-only graph projection over validated memory records.
 * Observations and reflections remain the source of truth.
 */
export class KnowledgeGraphManager {
	private readonly nodes = new Map<string, KnowledgeNode>();
	private readonly outgoing = new Map<string, KnowledgeEdge[]>();

	constructor(graph?: PersistedKnowledgeGraph) {
		if (graph) this.importGraph(graph);
	}

	static fromMemory(
		observations: readonly Observation[],
		reflections: readonly Reflection[],
	): KnowledgeGraphManager {
		const observationIds = new Set(observations.map((item) => item.id));
		const nodes: KnowledgeNode[] = [
			...observations.map((item) => ({
				id: item.id,
				type: "observation" as const,
				content: item.content,
				metadata: {
					timestamp: item.timestamp,
					relevance: item.relevance,
					sourceEntryIds: [...item.sourceEntryIds],
				},
				tokens: item.tokenCount,
			})),
			...reflections.map((item) => ({
				id: item.id,
				type: "reflection" as const,
				content: item.content,
				metadata: {
					supportingObservationIds: [...item.supportingObservationIds],
				},
				tokens: item.tokenCount,
			})),
		];
		const edges: KnowledgeEdge[] = reflections.flatMap((reflection) =>
			reflection.supportingObservationIds
				.filter((id) => observationIds.has(id))
				.map((id) => ({
					source: reflection.id,
					target: id,
					relationship: "supported_by" as const,
					weight: 1,
					metadata: {},
				})),
		);
		return new KnowledgeGraphManager({ nodes, edges });
	}

	getNode(id: string): KnowledgeNode | undefined {
		return this.nodes.get(id);
	}

	getRelatedNodes(nodeId: string): KnowledgeNode[] {
		return (this.outgoing.get(nodeId) ?? [])
			.map((edge) => this.nodes.get(edge.target))
			.filter((node): node is KnowledgeNode => node !== undefined);
	}

	exportGraph(): PersistedKnowledgeGraph {
		return {
			nodes: Array.from(this.nodes.values(), (node) => ({
				...node,
				metadata: { ...node.metadata },
			})),
			edges: Array.from(this.outgoing.values())
				.flat()
				.map((edge) => ({ ...edge, metadata: { ...edge.metadata } })),
		};
	}

	private importGraph(graph: PersistedKnowledgeGraph): void {
		for (const node of graph.nodes) this.nodes.set(node.id, node);
		for (const edge of graph.edges) {
			if (!this.nodes.has(edge.source) || !this.nodes.has(edge.target))
				continue;
			const edges = this.outgoing.get(edge.source) ?? [];
			edges.push(edge);
			this.outgoing.set(edge.source, edges);
		}
	}
}
