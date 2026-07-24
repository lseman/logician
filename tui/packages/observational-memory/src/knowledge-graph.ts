// ── Knowledge Graph for RAG ────────────────────────────────────────────────
// Provides graph-based relationships between documents, concepts, and code entities.
// Enhances RAG with semantic relationships and contextual understanding.
//
// NOT WIRED UP: nothing in this package constructs a KnowledgeGraphManager —
// it's only re-exported from index.ts. No caller feeds it observations/
// reflections, and its graph is never persisted via FilePersistence. Either
// finish integrating it (populate from observer/reflector output, persist
// alongside FoldedMemory, expose via recall) or remove it.

import type { Observation, Reflection } from "./types.ts";

export interface KnowledgeNode {
	id: string;
	type: 'document' | 'concept' | 'code_entity' | 'file' | 'function' | 'class' | 'variable';
	content: string;
	metadata: Record<string, any>;
	tokens: number;
}

export interface KnowledgeEdge {
	source: string;
	target: string;
	relationship: string;
	weight: number;
	metadata: Record<string, any>;
}

export interface KnowledgeGraph {
	nodes: Map<string, KnowledgeNode>;
	edges: Map<string, Map<string, KnowledgeEdge>>;
}

export class KnowledgeGraphManager {
	private graph: KnowledgeGraph;
	private nodeIndex: Map<string, string[]>; // concept -> [node_ids]

	constructor() {
		this.graph = {
			nodes: new Map(),
			edges: new Map(),
		};
		this.nodeIndex = new Map();
	}

	addNode(node: KnowledgeNode): void {
		this.graph.nodes.set(node.id, node);
		
		// Index by content concepts
		const concepts = this.extractConcepts(node.content);
		for (const concept of concepts) {
			if (!this.nodeIndex.has(concept)) {
				this.nodeIndex.set(concept, []);
			}
			this.nodeIndex.get(concept)!.push(node.id);
		}
	}

	addEdge(edge: KnowledgeEdge): void {
		if (!this.graph.edges.has(edge.source)) {
			this.graph.edges.set(edge.source, new Map());
		}
		this.graph.edges.get(edge.source)!.set(edge.target, edge);
	}

	getNode(id: string): KnowledgeNode | undefined {
		return this.graph.nodes.get(id);
	}

	getEdges(sourceId: string): Map<string, KnowledgeEdge> {
		return this.graph.edges.get(sourceId) || new Map();
	}

	getRelatedNodes(nodeId: string, maxDepth: number = 2): KnowledgeNode[] {
		const related = new Set<string>();
		const visited = new Set<string>();
		const queue: [{ id: string, depth: number }] = [{ id: nodeId, depth: 0 }];
		
		while (queue.length > 0) {
			const current = queue.shift()!;
			
			if (current.depth > maxDepth) continue;
			if (visited.has(current.id)) continue;
			visited.add(current.id);
			
			if (current.id !== nodeId) {
				related.add(current.id);
			}
			
			if (current.depth < maxDepth) {
				const edges = this.getEdges(current.id);
				for (const [targetId, edge] of edges.entries()) {
					if (!visited.has(targetId)) {
						queue.push({ id: targetId, depth: current.depth + 1 });
					}
				}
			}
		}
		
		return Array.from(related).map(id => this.graph.nodes.get(id)!).filter(Boolean) as KnowledgeNode[];
	}

	searchByConcept(concept: string): KnowledgeNode[] {
		const nodeIds = this.nodeIndex.get(concept) || [];
		return nodeIds.map(id => this.graph.nodes.get(id)).filter(Boolean) as KnowledgeNode[];
	}

	// Extract concepts from content (simplified)
	private extractConcepts(content: string): string[] {
		// Extract potential concepts: words, phrases, code entities
		const concepts: string[] = [];
		
		// Extract code entities (functions, classes, variables)
		const codeEntities = content.match(/\b(?:function|class|interface|type|const|let|var|def|class|fn)\s+[a-zA-Z0-9_]+\b/g);
		if (codeEntities) {
			concepts.push(...codeEntities.map(e => e.toLowerCase()));
		}
		
		// Extract technical terms (capitalized words, acronyms)
		const technicalTerms = content.match(/\b[A-Z]{2,}[a-z]*\b|[a-z]+\.[a-zA-Z0-9_]+\b/g);
		if (technicalTerms) {
			concepts.push(...technicalTerms);
		}
		
		// Extract common technical words
		const commonTerms = content.toLowerCase().match(/\b(?:api|database|service|component|module|package|library|framework|algorithm|data|structure|interface|protocol)\b/g);
		if (commonTerms) {
			concepts.push(...commonTerms);
		}
		
		return [...new Set(concepts)].filter(c => c.length > 2);
	}

	// Generate graph insights for RAG
	getGraphInsights(query: string): string {
		const concepts = this.extractConcepts(query);
		const relatedNodes: { concept: string, nodes: KnowledgeNode[] }[] = [];
		
		for (const concept of concepts) {
			const nodes = this.searchByConcept(concept);
			if (nodes.length > 0) {
				relatedNodes.push({ concept, nodes });
			}
		}
		
		if (relatedNodes.length === 0) {
			return "No specific graph relationships found for this query.";
		}
		
		let insights = "Graph relationships relevant to your query:\n\n";
		for (const { concept, nodes } of relatedNodes) {
			insights += `Concept: ${concept}\n`;
			for (const node of nodes.slice(0, 3)) { // Limit to 3 nodes per concept
				insights += `  - ${node.type}: ${node.content.substring(0, 100)}${node.content.length > 100 ? '...' : ''}\n`;
				
				// Add related nodes
				const related = this.getRelatedNodes(node.id, 1);
				if (related.length > 0) {
					insights += `    Related: ${related.map(r => `${r.type}(${r.id.substring(0, 8)}).`).join(', ')}\n`;
				}
			}
			insights += '\n';
		}
		
		return insights;
	}

	// Export graph for persistence
	exportGraph(): { nodes: KnowledgeNode[], edges: KnowledgeEdge[] } {
		const edges: KnowledgeEdge[] = [];
		for (const [sourceId, targetMap] of this.graph.edges.entries()) {
			for (const [targetId, edge] of targetMap.entries()) {
				edges.push(edge);
			}
		}
		
		return {
			nodes: Array.from(this.graph.nodes.values()),
			edges
		};
	}

	// Import graph from persistence
	importGraph(data: { nodes: KnowledgeNode[], edges: KnowledgeEdge[] }): void {
		for (const node of data.nodes) {
			this.addNode(node);
		}
		for (const edge of data.edges) {
			this.addEdge(edge);
		}
	}
}