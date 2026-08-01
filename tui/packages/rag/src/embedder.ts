// ── Embedding Provider ───────────────────────────────────────────────────────
// Abstract interface for converting text to vectors. Pluggable backends.

export interface IEmbedder {
  dimension: number;
  embed(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
}

/** Cosine similarity between two vectors. */
export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// biome-ignore lint: kept for pre-existing callers
export function cosine(a: number[], b: number[]): number {
  return cosineSimilarity(a, b);
}

type FeatureExtractionPipeline = (
  texts: string | string[],
  options: { pooling: "mean"; normalize: boolean },
) => Promise<{ data: Float32Array | number[]; dims: number[] }>;

/**
 * Local ONNX sentence-embedding model (all-MiniLM-L6-v2, 384-dim), run
 * in-process via @huggingface/transformers. No network calls after the
 * model is cached on first use.
 */
export class TransformersEmbedder implements IEmbedder {
  readonly dimension = 384;
  private readonly modelId: string;
  private extractor: FeatureExtractionPipeline | null = null;
  private loading: Promise<FeatureExtractionPipeline> | null = null;

  constructor(modelId = "Xenova/all-MiniLM-L6-v2") {
    this.modelId = modelId;
  }

  private async getExtractor(): Promise<FeatureExtractionPipeline> {
    if (this.extractor) return this.extractor;
    if (!this.loading) {
      this.loading = import("@huggingface/transformers").then(async ({ pipeline }) => {
        const extractor = (await pipeline(
          "feature-extraction",
          this.modelId,
        )) as unknown as FeatureExtractionPipeline;
        this.extractor = extractor;
        return extractor;
      });
    }
    return this.loading;
  }

  async embed(text: string): Promise<number[]> {
    const [vec] = await this.embedBatch([text]);
    return vec;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    const extractor = await this.getExtractor();
    const output = await extractor(texts, { pooling: "mean", normalize: true });
    const dims = output.dims;
    const width = dims[dims.length - 1];
    const data = output.data instanceof Float32Array ? output.data : new Float32Array(output.data);
    const vectors: number[][] = [];
    for (let i = 0; i < texts.length; i++) {
      vectors.push(Array.from(data.subarray(i * width, (i + 1) * width)));
    }
    return vectors;
  }
}

/**
 * Deterministic pseudo-random embedder for tests/offline fixtures where a
 * real model isn't wanted. Vectors carry no semantic meaning.
 */
export class SeededEmbedder implements IEmbedder {
  readonly dimension: number;

  constructor(dimension = 384) {
    this.dimension = dimension;
  }

  /** Deterministic hash of string → float array [-1, 1]. */
  private hash(text: string): number[] {
    const bytes = new TextEncoder().encode(text);
    let h = 0x811c9dc5; // FNV-1a offset basis
    for (const b of bytes) {
      h ^= b;
      h = Math.imul(h, 0x01000193);
    }
    const result: number[] = new Array(this.dimension);
    for (let i = 0; i < this.dimension; i++) {
      let seed = h ^ i;
      result[i] = ((seed >>> 0) % 1000) / 500 - 1; // [-1, 1]
      seed = Math.imul(seed, 1103515245 + 12345);
    }
    return result;
  }

  async embed(text: string): Promise<number[]> {
    return this.hash(text);
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    return texts.map(t => this.hash(t));
  }
}
