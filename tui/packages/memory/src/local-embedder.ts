export interface MemoryEmbedder {
	readonly dimensions: number;
	readonly model: string;
	isReady(): boolean;
	warmup(): Promise<void>;
	embed(text: string): Promise<number[]>;
	embedBatch(texts: string[]): Promise<number[][]>;
}

type FeatureExtractor = (
	texts: string[],
	options: { pooling: "mean"; normalize: true },
) => Promise<{
	tolist?: () => number[][];
	data?: ArrayLike<number>;
	dims?: number[];
}>;

/** Lazy local MiniLM embeddings. Loading/downloading never occurs unless enabled. */
export class LocalMemoryEmbedder implements MemoryEmbedder {
	readonly dimensions = 384;
	private extractor: FeatureExtractor | null = null;
	private loading: Promise<FeatureExtractor> | null = null;

	constructor(readonly model = "Xenova/all-MiniLM-L6-v2") {}

	isReady(): boolean {
		return this.extractor !== null;
	}

	async warmup(): Promise<void> {
		await this.getExtractor();
	}

	async embed(text: string): Promise<number[]> {
		const [embedding] = await this.embedBatch([text]);
		return embedding || [];
	}

	async embedBatch(texts: string[]): Promise<number[][]> {
		if (!texts.length) return [];
		const extractor = await this.getExtractor();
		const output = await extractor(texts, { pooling: "mean", normalize: true });
		if (output.tolist) return output.tolist();
		if (output.data && output.dims?.length) {
			const width = output.dims.at(-1) || this.dimensions;
			return Array.from({ length: texts.length }, (_, index) =>
				Array.from(output.data!).slice(index * width, (index + 1) * width),
			);
		}
		throw new Error(
			"Local embedding model returned an unsupported tensor shape",
		);
	}

	private getExtractor(): Promise<FeatureExtractor> {
		if (this.extractor) return Promise.resolve(this.extractor);
		if (this.loading) return this.loading;
		this.loading = import("@huggingface/transformers")
			.then(
				async ({ pipeline }) =>
					pipeline("feature-extraction", this.model, {
						dtype: "q8",
					}) as Promise<FeatureExtractor>,
			)
			.then(extractor => {
				this.extractor = extractor;
				return extractor;
			})
			.finally(() => {
				this.loading = null;
			});
		return this.loading;
	}
}
