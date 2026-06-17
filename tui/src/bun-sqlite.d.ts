declare module "bun:sqlite" {
	export class Database {
		constructor(
			path: string,
			options?: { readonly?: boolean; verbose?: boolean },
		);
		exec(sql: string): void;
		prepare<T extends Record<string, unknown>[]>(sql: string): Statement<T>;
		pragma(name: string): unknown;
		close(): void;
		sync(): void;
	}

	export interface Statement<T = Record<string, unknown>[]> {
		run(...params: unknown[]): {
			changes: number;
			lastInsertRowId: bigint | number;
		};
		get(...params: unknown[]): T extends Array<infer R> ? R : unknown;
		all(...params: unknown[]): T;
		iterate(
			...params: unknown[]
		): IterableIterator<T extends Array<infer R> ? R : unknown>;
		bind(...params: unknown[]): this;
		raw<T extends unknown[]>(...params: unknown[]): T;
		expand(): void;
		setMaxHistorySize(size: number): this;
		historySize: number;
	}
}
