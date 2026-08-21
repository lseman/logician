/** Immutable, monotonically-versioned configuration snapshot. */
export interface ConfigurationSnapshot<T extends object> {
	revision: number;
	value: Readonly<T>;
}

export interface ConfigurationStoreOptions<T extends object> {
	clone(value: T): T;
	validate?(value: T): readonly string[];
}

/**
 * Owns runtime configuration revisions. Callers can read or replace snapshots,
 * but never mutate the store's active value in place.
 */
export class ConfigurationStore<T extends object> {
	private active: ConfigurationSnapshot<T>;

	constructor(
		initial: T,
		private readonly options: ConfigurationStoreOptions<T>,
	) {
		const value = this.options.clone(initial);
		this.assertValid(value);
		this.active = { revision: 0, value: Object.freeze(value) };
	}

	get current(): Readonly<T> {
		return this.active.value;
	}

	snapshot(): ConfigurationSnapshot<T> {
		return {
			revision: this.active.revision,
			value: Object.freeze(this.options.clone(this.active.value as T)),
		};
	}

	update(patch: Partial<T>): ConfigurationSnapshot<T> {
		const next = this.options.clone({
			...(this.active.value as T),
			...patch,
		});
		this.assertValid(next);
		this.active = {
			revision: this.active.revision + 1,
			value: Object.freeze(next),
		};
		return this.snapshot();
	}

	private assertValid(value: T): void {
		const errors = this.options.validate?.(value) ?? [];
		if (errors.length > 0) throw new Error(errors.join("; "));
	}
}
