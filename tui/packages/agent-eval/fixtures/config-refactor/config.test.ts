import { expect, test } from "bun:test";
import { readConfig, readWorkerConfig } from "./src/config.ts";

for (const read of [readConfig, readWorkerConfig]) {
	test(`${read.name} supplies defaults`, () => {
		expect(read({})).toEqual({ host: "127.0.0.1", port: 8080, secure: false });
	});
	test(`${read.name} trims and parses values`, () => {
		expect(
			read({
				APP_HOST: " example.test ",
				APP_PORT: "443",
				APP_SECURE: " TRUE ",
			}),
		).toEqual({ host: "example.test", port: 443, secure: true });
	});
	test(`${read.name} rejects invalid booleans`, () => {
		expect(() => read({ APP_SECURE: "yes" })).toThrow(
			"APP_SECURE must be true or false",
		);
	});
}
