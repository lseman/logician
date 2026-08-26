import { describe, expect, test } from "bun:test";
import { LegroomGateway } from "../../capabilities/legroom/legroom-gateway.ts";

describe("LegroomGateway", () => {
	test("fails guarded operations consistently while disabled", async () => {
		const gateway = new LegroomGateway({ mode: "off" });
		await expect(gateway.storeRetrieve("store", "hash")).rejects.toThrow(
			"Legroom SDK is not enabled",
		);
		await expect(gateway.workerStats()).rejects.toThrow(
			"Legroom SDK is not enabled",
		);
	});

	test("preserves provider payloads without starting a worker while disabled", async () => {
		const gateway = new LegroomGateway({ mode: "off" });
		const hooks = gateway.createHooks(undefined);
		const payload = { messages: [{ role: "user", content: "hello" }] };
		const result = await hooks?.beforeProviderPayload?.({
			payload,
			model: "test",
		});
		expect(result?.payload).toBe(payload);
	});

	test("disabling closes the worker and updates the owned state", () => {
		const gateway = new LegroomGateway({ mode: "sdk" });
		expect(gateway.isEnabled()).toBe(true);
		gateway.setEnabled(false);
		expect(gateway.isEnabled()).toBe(false);
	});
});
