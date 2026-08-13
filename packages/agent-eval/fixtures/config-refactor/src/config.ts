export interface Config {
	host: string;
	port: number;
	secure: boolean;
}

export function readConfig(env: Record<string, string | undefined>): Config {
	const host = env.APP_HOST ? env.APP_HOST.trim() : "127.0.0.1";
	const rawPort = env.APP_PORT ? env.APP_PORT.trim() : "8080";
	const port = Number(rawPort);
	if (!Number.isInteger(port) || port < 1 || port > 65535) {
		throw new Error("APP_PORT must be an integer from 1 to 65535");
	}
	const secure = env.APP_SECURE
		? env.APP_SECURE.trim().toLowerCase() === "true"
		: false;
	return { host, port, secure };
}

export function readWorkerConfig(
	env: Record<string, string | undefined>,
): Config {
	const host = env.APP_HOST ? env.APP_HOST.trim() : "127.0.0.1";
	const rawPort = env.APP_PORT ? env.APP_PORT.trim() : "8080";
	const port = Number(rawPort);
	if (!Number.isInteger(port) || port < 1 || port > 65535) {
		throw new Error("APP_PORT must be an integer from 1 to 65535");
	}
	const rawSecure = env.APP_SECURE
		? env.APP_SECURE.trim().toLowerCase()
		: "false";
	if (rawSecure !== "true" && rawSecure !== "false") {
		throw new Error("APP_SECURE must be true or false");
	}
	return { host, port, secure: rawSecure === "true" };
}
