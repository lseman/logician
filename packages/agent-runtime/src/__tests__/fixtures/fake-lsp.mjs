let buffer = Buffer.alloc(0);

function send(message) {
	const body = JSON.stringify(message);
	process.stdout.write(
		`Content-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`,
	);
}

process.stdin.on("data", chunk => {
	buffer = Buffer.concat([buffer, chunk]);
	while (true) {
		const headerEnd = buffer.indexOf("\r\n\r\n");
		if (headerEnd < 0) return;
		const header = buffer.subarray(0, headerEnd).toString("ascii");
		const match = /Content-Length:\s*(\d+)/i.exec(header);
		if (!match) return;
		const length = Number(match[1]);
		const start = headerEnd + 4;
		if (buffer.length < start + length) return;
		const message = JSON.parse(
			buffer.subarray(start, start + length).toString("utf8"),
		);
		buffer = buffer.subarray(start + length);
		if (message.method === "initialize") {
			send({ jsonrpc: "2.0", id: message.id, result: { capabilities: {} } });
		}
		if (
			message.method === "textDocument/didOpen" ||
			message.method === "textDocument/didChange"
		) {
			const document = message.params.textDocument;
			send({
				jsonrpc: "2.0",
				method: "textDocument/publishDiagnostics",
				params: {
					uri: document.uri,
					diagnostics: [
						{
							range: {
								start: { line: 1, character: 2 },
								end: { line: 1, character: 3 },
							},
							severity: 1,
							code: "fake-1",
							source: "fake-lsp",
							message: "synthetic diagnostic",
						},
					],
				},
			});
		}
	}
});
