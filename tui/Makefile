BIN := dist/logician
ENTRY := src/index.ts
BUN ?= $(shell command -v bun 2>/dev/null || printf '%s/.bun/bin/bun' "$$HOME")

.PHONY: build binary clean check format

build: binary

binary:
	@test -x "$(BUN)" || { echo "bun is required to build $(BIN)"; exit 1; }
	@mkdir -p dist
	$(BUN) build --compile $(ENTRY) --outfile $(BIN)
	@cp AGENTS.md dist/AGENTS.md
	@cp .mcp.json dist/.mcp.json
	@chmod +x $(BIN)
	@echo "Built $(BIN)"

check:
	npx tsc --noEmit

format:
	npx prettier --write "src/**/*.ts" "*.json" "*.md" --tab-width 4

clean:
	rm -rf dist
