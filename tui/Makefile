BIN := dist/logician
ENTRY := src/index.ts
BUN ?= $(shell command -v bun 2>/dev/null || printf '%s/.bun/bin/bun' "$$HOME")

.PHONY: build binary install dev start check lint lint-fix format clean

build: binary

binary:
	@test -x "$(BUN)" || { echo "bun is required to build $(BIN)"; exit 1; }
	@mkdir -p dist
	$(BUN) build --target=bun --compile $(ENTRY) --outfile $(BIN)
	@cp -r themes dist/
	@chmod +x $(BIN)
	@echo "Built $(BIN)"

install:
	npm install
	@mkdir -p "$${HOME}/.logician/themes"
	@cp -r themes/* "$$HOME/.logician/themes/" 2>/dev/null || true
	@echo "Installed themes to $$HOME/.logician/themes/"

dev:
	npm run dev

start:
	npm start

check:
	npm run typecheck

lint:
	npm run lint

lint-fix:
	npm run lint:fix

format:
	npx @biomejs/biome format --write src "*.json"

clean:
	rm -rf dist
