BIN := dist/logician
ENTRY := src/index.ts
BUN ?= $(shell command -v bun 2>/dev/null || printf '%s/.bun/bin/bun' "$$HOME")

.PHONY: build binary install dev start check lint lint-fix format clean

build: binary

binary:
	@test -x "$(BUN)" || { echo "bun is required to build $(BIN)"; exit 1; }
	@cd $(CURDIR)/../.. && $(BUN) install
	@mkdir -p dist
	$(BUN) build --target=bun --compile $(ENTRY) --outfile $(BIN)
	@chmod +x $(BIN)
	@echo "Built $(BIN)"

install:
	$(BUN) install

dev:
	$(BUN) run dev

start:
	$(BUN) run start

check:
	$(BUN) run typecheck

lint:
	$(BUN) run lint

lint-fix:
	$(BUN) run lint:fix

format:
	@cd $(CURDIR)/../.. && $(BUN) run format

clean:
	rm -rf dist
