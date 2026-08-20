BIN := dist/logician
TMP_BIN := $(BIN).tmp
ENTRY := src/index.ts
BUN ?= $(shell command -v bun 2>/dev/null || printf '%s/.bun/bin/bun' "$$HOME")
INSTALL_BIN ?= $(HOME)/.local/bin/logician

.PHONY: build binary install dev start check lint lint-fix format clean

build: binary

binary:
	@test -x "$(BUN)" || { echo "bun is required to build $(BIN)"; exit 1; }
	@cd $(CURDIR)/../.. && $(BUN) install
	@mkdir -p dist
	@rm -f $(TMP_BIN)
	$(BUN) build --target=bun --compile $(ENTRY) --outfile $(TMP_BIN)
	@mv -f $(TMP_BIN) $(BIN)
	@chmod +x $(BIN)
	@echo "Built $(abspath $(BIN))"

install: binary
	@mkdir -p $(dir $(INSTALL_BIN))
	@ln -sfn $(abspath $(BIN)) $(INSTALL_BIN)
	@test "$$(readlink -f $(INSTALL_BIN) 2>/dev/null)" = "$(abspath $(BIN))"
	@echo "Installed $(INSTALL_BIN) -> $(abspath $(BIN))"

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
