# Root Makefile — delegates to the tui/ project (the actual app lives there).
# Every target runs inside tui/ so `make build` works from the repo root.

TUI := tui

.PHONY: build binary install dev start check lint lint-fix format clean

build binary install dev start check lint lint-fix format clean:
	$(MAKE) -C $(TUI) $@
