# Root Makefile — delegates executable targets to the terminal application.
# Reusable modules remain independently addressable through the root workspace.

TUI := apps/tui

.PHONY: build binary install dev start check lint lint-fix format clean

build binary install dev start check lint lint-fix format clean:
	$(MAKE) -C $(TUI) $@
