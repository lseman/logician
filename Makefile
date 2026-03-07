.PHONY: binary dev typecheck install clean check-soul

BINARY := logician
RUST_CLI_DIR := rust-cli

# Build Rust CLI in release mode and copy binary to repo root
binary:
	cd $(RUST_CLI_DIR) && cargo build --release
	cp $(RUST_CLI_DIR)/target/release/$(BINARY) ./$(BINARY)
	@echo "Binary ready: ./$(BINARY)"

# Dev mode (cargo run)
dev:
	cd $(RUST_CLI_DIR) && cargo run

# Type-check only
typecheck:
	cd $(RUST_CLI_DIR) && cargo check

# Pre-fetch Rust dependencies
install:
	cd $(RUST_CLI_DIR) && cargo fetch

# Remove built artefacts
clean:
	rm -f $(BINARY)
	cd $(RUST_CLI_DIR) && cargo clean

# Validate SOUL tool references against registered skills
check-soul:
	python scripts/check_soul_tools.py
