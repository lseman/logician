#!/bin/sh
# Logician installer — binary TUI for the Logician coding agent
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/lseman/logician/main/tui/install.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/lseman/logician/main/tui/install.sh | bash -s -- 0.2.0
#   curl -fsSL https://raw.githubusercontent.com/lseman/logician/main/tui/install.sh | bash -s -- --dry-run
#   curl -fsSL https://raw.githubusercontent.com/lseman/logician/main/tui/install.sh | bash -s -- -v 0.2.0

# ── Colors & symbols ─────────────────────────────────────────────────────────
if [ -t 2 ] && [ -n "${TERM:-}" ]; then
  C_RESET="\033[0m"
  C_GREEN="\033[0;32m"
  C_YELLOW="\033[0;33m"
  C_RED="\033[0;31m"
  C_CYAN="\033[0;36m"
  C_BOLD="\033[1m"
  C_DIM="\033[2m"
else
  C_RESET=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_CYAN=""; C_BOLD=""; C_DIM=""
fi

OK="${C_GREEN}✓${C_RESET}"
WARN="${C_YELLOW}⚠${C_RESET}"
ERR="${C_RED}✗${C_RESET}"

# ── Options ──────────────────────────────────────────────────────────────────
DRY_RUN=0
VERBOSE=0
VERSION=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)        DRY_RUN=1; shift ;;
    -v|--verbose)     VERBOSE=1; shift ;;
    *)                VERSION="$1"; shift ;;
  esac
done

if [ -z "$VERSION" ]; then
  VERSION="latest"
fi

set -eu

INSTALL_BIN_DIR="${LOGICIAN_INSTALL_BIN_DIR:-$HOME/.local/bin}"
INSTALL_APP_DIR="${LOGICIAN_INSTALL_APP_DIR:-$HOME/.local/share/logician}"
SKIP_PATH_UPDATE="${LOGICIAN_INSTALL_SKIP_PATH_UPDATE:-0}"
path_action="already"
path_profile=""

# ── Helpers ──────────────────────────────────────────────────────────────────
step() { printf '%s\n' "${C_BOLD}${C_CYAN}▸${C_RESET} $1"; }
info() { printf '%s\n' "  $1"; }
ok()   { printf '%s\n' "  ${OK} $1"; }
warn() { printf '%s\n' "  ${WARN} $1" >&2; }
fail() { printf '%s\n' "  ${ERR} $1" >&2; exit 1; }
verbose() {
  if [ "$VERBOSE" = "1" ]; then printf '%s\n' "  [verbose] $1"; fi
}

# ── Banner ───────────────────────────────────────────────────────────────────
printf '%s\n' "${C_CYAN}${C_BOLD}  ██╗      ██████╗  ██████╗ ██╗ ██████╗ ██╗ █████╗ ███╗   ██╗"
printf '%s\n' "  ██║     ██╔═══██╗██╔════╝ ██║██╔════╝ ██║██╔══██╗████╗  ██║"
printf '%s\n' "  ██║     ██║   ██║██║  ███╗██║██║      ██║███████║██╔██╗ ██║"
printf '%s\n' "  ██║     ██║   ██║██║   ██║██║██║      ██║██╔══██║██║╚██╗██║"
printf '%s\n' "  ███████╗╚██████╔╝╚██████╔╝██║╚██████╗ ██║██║  ██║██║ ╚████║"
printf '%s\n' "  ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝ ╚═════╝ ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝${C_RESET}"
printf '\n\n'
printf '%s\n\n' "  Install script"

# ── Download helpers ─────────────────────────────────────────────────────────
HTTP_OK=0

download_file() {
  url="$1"
  output="$2"
  HTTP_OK=0

  if command -v curl >/dev/null 2>&1; then
    verbose "Using curl to fetch $url"
    if [ -t 2 ]; then
      curl -fL --progress-bar "$url" -o "$output" && HTTP_OK=1
    else
      curl -fsSL "$url" -o "$output" && HTTP_OK=1
    fi
  elif command -v wget >/dev/null 2>&1; then
    verbose "Using wget to fetch $url"
    if [ -t 2 ]; then
      wget --show-progress -O "$output" "$url" && HTTP_OK=1
    else
      wget -q -O "$output" "$url" && HTTP_OK=1
    fi
  else
    fail "curl or wget is required but not found. Install one and retry."
  fi
}

download_text() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$1"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O - "$1"
  else
    fail "curl or wget is required but not found. Install one and retry."
  fi
}

# ── Platform detection ───────────────────────────────────────────────────────
step "Detecting platform"

os=""
case "$(uname -s)" in
  Darwin) os="darwin" ;;
  Linux)  os="linux" ;;
  *) fail "Unsupported OS: $(uname -s). Logician supports macOS and Linux." ;;
esac
ok "OS: ${os}"

arch=""
case "$(uname -m)" in
  x86_64|amd64) arch="x86_64" ;;
  arm64|aarch64) arch="arm64" ;;
  *) fail "Unsupported architecture: $(uname -m). Logician supports x86_64 and arm64." ;;
esac
ok "Architecture: ${arch}"

# ── Version resolution ───────────────────────────────────────────────────────
step "Resolving version"

resolved_version=""
if printf '%s' "$VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+'; then
  resolved_version="$VERSION"
  info "Using pinned version: ${C_BOLD}${resolved_version}${C_RESET}"
elif printf '%s' "$VERSION" | grep -q '^v'; then
  resolved_version="${VERSION#v}"
  info "Using pinned version: ${C_BOLD}${resolved_version}${C_RESET}"
else
  resolved_version="latest"
fi

if [ "$resolved_version" = "latest" ]; then
  verbose "Fetching latest release from GitHub"
  release_json="$(download_text "https://api.github.com/repos/lseman/logician/releases/latest")" || true
  resolved_version="$(printf '%s\n' "$release_json" | sed -n 's/.*"tag_name":"v\{0,1\}\([^",]*\)".*/\1/p' | head -n 1)"
  if [ -z "$resolved_version" ]; then
    warn "Could not fetch latest version (rate limited or offline). Falling back to v0.1.0."
    resolved_version="0.1.0"
  fi
  ok "Latest release: ${C_BOLD}${resolved_version}${C_RESET}"
fi

# ── Self-update check ────────────────────────────────────────────────────────
installed_version=""
if [ -x "$INSTALL_BIN_DIR/logician" ]; then
  installed_version="$(
    "$INSTALL_BIN_DIR/logician" --version 2>/dev/null \
    | sed -n 's/.*v\{0,1\}\([0-9]\{1,\}\.[0-9]\{1,\}\.[0-9]\{1,\}\).*/\1/p' \
    | head -1
  )" || true
  if [ -n "$installed_version" ]; then
    if [ "$installed_version" = "$resolved_version" ]; then
      warn "Logician ${installed_version} is already installed. Reinstalling."
    fi
  fi
fi

# ── Resolve URL ──────────────────────────────────────────────────────────────
asset_name="logician-${os}-${arch}.tar.gz"
base_url="https://github.com/lseman/logician/releases/download/v${resolved_version}"
download_url="${base_url}/${asset_name}"

step "Downloading Logician ${C_BOLD}${resolved_version}${C_RESET}"
info "URL: ${download_url}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT INT TERM

if ! download_file "$download_url" "$tmp_dir/${asset_name}.tar.gz"; then
  fail "Download failed. Check your connection or pin a version."
fi
ok "Downloaded ${C_BOLD}${asset_name}${C_RESET}"

# ── Verify checksum ──────────────────────────────────────────────────────────
step "Verifying checksum"

checksum_url="${base_url}/${asset_name}.sha256"
checksum_file=""
if download_file "$checksum_url" "$tmp_dir/${asset_name}.sha256" 2>/dev/null; then
  checksum_file="$tmp_dir/${asset_name}.sha256"
fi

if [ -n "$checksum_file" ] && [ -f "$checksum_file" ]; then
  expected_hash="$(awk '{print $1}' "$checksum_file" 2>/dev/null)"
  actual_hash="$(sha256sum "$tmp_dir/${asset_name}.tar.gz" | awk '{print $1}')"
  if [ "$expected_hash" = "$actual_hash" ]; then
    ok "Checksum verified (${C_BOLD}${actual_hash:0:12}...${C_RESET})"
  else
    fail "Checksum mismatch! Expected ${C_BOLD}${expected_hash}${C_RESET}, got ${C_BOLD}${actual_hash}${C_RESET}."
  fi
else
  warn "No checksum file found at ${checksum_url} — skipping verification."
fi

# ── Dry run exit ─────────────────────────────────────────────────────────────
if [ "$DRY_RUN" = "1" ]; then
  step "Dry run — would install:"
  info "  Binary: ${INSTALL_BIN_DIR}/logician"
  info "  App dir: ${INSTALL_APP_DIR}/logician-${resolved_version}"
  info "  Version: ${resolved_version} (${os}-${arch})"
  printf '%s\n' "${C_BOLD}${C_GREEN}All checks passed — nothing installed.${C_RESET}"
  exit 0
fi

# ── Extract & install ────────────────────────────────────────────────────────
step "Installing"

mkdir -p "$INSTALL_BIN_DIR" "$INSTALL_APP_DIR"
rm -rf "$INSTALL_APP_DIR/logician-${resolved_version}"

verbose "Extracting archive to ${INSTALL_APP_DIR}"
if ! tar -xzf "$tmp_dir/${asset_name}.tar.gz" -C "$INSTALL_APP_DIR"; then
  fail "Failed to extract archive."
fi
ok "Extracted archive"

# Bun --compile produces a single binary; find it
binary=""
for f in "$INSTALL_APP_DIR/logician-${resolved_version}"/*; do
  [ -f "$f" ] && binary="$f" && break
done

if [ -z "$binary" ]; then
  fail "No binary found in archive."
fi

verbose "Copying binary to ${INSTALL_BIN_DIR}/logician"
cp "$binary" "$INSTALL_BIN_DIR/logician"
chmod 0755 "$INSTALL_BIN_DIR/logician"
ok "Binary installed to ${C_BOLD}${INSTALL_BIN_DIR}/logician${C_RESET}"

# ── PATH setup ───────────────────────────────────────────────────────────────
add_to_path() {
  path_action="already"
  case ":$PATH:" in *":$INSTALL_BIN_DIR:"*) return ;; esac
  if [ "$SKIP_PATH_UPDATE" = "1" ]; then path_action="skipped"; return; fi

  profile="${LOGICIAN_INSTALL_SHELL_PROFILE:-$HOME/.profile}"
  if [ -z "${LOGICIAN_INSTALL_SHELL_PROFILE:-}" ]; then
    case "${SHELL:-}" in */zsh) profile="$HOME/.zshrc";; */bash) profile="$HOME/.bashrc";; esac
  fi
  path_profile="$profile"
  path_line="export PATH=\"$INSTALL_BIN_DIR:\$PATH\""
  if [ -f "$profile" ] && grep -F "$path_line" "$profile" >/dev/null 2>&1; then
    path_action="configured"; return
  fi
  printf '\n# Added by Logician installer\n%s\n' "$path_line" >>"$profile"
  path_action="added"
}

add_to_path

case "$path_action" in
  added)      printf '%s\n' "  ${OK} PATH updated in ${C_BOLD}${path_profile}${C_RESET}" ;;
  configured) printf '%s\n' "  ${OK} PATH already configured in ${C_BOLD}${path_profile}${C_RESET}" ;;
  skipped)    printf '%s\n' "  ${WARN} PATH update skipped (LOGICIAN_INSTALL_SKIP_PATH_UPDATE=1)" ;;
  *)          printf '%s\n' "  ${OK} ${C_BOLD}${INSTALL_BIN_DIR}${C_RESET} is already on PATH" ;;
esac

# ── Final output ─────────────────────────────────────────────────────────────
printf '\n%s\n\n' "${C_BOLD}${C_GREEN}  Logician ${resolved_version} installed successfully!${C_RESET}"

if [ "$path_action" = "added" ]; then
  printf '%s\n\n' \
    "  Run the following to activate your PATH, or start a new terminal:"
  printf '%s\n' ""
  printf '%s%s%s\n' "  ${C_CYAN}  . ${path_profile}${C_RESET}" "  # or"
  printf '%s%s\n\n' "  ${C_CYAN}  exec $$${C_RESET}"
fi

printf '%s\n' "  ${C_BOLD}logician${C_RESET}  — start the TUI"
printf '%s\n' "  ${C_DIM}github.com/lseman/logician${C_RESET}\n"
