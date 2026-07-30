#!/bin/sh
# Logician installer
# Usage: curl -fsSL https://raw.githubusercontent.com/seman/logician/main/tui/install.sh | bash
#        curl -fsSL https://raw.githubusercontent.com/seman/logician/main/tui/install.sh | bash -s -- 0.2.0

set -eu

VERSION="${1:-latest}"
INSTALL_BIN_DIR="${LOGICIAN_INSTALL_BIN_DIR:-$HOME/.local/bin}"
INSTALL_APP_DIR="${LOGICIAN_INSTALL_APP_DIR:-$HOME/.local/share/logician}"
SKIP_PATH_UPDATE="${LOGICIAN_INSTALL_SKIP_PATH_UPDATE:-0}"
path_action="already"
path_profile=""

step() { printf '==> %s\n' "$1"; }

download_file() {
  url="$1"
  output="$2"
  if command -v curl >/dev/null 2>&1; then
    if [ -t 2 ]; then curl -fL --progress-bar "$url" -o "$output"
    else curl -fsSL "$url" -o "$output"; fi
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    if [ -t 2 ]; then wget --show-progress -O "$output" "$url"
    else wget -q -O "$output" "$url"; fi
    return
  fi
  echo "curl or wget is required." >&2; exit 1
}

download_text() {
  if command -v curl >/dev/null 2>&1; then curl -fsSL "$1"
  elif command -v wget >/dev/null 2>&1; then wget -q -O - "$1"
  else echo "curl or wget is required." >&2; exit 1; fi
}

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

# --- Platform detection ---
case "$(uname -s)" in
  Darwin) os="darwin" ;;
  Linux)  os="linux" ;;
  *) echo "Supports macOS and Linux only." >&2; exit 1 ;;
esac

case "$(uname -m)" in
  x86_64|amd64) arch="x86_64" ;;
  arm64|aarch64) arch="arm64" ;;
  *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

# --- Resolve version ---
case "$VERSION" in ""|latest|stable) VERSION="latest";; v*) VERSION="${VERSION#v}";; esac

if [ "$VERSION" = "latest" ]; then
  release_json="$(download_text "https://api.github.com/repos/seman/logician/releases/latest")"
  resolved_version="$(printf '%s\n' "$release_json" | sed -n 's/.*"tag_name":"v\{0,1\}\([^",]*\)".*/\1/p' | head -n 1)"
  if [ -z "$resolved_version" ]; then echo "Failed to resolve latest version." >&2; exit 1; fi
else
  resolved_version="$VERSION"
fi

asset_name="logician-${os}-${arch}"
base_url="https://github.com/seman/logician/releases/download/v${resolved_version}"
download_url="${base_url}/${asset_name}"

step "Installing Logician ${resolved_version} (${os}-${arch})"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT INT TERM

step "Downloading"
if ! download_file "$download_url" "$tmp_dir/${asset_name}.tar.gz"; then
  cat >&2 <<EOF
Failed to download ${asset_name} from:
  ${download_url}

Pass an exact version: curl ... | bash -s -- 0.2.0
EOF
  exit 1
fi

# --- Extract ---
mkdir -p "$INSTALL_APP_DIR" "$INSTALL_BIN_DIR"
rm -rf "$INSTALL_APP_DIR/logician-${resolved_version}"
tar -xzf "$tmp_dir/${asset_name}.tar.gz" -C "$INSTALL_APP_DIR"

# Bun --compile produces a single binary; find it
binary=""
for f in "$INSTALL_APP_DIR/logician-${resolved_version}"/*; do
  [ -f "$f" ] && binary="$f" && break
done

if [ -z "$binary" ]; then
  echo "Error: no binary found in archive." >&2; exit 1
fi

cp "$binary" "$INSTALL_BIN_DIR/logician"
chmod 0755 "$INSTALL_BIN_DIR/logician"

add_to_path

case "$path_action" in
  added)     step "PATH updated in $path_profile";;
  configured) step "PATH already configured in $path_profile";;
  skipped)   step "PATH update skipped";;
  *)         step "$INSTALL_BIN_DIR is already on PATH";;
esac

printf 'Logician %s installed to %s/logician\n' "$resolved_version" "$INSTALL_BIN_DIR"
