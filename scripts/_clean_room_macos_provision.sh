#!/usr/bin/env bash
# In-guest provisioning for the macOS clean-room test (PRD section 17.3).
#
# Runs *inside* a fresh Tart macOS VM, piped over SSH stdin by
# scripts/clean_room_macos.sh. It installs Python 3.13 + the Whirld package (base
# deps only — deliberately NO Homebrew GDAL) into a throwaway venv and runs the
# shared, OS-agnostic smoke test scripts/clean_room_test.py.
#
# The host repo is mounted read-only by Tart at "/Volumes/My Shared Files/whirld".
# WHIRLD_TEST_STAC_URL / WHIRLD_TEST_STAC_BBOX, if exported by the SSH caller, flow
# straight through to the smoke test for the live /vsicurl/ check.
set -euo pipefail

MOUNT="/Volumes/My Shared Files/whirld"
BUILD="$HOME/whirld-build"
VENV="$HOME/cr-venv"

echo "[macos-guest] $(sw_vers -productName) $(sw_vers -productVersion) ($(uname -m))"

if [ ! -d "$MOUNT" ]; then
  echo "[macos-guest] FAIL: repo mount not found at '$MOUNT'" >&2
  exit 1
fi

# Copy just the build inputs out of the mount — never the host .venv/.git.
echo "[macos-guest] staging build inputs"
rm -rf "$BUILD"
mkdir -p "$BUILD/scripts"
cp "$MOUNT/pyproject.toml" "$MOUNT/README.md" "$BUILD/"
cp -R "$MOUNT/src" "$BUILD/src"
cp "$MOUNT/scripts/clean_room_test.py" "$BUILD/scripts/clean_room_test.py"

# Python 3.13 via Homebrew (base images ship brew but not 3.13; system python3 is
# too old). No GDAL formula is installed — proving rasterio's bundled GDAL suffices.
echo "[macos-guest] installing python@3.13 (brew)"
export HOMEBREW_NO_AUTO_UPDATE=1 HOMEBREW_NO_INSTALL_CLEANUP=1
brew install python@3.13 >/dev/null
PY="$(brew --prefix python@3.13)/bin/python3.13"

echo "[macos-guest] python: $("$PY" --version)"
echo "[macos-guest] creating venv + installing whirld (base deps only)"
"$PY" -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet "$BUILD"

# Run the shared smoke test in an isolated WHIRLD_HOME. WHIRLD_TEST_STAC_URL/_BBOX
# pass through from the environment if the caller exported them.
WHIRLD_HOME="$(mktemp -d)"
export WHIRLD_HOME
echo "[macos-guest] running clean_room_test.py"
"$VENV/bin/python" "$BUILD/scripts/clean_room_test.py"
