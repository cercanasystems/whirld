#!/usr/bin/env bash
# macOS clean-room install + smoke test for Whirld (PRD section 17.3, macOS ARM leg).
#
# Provisions a throwaway macOS VM with Tart (Cirrus Labs' Apple-Silicon VM CLI),
# installs ONLY the published package + its base dependencies (no Homebrew GDAL, no
# dev tools, no pre-built virtualenv), and runs the shared scripts/clean_room_test.py
# inside it. This is the macOS counterpart to docker/clean-room.Dockerfile and reuses
# the exact same smoke test — only the environment provisioning differs.
#
# Prerequisites (Apple Silicon host):
#   brew install cirruslabs/cli/tart hudochenkov/sshpass/sshpass
#
# Usage:
#   scripts/clean_room_macos.sh                       # hermetic: pull + embed GeoTIFF + STAC item
#   WHIRLD_TEST_STAC_URL=<earth-search item> \
#     WHIRLD_TEST_STAC_BBOX=<min_lon,min_lat,max_lon,max_lat> \
#     scripts/clean_room_macos.sh                      # + live /vsicurl/ range reads
#
# Env knobs:
#   VM_NAME     (default: whirld-cleanroom-macos)
#   BASE_IMAGE  (default: ghcr.io/cirruslabs/macos-sequoia-base:latest)
#   KEEP_VM     (set to keep the VM after the run; default: delete it)
set -euo pipefail

VM_NAME="${VM_NAME:-whirld-cleanroom-macos}"
BASE_IMAGE="${BASE_IMAGE:-ghcr.io/cirruslabs/macos-sequoia-base:latest}"
SSH_USER="admin"
SSH_PASS="admin"   # cirruslabs base-image default credentials
IP_TIMEOUT=180     # seconds to wait for the VM to get an IP
SSH_TIMEOUT=180    # seconds to wait for sshd to answer

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROVISION="$REPO_ROOT/scripts/_clean_room_macos_provision.sh"

# --- preconditions ----------------------------------------------------------
[ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] || {
  echo "ERROR: Tart requires an Apple-Silicon (arm64) macOS host." >&2
  exit 1
}
command -v tart >/dev/null || {
  echo "ERROR: tart not found. Install it:" >&2
  echo "       brew install cirruslabs/cli/tart" >&2
  exit 1
}
command -v sshpass >/dev/null || {
  echo "ERROR: sshpass not found (needed for the VM's admin/admin login). Install it:" >&2
  echo "       brew install hudochenkov/sshpass/sshpass" >&2
  exit 1
}
[ -f "$PROVISION" ] || { echo "ERROR: missing $PROVISION" >&2; exit 1; }

VM_PID=""
cleanup() {
  # Best-effort teardown: stop the booted VM, then delete it unless KEEP_VM is set.
  [ -n "$VM_PID" ] && kill "$VM_PID" 2>/dev/null || true
  tart stop "$VM_NAME" 2>/dev/null || true
  if [ -z "${KEEP_VM:-}" ]; then
    tart delete "$VM_NAME" 2>/dev/null || true
    echo "[macos-host] deleted VM '$VM_NAME' (set KEEP_VM=1 to keep it)"
  else
    echo "[macos-host] kept VM '$VM_NAME' (KEEP_VM set)"
  fi
}
trap cleanup EXIT

ssh_run() {  # run a remote command; inherits this function call's stdin (for bash -s)
  sshpass -p "$SSH_PASS" ssh \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o LogLevel=ERROR "$SSH_USER@$VM_IP" "$@"
}

# --- provision a fresh VM ---------------------------------------------------
# Clone only if the VM doesn't already exist (a prior interrupted run or KEEP_VM
# may have left it); the heavy base image is cached regardless.
if tart list 2>/dev/null | awk '{print $2}' | grep -qx "$VM_NAME"; then
  echo "[macos-host] reusing existing VM '$VM_NAME'"
else
  echo "[macos-host] cloning $BASE_IMAGE -> $VM_NAME (first run downloads a large image)"
  tart clone "$BASE_IMAGE" "$VM_NAME"
fi

echo "[macos-host] booting headless with the repo mounted"
tart run --no-graphics --dir=whirld:"$REPO_ROOT" "$VM_NAME" &
VM_PID=$!

echo "[macos-host] waiting for the VM IP (up to ${IP_TIMEOUT}s)"
VM_IP=""
for _ in $(seq "$IP_TIMEOUT"); do
  VM_IP="$(tart ip "$VM_NAME" 2>/dev/null || true)"
  [ -n "$VM_IP" ] && break
  sleep 1
done
[ -n "$VM_IP" ] || { echo "ERROR: VM did not get an IP in ${IP_TIMEOUT}s." >&2; exit 1; }
echo "[macos-host] VM IP: $VM_IP"

echo "[macos-host] waiting for sshd (up to ${SSH_TIMEOUT}s)"
SSH_READY=""
for _ in $(seq "$SSH_TIMEOUT"); do
  if ssh_run true 2>/dev/null; then SSH_READY=1; break; fi
  sleep 2
done
[ -n "$SSH_READY" ] || { echo "ERROR: sshd not reachable in ${SSH_TIMEOUT}s." >&2; exit 1; }

# --- run the smoke test in the guest ----------------------------------------
# Forward only the STAC test vars; pipe the provision script over stdin so the
# spaces in the mount path never reach a command line.
echo "[macos-host] running the clean-room smoke test in the VM"
ssh_run \
  "WHIRLD_TEST_STAC_URL='${WHIRLD_TEST_STAC_URL:-}' \
   WHIRLD_TEST_STAC_BBOX='${WHIRLD_TEST_STAC_BBOX:-}' \
   bash -s" < "$PROVISION"

echo "[macos-host] clean-room PASSED on macOS arm64"
