#!/usr/bin/env bash
# AAUBot-Python — host install + GPU/Docker setup for the VM.
#
# Idempotent: safe to re-run. Each step prints PASS/FAIL and, on failure,
# dumps diagnostics so you can paste the output back for help.
#
# Covers the problems seen during setup:
#   - corrupted apt source file (404 HTML written into sources.list.d/*.list)
#   - deprecated per-distro NVIDIA repo URL  -> correct generic stable/deb repo
#   - no systemd as PID 1                    -> service / init.d fallback
#   - "permission denied" on docker.sock     -> add user to docker group
#   - overlay2 "operation not permitted"     -> switch storage-driver to vfs
#   - must preserve daemon.json (nvidia runtime) when editing it
set -uo pipefail

# --- config ---------------------------------------------------------------
NVIDIA_LIST_URL="https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list"
NVIDIA_GPG_URL="https://nvidia.github.io/libnvidia-container/gpgkey"
GPU_TEST_IMAGE="nvidia/cuda:12.4.0-base-ubuntu22.04"
KEYRING="/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
LIST_FILE="/etc/apt/sources.list.d/nvidia-container-toolkit.list"

# --- pretty printing -------------------------------------------------------
GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'; BOLD=$'\033[1m'; RST=$'\033[0m'
step=0
ok()   { echo "${GREEN}PASS${RST} $*"; }
fail() { echo "${RED}FAIL${RST} $*"; }
warn() { echo "${YELLOW}WARN${RST} $*"; }
hdr()  { step=$((step+1)); echo; echo "${BOLD}[$step] $*${RST}"; }

# Sudo-aware: use sudo only when not already root
s() { if [ "$(id -u)" -eq 0 ]; then "$@"; else sudo "$@"; fi; }

# --- diagnostics dump (called on hard failure or at the end) ---------------
debug_dump() {
    echo
    echo "================= DIAGNOSTICS ================="
    echo "--- whoami / groups ---"; id 2>&1
    echo "--- OS ---"; (. /etc/os-release && echo "$PRETTY_NAME") 2>&1
    echo "--- kernel / WSL2 check ---"; grep -i microsoft /proc/version 2>&1 || echo "(not WSL2)"
    echo "--- PID 1 (init system) ---"; ps -p 1 -o comm= 2>&1
    echo "--- host nvidia-smi ---"; nvidia-smi 2>&1 || echo "(nvidia-smi missing — host NVIDIA driver not installed?)"
    echo "--- docker version ---"; docker version 2>&1 || echo "(docker not reachable — socket perms or daemon down)"
    echo "--- docker info (storage driver, runtime) ---"; docker info 2>&1 | grep -Ei 'storage|runtime|server version|kernel|runtimes' || true
    echo "--- /etc/docker/daemon.json ---"; cat /etc/docker/daemon.json 2>&1 || echo "(no daemon.json)"
    echo "--- NVIDIA repo .list file ---"; cat "$LIST_FILE" 2>&1 || echo "(no $LIST_FILE)"
    echo "--- docker group members ---"; getent group docker 2>&1 || echo "(no docker group)"
    echo "--- docker daemon logs (last 30) ---"
    if command -v journalctl >/dev/null && [ -d /var/log/journal ]; then
        s journalctl -u docker --no-pager -n 30 2>&1 || true
    else
        s tail -n 30 /var/log/docker.log 2>&1 || s tail -n 30 /var/log/syslog 2>&1 | grep -i docker || echo "(no logs found)"
    fi
    echo "=============== END DIAGNOSTICS ============="
    echo
    echo "If a step FAILED, paste everything from that step through END DIAGNOSTICS for help."
}

trap 'fail "aborted at step $step"; debug_dump; exit 1' ERR

# ===========================================================================
echo "${BOLD}=== AAUBot-Python host setup ===${RST}"
echo "Goal: GPU visible inside a Docker container, then AAUBot can run."
echo

# --- 0. environment scan ---------------------------------------------------
hdr "Detect environment"
OS_ID=""; (. /etc/os-release && OS_ID=$ID) 2>/dev/null
echo "  distro:  ${OS_ID:-unknown}"
echo "  init:    $(ps -p 1 -o comm= 2>/dev/null || echo unknown)"
echo "  WSL2:    $(grep -iq microsoft /proc/version && echo yes || echo no)"
echo "  docker:  $(command -v docker >/dev/null && echo present || echo MISSING)"
echo "  compose: $(docker compose version 2>/dev/null | head -1 || echo MISSING-v2)"

# --- 1. fix corrupted apt source files ------------------------------------
hdr "Remove corrupted apt source files (404 HTML written as .list)"
CORRUPT=0
for f in /etc/apt/sources.list.d/*.list; do
    [ -f "$f" ] || continue
    if grep -qiE '<!doctype|<html' "$f" 2>/dev/null; then
        warn "removing HTML-contaminated source: $f"
        s rm -f "$f"
        CORRUPT=$((CORRUPT+1))
    fi
done
[ "$CORRUPT" -eq 0 ] && ok "no corrupted source files" || ok "removed $CORRUPT corrupted file(s)"

# --- 2. prerequisites ------------------------------------------------------
hdr "Install prerequisites (ca-certificates curl gnupg2 jq)"
s apt-get update
s apt-get install -y --no-install-recommends ca-certificates curl gnupg2 jq
ok "prerequisites installed"

# --- 3. NVIDIA Container Toolkit repo (correct generic stable/deb) --------
hdr "Add NVIDIA Container Toolkit repo (generic stable/deb)"
s curl -fsSL "$NVIDIA_GPG_URL" | s gpg --dearmor -o "$KEYRING"
s curl -s -L "$NVIDIA_LIST_URL" \
  | sed 's#deb https://#deb [signed-by='"$KEYRING"'] https://#g' \
  | s tee "$LIST_FILE" >/dev/null
ok "repo added at $LIST_FILE"

# --- 4. install toolkit ----------------------------------------------------
hdr "Install nvidia-container-toolkit"
s apt-get update
if s apt-get install -y nvidia-container-toolkit; then
    ok "nvidia-container-toolkit $(nvidia-ctk version 2>/dev/null | head -1 || echo installed)"
else
    fail "toolkit install failed"; debug_dump; exit 1
fi

# --- 5. configure Docker daemon (nvidia runtime + vfs storage driver) -----
hdr "Configure Docker daemon (nvidia runtime + vfs storage-driver)"
# nvidia-ctk writes the nvidia runtime block into daemon.json
s nvidia-ctk runtime configure --runtime=docker
# merge storage-driver=vfs WITHOUT dropping the nvidia runtime (overlay2 often
# fails with "operation not permitted" on restricted/WSL2 kernels)
if [ -f /etc/docker/daemon.json ]; then
    s jq '. + {"storage-driver": "vfs"}' /etc/docker/daemon.json | s tee /etc/docker/daemon.json.new >/dev/null
    s mv /etc/docker/daemon.json.new /etc/docker/daemon.json
fi
echo "  daemon.json now:"; s cat /etc/docker/daemon.json | sed 's/^/    /'
ok "daemon configured (nvidia runtime + vfs)"

# --- 6. restart Docker (systemd / service / init.d fallbacks) --------------
hdr "Restart Docker daemon (non-systemd aware)"
if s systemctl restart docker 2>/dev/null; then
    ok "restarted via systemctl"
elif s service docker restart 2>/dev/null; then
    ok "restarted via service"
elif [ -x /etc/init.d/docker ] && s /etc/init.d/docker restart 2>/dev/null; then
    ok "restarted via init.d"
else
    fail "could not restart docker"; debug_dump; exit 1
fi

# --- 7. add user to docker group (fixes docker.sock permission denied) -----
hdr "Add current user to docker group"
if id -nG | tr ' ' '\n' | grep -qx docker; then
    ok "user already in docker group"
else
    s usermod -aG docker "$USER"
    warn "user added to docker group — run 'newgrp docker' or log out/in for it to take effect"
fi

# --- 8. host GPU check -----------------------------------------------------
hdr "Check host NVIDIA driver"
if nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    ok "host GPU: ${GPU_NAME:-?} (${VRAM:-?} MiB total)"
else
    fail "nvidia-smi not working on host"
    warn "The host NVIDIA driver is missing. nvidia-container-toolkit is NOT the driver."
    warn "Install the NVIDIA driver for your distro first, then re-run this script."
    debug_dump; exit 1
fi

# --- 9. container GPU check ------------------------------------------------
hdr "Check GPU visibility inside a container"
# if not yet in docker group, use sudo for this probe
if id -nG | tr ' ' '\n' | grep -qx docker; then DOCKER=docker; else DOCKER="sudo docker"; fi
if $DOCKER run --rm --gpus all "$GPU_TEST_IMAGE" nvidia-smi >/tmp/gpu_container.log 2>&1; then
    ok "GPU visible inside container"
    grep -E "GPU  Name|MiB /" /tmp/gpu_container.log | head -2 | sed 's/^/    /'
else
    fail "GPU NOT visible inside container"; warn "see /tmp/gpu_container.log"
    tail -n 5 /tmp/gpu_container.log | sed 's/^/    /'
    debug_dump; exit 1
fi

# --- done ------------------------------------------------------------------
echo
echo "${GREEN}${BOLD}=== SETUP COMPLETE ===${RST}"
echo "GPU is wired into Docker. Next, from the AAUBot repo root:"
echo "  cp .env.example .env"
echo "  # edit config.yaml: set test.enabled: true   (Chainlit browser UI)"
echo "  docker compose up --build"
echo
echo "If 'docker compose up' says 'permission denied', run 'newgrp docker'"
echo "or open a fresh SSH session so the docker-group change applies."
echo
warn "If you hit a Blackwell/CUDA-architecture error from vLLM later, that is"
warn "a separate issue (vLLM image must support the B200) — re-run with that error."
