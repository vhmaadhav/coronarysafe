#!/usr/bin/env bash
# Usage (on the DROPLET, as root, after rsync of the repo to /root/coronarysafe):
#   bash /root/coronarysafe/droplet/setup.sh 2>&1 | tee -a /root/setup.log
#
# Idempotent. Re-run after a spot preemption (resume.sh does this for you).
#
# Two paths, chosen automatically:
#   A) PREFERRED: the AMD "PyTorch" Quick Start image (Docker, PyTorch 2.14 on ROCm 10.1 host, Ubuntu 24.04).
#      Detected from `docker images`. A container CS_CONTAINER is started with the GPU devices and our repo and
#      scratch mounted, and this script re-runs itself inside it to add the extra packages to a venv that
#      sees the image's torch (--system-site-packages).
#   B) FALLBACK: bare host with ROCm. Installs a ROCm PyTorch wheel from download.pytorch.org.
#      ROCm 6.2/6.3/6.4 are verified against the official index. ROCm 7.x is UNVERIFIED (nightly index).
#
# Fails loudly on any check. Billing starts when the droplet is created; the boot epoch is written to
# /root/.droplet_start (unless it exists already) and is the clock run_infer.py uses.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=env.sh
. "$ROOT/droplet/env.sh"

IMAGECAS_DIR="${IMAGECAS_DIR:-/root/ImageCAS-X}"
IMAGECAS_COMMIT="dbc7343187adf45ca9306dc3460eebc70f92b204"  # ls-remote HEAD, 2026-11-13
PYTORCH_IMAGE="${PYTORCH_IMAGE:-}"       # override the Quick Start image name if auto-detect misses
INSIDE="${CS_INSIDE:-0}"                 # 1 when this script runs inside the container

log() { echo "setup: $*"; }
die() { echo "setup: ERROR: $*" >&2; exit 2; }

# 1. Billing clock.
if [ ! -f /root/.droplet_start ]; then
  awk -v now="$(date +%s)" '{ printf "%d\n", now - $1 }' /proc/uptime > /root/.droplet_start
  log "recorded droplet start epoch in /root/.droplet_start"
fi

# 2. Scratch disk and the /root/data symlink (host only; the container mounts the same path).
if [ "$INSIDE" != 1 ]; then
  setup_data_root || die "data root setup failed"
  log "data root: $(readlink -f /root/data) (scratch $SCRATCH_DIR)"
fi

# 3. Choose the path. The Quick Start image is preferred when present.
if [ "$INSIDE" != 1 ] && command -v docker >/dev/null 2>&1; then
  if [ -z "$PYTORCH_IMAGE" ]; then
    PYTORCH_IMAGE="$(docker images --format '{{.Repository}}:{{.Tag}}' \
      | grep -iE 'pytorch|rocm' | grep -viE 'vllm|sglang|jax|megatron' | head -1 || true)"
  fi
fi

if [ "$INSIDE" != 1 ] && [ -n "$PYTORCH_IMAGE" ]; then
  log "path A: PyTorch Quick Start image $PYTORCH_IMAGE"
  if ! docker ps --format '{{.Names}}' | grep -qx cs-infer; then
    docker rm -f cs-infer >/dev/null 2>&1 || true
    docker run -d --name cs-infer \
      --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 64g \
      --security-opt seccomp=unconfined \
      -v /root:/root -v "$SCRATCH_DIR:$SCRATCH_DIR" \
      -e CS_INSIDE=1 -e DATA_ROOT=/root/data \
      "$PYTORCH_IMAGE" sleep infinity >/dev/null
    log "started container cs-infer"
  fi
  # Re-run this script inside the container. The repo is mounted at /root/coronarysafe via /root.
  docker exec -e CS_INSIDE=1 -w /root/coronarysafe cs-infer bash /root/coronarysafe/droplet/setup.sh
  exit 0
fi

PY="${PYTHON:-python3}"
CONTAINER_MODE=0
if [ "$INSIDE" = 1 ]; then CONTAINER_MODE=1; fi

# 4. Virtualenv. Inside the container the image's torch must stay visible, so use --system-site-packages.
if [ ! -x "$VENV/bin/python" ]; then
  if [ "$CONTAINER_MODE" = 1 ]; then
    "$PY" -m venv --system-site-packages "$VENV"
  else
    "$PY" -m venv "$VENV"
  fi
fi
PIP="$VENV/bin/pip"
"$PIP" install -q -U pip wheel

# 5. Torch. Path A keeps the image's torch (must be a HIP build). Path B installs a ROCm wheel.
if [ "$CONTAINER_MODE" = 1 ]; then
  log "path A (container): using the image's torch; not reinstalling"
else
  if [ ! -f /opt/rocm/.info/version ]; then
    die "no /opt/rocm/.info/version. Bare-host path needs ROCm. Use the PyTorch Quick Start image (README step 3)."
  fi
  ROCM="$(head -1 /opt/rocm/.info/version | grep -oE '^[0-9]+\.[0-9]+' || true)"
  [ -n "$ROCM" ] || die "could not parse ROCm version from /opt/rocm/.info/version"
  case "$ROCM" in
    6.2) TORCH_INDEX="https://download.pytorch.org/whl/rocm6.2"; TORCH_SPEC="torch==2.5.1"; PRE="" ;;   # verified: index lists 2.5.0, 2.5.1
    6.3) TORCH_INDEX="https://download.pytorch.org/whl/rocm6.3"; TORCH_SPEC="torch==2.7.1"; PRE="" ;;   # verified: 2.7.0, 2.7.1, 2.8.0, 2.9.x
    6.4) TORCH_INDEX="https://download.pytorch.org/whl/rocm6.4"; TORCH_SPEC="torch==2.8.0"; PRE="" ;;   # verified: 2.8.0, 2.9.0, 2.9.1
    7.*) TORCH_INDEX="https://download.pytorch.org/whl/nightly/rocm7.2"; TORCH_SPEC="torch"; PRE="--pre"
         log "UNVERIFIED: ROCm $ROCM maps to the nightly rocm7.2 index (AMD docs). No stable index confirmed." ;;
    *) die "ROCm $ROCM has no mapped torch index. Set TORCH_INDEX and TORCH_SPEC env vars, or use the Quick Start image." ;;
  esac
  TORCH_INDEX="${TORCH_INDEX_OVERRIDE:-$TORCH_INDEX}"
  TORCH_SPEC="${TORCH_SPEC_OVERRIDE:-$TORCH_SPEC}"
  if ! "$VENV/bin/python" -c "import torch, sys; sys.exit(0 if torch.version.hip else 1)" 2>/dev/null; then
    log "ROCm $ROCM -> $TORCH_SPEC from $TORCH_INDEX"
    "$PIP" install -q $PRE --index-url "$TORCH_INDEX" "$TORCH_SPEC"
  fi
fi

# 6. Pinned stack. torch is not in this list: nnunetv2 and monai need only torch>=2.1.
"$PIP" install -q \
  "numpy==1.26.4" "monai==1.4.0" "nnunetv2==2.6.2" "SimpleITK==2.4.0" "nibabel==5.3.2" \
  "scikit-image==0.24.0" "scipy==1.14.1" "scikit-learn==1.5.2" "vtk==9.3.1" \
  tqdm matplotlib openpyxl pandas "kaggle==2.2.4" six requests pyyaml
log "python stack installed"

# 7. System packages (host only). Needed for the split archives and vtk/opengl imports.
# Installed only when a tool is missing (the Quick Start image may already have them).
need_pkgs=()
for tool in 7z zip unzip git; do command -v "$tool" >/dev/null || need_pkgs+=("$tool"); done
if [ "${#need_pkgs[@]}" -gt 0 ] && command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq git p7zip-full unzip zip python3-venv libgl1 libglib2.0-0 libxt6 \
    libxrender1 libsm6 libice6 libx11-6 >/dev/null
fi
command -v 7z >/dev/null || die "7z missing (apt install p7zip-full)"
command -v zip >/dev/null || die "zip missing (apt install zip)"

# 8. ImageCAS-X at the pinned commit, editable, without re-resolving deps.
if [ ! -d "$IMAGECAS_DIR/.git" ]; then
  git clone -q https://github.com/kitbransby/ImageCAS-X "$IMAGECAS_DIR"
fi
git -C "$IMAGECAS_DIR" fetch -q origin
git -C "$IMAGECAS_DIR" checkout -q "$IMAGECAS_COMMIT"
"$PIP" install -q --no-deps -e "$IMAGECAS_DIR"
log "ImageCAS-X at $(git -C "$IMAGECAS_DIR" rev-parse --short HEAD)"

# 9. MIOpen cache and runtime env, then the sanity check: GPU visible, Conv3d fp32 and bf16 run, and the
#    nnU-Net predictor, ImageCAS-X and the QA code import. Non-zero exit on any failure.
export_runtime_env
"$VENV/bin/python" - <<'PYEOF'
import sys
import torch

def fail(msg):
    print(f"sanity: FAIL {msg}")
    sys.exit(1)

print("sanity: torch", torch.__version__, "hip", torch.version.hip)
if torch.version.hip is None:
    fail("torch is not a ROCm (HIP) build")
if not torch.cuda.is_available():
    fail("torch.cuda.is_available() is False; check the GPU, /dev/kfd and /dev/dri")
n = torch.cuda.device_count()
name = torch.cuda.get_device_name(0)
print("sanity: device", name, "count", n)
if "MI300" not in name and "gfx94" not in name:
    print("sanity: WARNING device name does not look like MI300X; check before spending budget")
conv = torch.nn.Conv3d(1, 8, kernel_size=3, padding=1).cuda()
x = torch.randn(1, 1, 64, 64, 64, device="cuda")
y32 = conv(x)
torch.cuda.synchronize()
with torch.autocast("cuda", dtype=torch.bfloat16):
    y16 = conv(x)
torch.cuda.synchronize()
print("sanity: conv3d fp32", tuple(y32.shape), "bf16", tuple(y16.shape), y16.dtype)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: F401
import SimpleITK, nibabel, monai, skimage  # noqa: F401
import inference, models.registry  # noqa: F401 (ImageCAS-X)
print("sanity: nnunetv2 predictor, ImageCAS-X, monai, nibabel, SimpleITK import ok")
print("sanity: PASS")
PYEOF

# 10. Record what was verified, for the RUNLOG and the fetch step.
mkdir -p "$ROOT/docs"
echo "setup: done ($([ "$CONTAINER_MODE" = 1 ] && echo container || echo host)). Next: bash $ROOT/droplet/stage.sh"
