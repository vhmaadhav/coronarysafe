#!/usr/bin/env bash
# Usage (on the DROPLET):  bash /root/coronarysafe/droplet/cs.sh droplet/run_infer.py --dry-run --methods cas_net --split test
# Runs a script with the venv Python, inside the Quick Start container when it is running, else on the host.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=env.sh
. "$ROOT/droplet/env.sh"
export_runtime_env
export DATA_ROOT="${DATA_ROOT:-/root/data}"
cs_exec "$PY_CS" "$@"
