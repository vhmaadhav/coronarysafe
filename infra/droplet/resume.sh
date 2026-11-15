#!/usr/bin/env bash
# Usage (on the DROPLET, after a spot preemption or reboot):
#   bash /root/coronarysafe/infra/droplet/resume.sh 2>&1 | tee -a /root/resume.log
#
# Re-runs every step in order. Each step is idempotent, so a restart costs only the unfinished work:
#   setup.sh  (checks the venv and the container, re-verifies the sanity check)
#   stage.sh  (skips finished groups and verified Zenodo files)
#   run_all.sh (skips every job whose meta.json is current; the budget keeps counting from /root/.droplet_start)
# A preempted spot instance may come back with a new IP and a wiped boot disk. The scratch disk and /root
# content survive only if it is a persistent volume: check `df` before starting.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=env.sh
. "$ROOT/infra/droplet/env.sh"

if [ -n "$CS_CONTAINER" ] || docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx cs-infer; then
  docker start cs-infer >/dev/null 2>&1 || true
fi
bash "$ROOT/infra/droplet/setup.sh"
bash "$ROOT/infra/droplet/stage.sh"
bash "$ROOT/infra/droplet/run_all.sh"
