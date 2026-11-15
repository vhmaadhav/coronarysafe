# Inference and labelling on the AMD MI300X droplet

Two ways to run. **Use A (PyTorch Quick Start container) unless it is unavailable.** B is the fallback.

- Droplet: MI300X x1 (192 GB VRAM, 20 vCPU, 240 GB RAM), boot disk 720 GB, scratch 5 TB NVMe, $1.99/h.
- Credit: **$100, expiring in 2 days**. After that the card is charged, so the budget cap must hold.
- **Budget: `BUDGET_USD=90`** (default in `run_infer.py`). The queue hard-stops at 90 with exit code 3
  and leaves every finished prediction on disk. Raise it only with Maadhav's approval.
- **Spot droplets are possible** (cheaper, interruptible, on-demand may be out of capacity). Every step is
  resumable. After a preemption, run `resume.sh` (step 7).
- **Billing does not stop when the droplet is powered off.** Destroy it in the AMD console at the end.

Data root: `/root/data`, a symlink to `<scratch>/coronarysafe-data` on the 5 TB disk. Scratch is found
automatically (largest mount with at least 1 TB); set `SCRATCH=/path` to override.

## 0. Laptop, before creating the droplet

- `~/.ssh/amd_devcloud` key, `~/.kaggle/kaggle.json`.
- Nothing else. No Modal runs are needed any more (labelling runs on the droplet).

Create the droplet in the AMD console: MI300X x1, scratch disk attached. Choose **PyTorch Quick Start**
(Ubuntu 24.04, Docker image with PyTorch) for path A. Note the IP as `<ip>`.

## 1. Copy the code and the Kaggle token (laptop)

```bash
cd /home/maadhav/Documents/Development/coronarysafe
rsync -az -e "ssh -i ~/.ssh/amd_devcloud" --exclude .venv --exclude outputs --exclude data \
  --exclude __pycache__ --exclude '*.egg-info' ./ root@<ip>:/root/coronarysafe/
ssh -i ~/.ssh/amd_devcloud root@<ip> 'mkdir -p /root/.kaggle'
scp -i ~/.ssh/amd_devcloud ~/.kaggle/kaggle.json root@<ip>:/root/.kaggle/kaggle.json
```

## 2. Log in and check the image (droplet)

```bash
ssh -i ~/.ssh/amd_devcloud root@<ip>
docker images --format '{{.Repository}}:{{.Tag}}'    # expect a PyTorch image
df -h | grep -E '/mnt|/scratch|/root'               # expect the 5 TB scratch mount
cat /opt/rocm/.info/version 2>/dev/null               # informational only
```

If `docker images` shows no PyTorch image, use path B (step 3). If the auto-detected name is wrong, set
`PYTORCH_IMAGE=<name>` before setup.

## 3. Setup (droplet)

```bash
bash /root/coronarysafe/infra/droplet/setup.sh 2>&1 | tee -a /root/setup.log
```

- **Path A (container):** starts `cs-infer` with `/dev/kfd`, `/dev/dri`, `--ipc=host`, `--shm-size 64g`,
  mounts `/root` and the scratch disk at the same paths, and runs the extra packages in a venv with
  `--system-site-packages` so the image's torch is used.
- **Path B (bare host):** installs a ROCm PyTorch wheel. ROCm 6.2, 6.3 and 6.4 are verified against the
  official index (torch 2.5.1, 2.7.1, 2.8.0). **ROCm 7.x is UNVERIFIED:** it maps to the nightly
  `rocm7.2` index. Override with `TORCH_INDEX_OVERRIDE` and `TORCH_SPEC_OVERRIDE` if needed.
  The exact ROCm version of the MI300X bare image is not confirmed: check `/opt/rocm/.info/version`.
- Ends with a sanity check (`sanity: PASS`). Non-zero exit on any failure.

## 4. Stage the data (droplet)

```bash
bash /root/coronarysafe/infra/droplet/stage.sh 2>&1 | tee -a /root/stage.log
```

Zenodo labels and weights are byte- and md5-checked. The five Kaggle groups download in parallel
(`.change2zip` is renamed to `.zip`; 7z, with `zip -s 0` as fallback). It needs about 150 GB on the
scratch disk and expects 1000 CT volumes at the end. It is resumable: a finished group is skipped.

## 5. First 10 minutes: smoke checklist (droplet)

Run these before the full queue. Use `bash infra/droplet/cs.sh` so commands run in the container when it is up.

```bash
cd /root/coronarysafe
# (a) setup sanity: must say PASS, device name must look like MI300X
grep -E "sanity: (PASS|device)" /root/setup.log | tail -3
# (b) inputs for every item, no GPU
DRY_RUN=1 bash infra/droplet/run_all.sh
# (c) one scan per method on the test split, with Dice against GT
bash infra/droplet/cs.sh infra/droplet/run_infer.py --methods cas_net,swin_unetr,ffr_unet,nnunet_f0 \
  --split test --limit 1 --evaluate --name smoke
# (d) check: cas_net Dice on that scan should be close to 0.9 (published 0.912 +- 0.028;
#     the G0 reproduction gave 0.909). Also check the per-method Dice is not 0 or NaN.
cat /root/data/eval/smoke.csv
```

Go / no-go:
- (a) must say PASS. (b) must print `inputs OK` for every item.
- (c) every method must finish with status ok. A failed method has its error in `docs/RUNLOG.md`.
- (d) cas_net Dice about 0.9. If it is far below (for example under 0.8), **stop** and check the
  preprocessing and the weights before spending more budget.

## 6. Full queue (droplet)

```bash
cd /root/coronarysafe
BUDGET_USD=90 nohup bash infra/droplet/run_all.sh > /root/run_all.out 2>&1 &
tail -f /root/run_all.out            # Ctrl-C does not stop the run
tail -5 /root/coronarysafe/docs/RUNLOG.md
```

Items, in order: (a) verify folds, 16 train scans x 5 folds with probabilities; (b) nnunet_f0..f4 on test;
(c) cas_net, swin_unetr and ffr_unet on val+test with mirror TTA; (d) nnunet_f0..f4 on val. Optional:
`RUN_OOF=1` adds nnunet_oof on train+val at the end (the fold reconstruction is still unverified; use it
only after (a) shows the lowest Dice in the reconstructed fold for most scans).

Each item is idempotent. A failed job is recorded and the queue continues. Exit 3 means the budget
stopped the queue: nothing is killed silently, and the RUNLOG has the cost.

Labelling (CPU, after the predictions for a split exist; can run while the GPU queue is still going):

```bash
bash /root/coronarysafe/infra/droplet/run_labels.sh cas_net,swin_unetr,ffr_unet,nnunet_oof test
bash /root/coronarysafe/infra/droplet/run_labels.sh cas_net,swin_unetr,ffr_unet,nnunet_oof val
```

Uses up to 16 processes; output in `/root/data/qa/v3/`, report `report_<split>.txt`.

## 7. Spot preemption or reboot

```bash
bash /root/coronarysafe/infra/droplet/resume.sh 2>&1 | tee -a /root/resume.log
```

It re-runs setup (checks the container and the sanity check), stage (skips finished groups), and the
queue (skips finished jobs). The budget keeps counting from `/root/.droplet_start`, which is kept on the
boot disk. If the boot disk is wiped, the clock restarts: check the AMD console cost before continuing.

## 8. Fetch the results (laptop), then destroy

```bash
cd /home/maadhav/Documents/Development/coronarysafe
bash infra/droplet/fetch.sh <ip>                 # add --with-probs only if you need the probability maps
```

It copies predictions (masks, meta, flips), `qa/`, `meta/` and the RUNLOG to `data/droplet/`, and checks the
meta.json count matches before it reminds you to destroy.

**Then DESTROY the droplet in the AMD console.** Stopping it is not enough.

## Time and cost estimate

Measured earlier on Modal L40S: CAS-Net about 14 s per scan without TTA, nnU-Net about 78 s per scan without
mirroring. MI300X numbers are **assumptions** (about 1.5 to 2x faster per stream, six streams).

Assumed split sizes: test 160 scans (ImageCAS-X), val about 160 (**unverified**: the real count comes from the
filelists after staging).

| Item | Work | Est. wall time | Est. cost at $1.99/h |
|---|---|---|---|
| Setup and staging | container or venv, download and unzip | 1.5 to 3.5 h | $3 to $7 |
| Smoke (section 5) | 4 methods x 1 scan | 0.1 to 0.2 h | under $1 |
| (a) verify folds | 80 nnU-Net predictions | 0.5 to 0.8 h | $1 to $2 |
| (b) nnunet_f0..f4 test | 800 predictions | 4.4 to 7.8 h | $9 to $16 |
| (c) three networks, TTA, val+test | 720 scans with TTA | 4 to 7 h, more for Swin | $8 to $14 and up |
| (d) nnunet_f0..f4 val | 800 predictions | 4.4 to 7.8 h | $9 to $16 |
| Labelling (CPU, overlaps GPU work) | 4 methods x val+test | 0.5 to 2 h | included in the wall time above |
| Optional (e) nnunet_oof train+val | 640 predictions | 3.6 to 6.2 h | $7 to $12 |
| **Total without (e)** | | **about 15 to 28 h** | **about $30 to $56** |
| **Total with (e)** | | **about 19 to 34 h** | **about $38 to $68** |

Budget: $90 leaves room for the upper estimate with (e). The **$100 credit expires in 2 days**, so
the run must fit in about 48 h wall time including setup. On spot pricing, set `GPU_RATE` to the spot rate
before starting, for example `GPU_RATE=1.2 BUDGET_USD=90 bash infra/droplet/run_all.sh`.

## Concurrency notes

- MI300X: 6 GPU worker processes by default (`--workers 0` is auto: min of 6, VRAM cap and RAM cap).
- CPU: 20 vCPU. Each worker gets `(vCPU - 4) / workers` torch threads (about 2 each at 6 workers).
- RAM: 240 GB. The per-worker RAM estimates are 12 GB (nnU-Net) and 8 GB (networks). Measure
  on the first run and adjust `RAM_GB_NNUNET` / `RAM_GB_NET` in `infra/droplet/run_infer.py`.
- Labelling pool: 16 processes (`infra/droplet/run_labels.py`).

## Files

- `env.sh` shared paths, scratch detection, container helper, nnU-Net and MIOpen env
- `setup.sh` path A (container) and path B (bare ROCm), sanity check
- `stage.sh`, `stage_zenodo.py`, `stage_kaggle_group.py` data staging
- `run_infer.py` job queue (`--dry-run`, `--summary`, `--evaluate`, `--limit`, budget hard stop)
- `run_all.sh` the ordered queue (a) to (e)
- `run_labels.sh`, `run_labels.py` CPU labelling pool
- `resume.sh` restart after preemption, `cs.sh` run a script in the container, `fetch.sh` laptop-side pull
