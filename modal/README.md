# Modal staging for coronarysafe

`stage_data.py` copies the coronary CTA data into the Modal Volume `coronarysafe-data`, mounted at `/data`.

- `labels`: Zenodo record 23894668 (ImageCAS-X labels and filelists, plus pretrained weights).
- `volumes`: Kaggle `xiaoweixumedicalai/imagecas` split archives (NIfTI CT volumes), one container per 200-case group.
- `inspect`: prints the manifest and the weights tree.

## One-time setup (you run this, not the agent)

1. Create a Kaggle API token at kaggle.com, under Settings, then "Create New Token". This downloads `kaggle.json`.
2. Create the Modal secret from your own values. Do not paste the key into chat or commit it:

   ```bash
   modal secret create kaggle KAGGLE_USERNAME=<your_kaggle_username> KAGGLE_KEY=<your_kaggle_key>
   ```

   Or read them from `kaggle.json` in your shell:

   ```bash
   modal secret create kaggle \
     KAGGLE_USERNAME="$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["username"])')" \
     KAGGLE_KEY="$(python3 -c 'import json,os;print(json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["key"])')"
   ```

3. Accept the dataset rules on the Kaggle website for `xiaoweixumedicalai/imagecas` if a download returns 403.

## Running the steps

Run from the repository root. Use `--detach` for the long steps so they keep running after your terminal closes.

```bash
modal run --detach modal/stage_data.py --step labels    # Zenodo labels and weights, about 2.5 GB
modal run --detach modal/stage_data.py --step volumes   # Kaggle NIfTI volumes, about 90 GB, 5 parallel containers
modal run modal/stage_data.py --step inspect            # print /data/manifest and the weights tree
```

Steps are idempotent. The labels step skips downloads that are already verified. Each volumes group is skipped once `manifest/groups/<group>.json` exists.

Outputs in the volume:

- `manifest/labels_tree.txt`: the directory tree to depth 4, with sizes.
- `manifest/volumes.csv`: file, group, kind, bytes, shape, and spacing (read from the NIfTI header only).
- `imagecas/volumes/`, `imagecas/orig_labels/`, `imagecas/meta/`: the extracted data.

Check the Modal version with `modal --version`. The code targets the Modal 1.x client, which was 1.6.3 when this was written.

# GPU inference (`modal/infer.py`)

Runs the ImageCAS-X baselines and the nnU-Net folds on Modal, writing predictions to
`/data/preds/<method>/<scan_id>/` in the original CT geometry. Network classes, resampling,
HU window, 50%-overlap tiling, mirror TTA and postprocessing are reused from
`kitbransby/ImageCAS-X` at commit `dbc7343187adf45ca9306dc3460eebc70f92b204` (MIT). The
image installs that commit with `pip install --no-deps`.

Methods: `cas_net`, `swin_unetr`, `ffr_unet`, `nnunet_f0` .. `nnunet_f4`. Each nnU-Net fold is
predicted separately with nnunetv2 (`use_folds=(k,)`). GPU is the module constant `GPU = "L40S"`.

```bash
modal run modal/infer.py --step check          # CPU: imports, builds all heads, counts staged CTs
modal run modal/infer.py --step folds          # CPU: writes /data/meta/nnunet_folds.csv
modal run modal/infer.py --step smoke          # 1 val scan x {cas_net, nnunet_f0}, then evaluated
modal run modal/infer.py --step run --methods cas_net,nnunet_f0 --split val
modal run modal/infer.py --step evaluate --methods cas_net --split test
```

Flags: `--tta` (mirror TTA, also writes `flips/` per flip). `--split` is one of `train`, `val`,
`test`, `valtest`, `all_nonexcluded`.

Outputs:

- `mask.nii.gz` (uint8) and `prob.nii.gz` (foreground probability). NIfTI-1 has no float16 type,
  so the values are rounded to float16 precision and stored as float32. The prob is cropped
  to the mask bbox +16 voxels when the full file exceeds 40 MB; the bbox is in `meta.json`.
- `meta.json`: weights path and sha256, repo commit, runtime, GPU, spacings, shape, tta, bbox.
- `flips/{none,flipX,flipY,flipXY}.nii.gz` when `--tta` (cas_net, swin_unetr, ffr_unet only).
- `/data/preds/_runs/<timestamp>.json` per `run`: counts and per-scan failures.
- `/data/eval/<method>_<split>.csv`: Dice, clDice, Betti-0/1 errors, centerline MD/HD95.

Idempotency: a scan is skipped when `meta.json` exists with the same `tta` flag and, if
probabilities are requested, they were saved. A run with a different `tta` recomputes it.

Deviations and caveats:

- CAS-Net's `prob.nii.gz` is sigmoid channel 1 (`prob_channel` in meta). The mask is the
  argmax of the 2 channels, as in the benchmark. The benchmark's own `--save-probs` keeps
  channel 0, which its comment flags as not a lumen probability.
- nnU-Net TTA is nnU-Net's built-in mirroring (all allowed axes), not the 4-flip scheme used
  for the others. It writes no `flips/`, so the TTA-disagreement feature is unavailable for nnU-Net.
- nnU-Net masks are nnU-Net's raw segmentation (no ImageCAS-X postprocessing), as in the
  benchmark's external `nnUNetv2_predict` path.
- The ImageCAS-X 0.5 mm resample cache is recomputed on the fly with the same `Resample` step.
- nnU-Net fold membership is reconstructed (see `nnunet_fold_of` in `infer.py`). It must be
  verified: held-out Dice for fold k should be lower than Dice on that model's training cases.
- Volumes are located by glob under `/data/imagecas/volumes`, so they can sit in any subfolder.
