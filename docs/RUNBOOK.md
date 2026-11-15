# CoronarySafe runbook

How each stage of the study was run. Commands are written from the scripts in this repository. Paths
are the droplet layout (`/root/coronarysafe` is the synced repository root, `/root/data` the data root,
`cs-infer` the Docker container). `<ip>` is the droplet address. Substitute your own values.

The run history (timings, costs and every failed job) lives in `docs/RUNLOG.md` on the droplet. That file
is not in the repository.

## 0. Environment

- Hardware: one AMD MI300X droplet with a scratch disk of about 5 TB. Set-up and cost are in
  [`infra/droplet/README.md`](../infra/droplet/README.md).
- Sync the repository from the laptop (excluding `.git`, `paper/`, caches and data):

  ```bash
  rsync -az -e "ssh -i ~/.ssh/<key>" --exclude .git --exclude __pycache__ --exclude '*.egg-info' \
    ./ root@<ip>:/root/coronarysafe/
  ```

- Copy the Kaggle API token to the droplet (`/root/.kaggle/kaggle.json`). It is needed only for staging.

## 1. Set up the container

```bash
bash /root/coronarysafe/infra/droplet/setup.sh 2>&1 | tee -a /root/setup.log
```

Starts `cs-infer` from the PyTorch image, installs the extra packages into `/root/venv-infer`, and ends with
a sanity check (`sanity: PASS`). `resume.sh` re-runs this step after a preemption.

## 2. Stage the data

```bash
bash /root/coronarysafe/infra/droplet/stage.sh 2>&1 | tee -a /root/stage.log
```

- Zenodo 23894668: ImageCAS-X labels and pretrained weights (`stage_zenodo.py`), byte- and md5-checked.
- Kaggle `xiaoweixumedicalai/imagecas`: five split groups in parallel (`stage_kaggle_group.py GROUP`).
- Done when 1000 CT volumes exist under `/root/data/imagecas/volumes`. The step is resumable.

## 3. Inference (segmenter predictions)

Smoke test first, one scan per method, with Dice against the reference:

```bash
bash /root/coronarysafe/infra/droplet/cs.sh infra/droplet/run_infer.py \
  --methods cas_net,swin_unetr,ffr_unet,nnunet_f0 --split test --limit 1 --evaluate --name smoke
```

Full queue (budget hard stop at `BUDGET_USD`, default 90):

```bash
cd /root/coronarysafe
BUDGET_USD=90 nohup bash infra/droplet/run_all.sh > /root/run_all.out 2>&1 &
```

The queue runs, in order: (a) `verify_folds` (16 train scans, five folds, probabilities saved);
(b) `nnunet_f0..f4` on test; (c) `cas_net`, `swin_unetr`, `ffr_unet` on val and test with mirror TTA;
(d) `nnunet_f0..f4` on val; (e) optional `nnunet_oof` on train and val, only with `RUN_OOF=1`.
Each job is idempotent and writes `/root/data/preds/<method>/<scan>/{mask.nii.gz,meta.json}`.
`DRY_RUN=1 bash infra/droplet/run_all.sh` checks every input without using the GPU.

Fetch the predictions and the run log to the laptop, then **destroy the droplet**:

```bash
bash infra/droplet/fetch.sh <ip>
```

## 4. Labels (failure taxonomy, per unit)

Labels are computed on the CPU from the prediction, the reference and the frozen taxonomy
(`configs/taxonomy_v3.yaml`), through the shared labelling core (`pipeline/label_core.py`):

```bash
bash /root/coronarysafe/infra/droplet/run_labels.sh cas_net,swin_unetr,ffr_unet,nnunet_oof test
bash /root/coronarysafe/infra/droplet/run_labels.sh cas_net,swin_unetr,ffr_unet,nnunet_oof val
```

Output: one JSON per unit under `/root/data/qa/v3/<method>/<scan>.json`, plus `report_<split>.txt`. The
per-unit record holds the critical flag, the burden, the Dice, the severity and the graph and uncertainty
features. Labelling uses up to 16 processes.

The S3 comparator (`nnunet_mv`, a majority vote of the five folds) is written by `m4/ensemble.py` and
labelled by `run_labels.sh nnunet_mv test` (see section 8).

## 5. Feature tables and dev scoring

- Patient-left direction for ostium features, fitted on train ground truth only:
  `python -m coronarysafe.m4.lr_convention fit --gt-dir <gt> --scans <train_scans.txt> --out <lr_convention.json>`
- M3 aorta prior (TotalSegmentator, aorta only, Apache-2.0):
  `bash /root/coronarysafe/infra/droplet/m4_aorta.sh <scan-list> [workers]` writes `/root/data/m4/aorta/<id>.nii.gz`.
- Assemble the dev table from the per-unit records:
  `python -m coronarysafe.m4.qa_table --qa-root /root/data/qa/v3 --methods cas_net swin_unetr --scans <val.txt> --out /root/data/m4/table_val.csv`  (CSV output of the table step)
- Dev scoring of the M4 model and the M3 ostium features, val only:
  `python -m coronarysafe.m4.score --root /root/data --model <model.pt> --aorta-dir /root/data/m4/aorta --out /root/data/m4/dev_units.json`
  (the `--unit-dir` and `--lr-json` arguments point at the unit records and the convention file).
- M6 proximal features (optional candidate, dev only): `python -m coronarysafe.m6.units` and `python -m coronarysafe.m6.sharp`.
- M5 evidence features and harm analysis: `python pipeline/m5_run.py features|harm|dev`.

## 6. M4 training (train split only)

```bash
# synthetic variants from TRAIN ground truth only; labels from the frozen taxonomy v3
python -m coronarysafe.m4.build --root /root/data --scans-file /root/data/m4/train_scans.txt \
  --out /root/data/m4/synth --variants 8 --workers 6
# plausibility CNN, early stopping on 20% of train patients
python -m coronarysafe.m4.train --data /root/data/m4/synth --out /root/data/m4/model
```

The model is trained only on train ground truth. The val and test scans are never read here.
Output: `model.pt`, `train_log.json` and the weights' sha256.

## 7. Dev comparison and freeze (Addendum H)

```bash
# --table: the v3 QA table of val units in parquet form (devcompare reads parquet)
python -m coronarysafe.m4.devcompare --table /root/data/m4/table_val.parquet \
  --units /root/data/m4/dev_units.json --m4-weights /root/data/m4/model/model.pt \
  --out-dir /root/data/m4/frozen
```

Candidates are compared by 5-fold patient-grouped cross-validation on val, using normalised recall at
20% and then AURC. The chosen model is written to `frozen_model.json`, `m4_model.pt` and
`lr_convention.json` in `/root/data/m4/frozen`. The sha256 of these three files is pinned in
`m4/test_eval.py` (`EXPECTED_SHA`) and in Addendum H. Every later step verifies those hashes before it runs.

## 8. Sealed test evaluation (Addendum H)

```bash
bash /root/coronarysafe/infra/droplet/run_test_eval.sh
```

This script runs, in order: the S3 ensemble comparators (`m4/ensemble.py`, label-free), the labels for
`nnunet_mv` on test, and the test evaluation (`python -m coronarysafe.m4.test_eval --split test --out /root/data/test_eval`).
The evaluation verifies the frozen bundle hashes at start-up and aborts on a mismatch. It produces S1 to S4,
the bootstrap for S2 (2,000 resamples, seed 0) and the per-unit feature and flag files.

Descriptive post hoc analyses (no refit) on the sealed outputs, through the module
`coronarysafe.analysis.posthoc`:

```bash
python -m coronarysafe.analysis.posthoc --in /root/data/test_eval --out /root/data/posthoc \
  --clinical /root/data/m5/clinical_test_vessel_rows.parquet --workers 6
```

`infra/droplet/run_posthoc.sh` is a wrapper that runs the same module from a fixed checkout path
(`/root/coronarysafe-m5`). Use it only after that path points at this repository.

## 9. Post-test sealed reads (addenda I to N)

The two reads are one-shot. Each is launched once by the coordinator, with a pre-launch manifest, and is
never repeated. Each has a val dry run that uses the same code path on val scans only.

- **Read 1** (Addendum M, reduced, and Addendum J):

  ```bash
  python -m coronarysafe.analysis.sealed_read1 --split val --out /root/data/addenda/read1_val
  python -m coronarysafe.analysis.sealed_read1 --split test --out /root/data/addenda/read1_test --manifest <manifest.json>
  ```

- **Read 2** (Addendum L test and Addendum K evaluation). Run after the K chain (section 10) and after the L code is fixed:

  ```bash
  python -m coronarysafe.analysis.sealed_read2 --split val --out /root/data/addenda/read2_val
  python -m coronarysafe.analysis.sealed_read2 --split test --out /root/data/addenda/read2_test --manifest <manifest.json>
  ```

Both reads refuse a non-empty output directory and abort if the computed manifest differs from the
pre-launch manifest. Their summaries are in `results/addenda/read1_test/` and `results/addenda/read2_test/`.

Val-only waves (never touch test), run by `infra/droplet/addenda/wave1_val.sh`:

- Addendum L phase 1 and phase 2: `coronarysafe.analysis.rule_stability dev-scores`, then
  `coronarysafe.analysis.burden_indep --split val` (phase 1, 160 dev units), then `coronarysafe.m4.valscore`
  (frozen-model scores on val for ffr_unet and nnunet_f0..f4), then `burden_indep` again on the 640 units.
- Addendum N: `coronarysafe.analysis.rule_stability n` (N1 rule stability on val; N2 in-sample ablation on the
  160 dev units).

Second-annotator check (Addendum I, negative feasibility, val only):
`src/coronarysafe/analysis/label_robust.py` (relabelling the original ImageCAS lumen against the ImageCAS-X
taxonomy) and `src/coronarysafe/analysis/label_robust_diag.py` (mask-mismatch diagnostics). Their
summaries are in `results/addenda/label_robust/`.

## 10. Acquisition-shift chain (Addendum K)

Perturbed test CTs are generated on the CPU, then the frozen CAS-Net is run on each perturbed set, then the
frozen per-unit labeller. The scripts are kept in `infra/droplet/addenda/` (the K code is in `infra/droplet/addenda/k_code/`).

1. Gate the regenerated volumes against the spec before any inference (`k_gate.py pre_regen ...`).
2. Generate the perturbed volumes and their per-scan parameters (`k_make.py noise contrast blur`), 8 workers.
   Output: `/root/data/imagecas/volumes_shift_K/<kind>/` and `/root/data/addenda/K/gen_params_<kind>.json`.
3. Gate again after generation (`k_gate.py post_regen ...`). The gate results go to `/root/data/addenda/K/spec_gate.json`.
4. Run the chain: `infra/droplet/addenda/k_chain.sh` (sequential) or `k_chain2.sh <noise_pid>` (contrast and
   blur in parallel). Each step runs `run_infer.py --methods cas_net@K_<kind> --split test --tta`, then
   `k_label.py cas_net@K_<kind> test`. Output: `preds/cas_net@K_<kind>/` and `qa/v3/cas_net@K_<kind>/`.
5. Read 2 (section 9) scores the K units with the frozen scorer.

**Known gap to check before re-running:** `pipeline/infer_core.py` reads a `<method>@<kind>` input from
`/root/data/imagecas/volumes_shift/<kind>/`, but `k_make.py` writes `volumes_shift_K/<kind>/`. The droplet run
needed a link or a copy between the two directories, which is not written down in the scripts. Confirm the
mapping (for example `volumes_shift/K_<kind>` pointing at `volumes_shift_K/<kind>`) before any re-run.

Shift-set generator used in the earlier pilot (CPU): `infra/droplet/make_shift.py` writes
`volumes_shift/<kind>` for `noise`, `zblur` and `contrast` at one fixed severity each.

## 11. Fetch and tear down

```bash
bash infra/droplet/fetch.sh <ip>      # predictions, meta, qa/, RUNLOG to the laptop, count checked
```

Then destroy the droplet in the provider console. Stopping it does not stop billing.

## Stage order at a glance

| Step | Entry point | Input | Output |
|---|---|---|---|
| Set-up | `infra/droplet/setup.sh` | container image | `cs-infer`, venv |
| Staging | `infra/droplet/stage.sh` | Zenodo, Kaggle | `/root/data/imagecas`, labels, weights |
| Inference | `infra/droplet/run_all.sh` | staged CTs, weights | `preds/<method>/` |
| Labels | `infra/droplet/run_labels.sh` | `preds/`, GT, `taxonomy_v3.yaml` | `qa/v3/` |
| Aorta prior | `infra/droplet/m4_aorta.sh` | CTs | `m4/aorta/` |
| M4 training | `coronarysafe.m4.build`, `m4.train` | train GT | `m4/model/` |
| Freeze | `coronarysafe.m4.devcompare` | val table and units | `m4/frozen/` (hashes pinned) |
| Sealed test | `infra/droplet/run_test_eval.sh` | frozen bundle, test preds | `test_eval/` |
| Sealed reads | `coronarysafe.analysis.sealed_read1`, `sealed_read2` | frozen bundle, test outputs | `addenda/read1_test/`, `addenda/read2_test/` |
| K chain | `infra/droplet/addenda/k_chain.sh` | test CTs | `volumes_shift_K/`, `preds/cas_net@K_*/`, `qa/v3/cas_net@K_*/` |
