# CoronarySafe

Label-free quality control for AI coronary-artery reconstructions from coronary CT angiography (CCTA).
CoronarySafe ranks reconstructions so that a reviewer who can check a fixed 20% of scans catches the
anatomically critical failures (missed or broken proximal LM, LAD, LCx and RCA segments, disconnections,
leaks and bridges), including failures that overlap scores such as Dice hide. This is a research prototype.
It has not been clinically validated.

## Headline results

Primary test: ImageCAS-X test split (160 patients), evaluated once with a model frozen beforehand. The
primary segmenter is FFR-UNet, which the QA model never saw. Numbers are copied from
[`results/RESULTS.md`](results/RESULTS.md) and its post-test addenda section.

| Check | Result | Bar | Verdict |
|---|---|---|---|
| S1: normalised recall of critical failures at a 20% review budget (FFR-UNet) | 0.906 | >= 0.75 | pass |
| S2: gain over the best pre-specified simple rule (component count >= 10 mm) | +12.5 pp [-0.3, +23.9] | >= +10 pp, CI > 0 | **fail** (narrow) |
| S3: single model vs 5-fold nnU-Net ensemble agreement | +0.41 [0.16, 0.56] | non-inferior | pass |
| Addendum M (disclosure): critical units caught at 32 of 160 reviews | 29 (model) vs 25 (rule) of 64 | n/a | descriptive |
| Addendum J: replication on nnU-Net folds f0-f4, same 160 patients | +11.4 pp [+3.0, +20.7] | >= +10 pp, CI > 0 | pass, with caveats |

Post-test addenda (exploratory or secondary; none changes S1 to S3, and S2 remains a fail):

- **K, acquisition-shift stress test (CAS-Net on perturbed test CTs).** Model edge over the rule:
  noise +0.119 [-0.023, +0.275]; contrast (x0.75) +0.078 [-0.079, +0.195]; blur (0.6 mm) +0.020
  [-0.135, +0.131]. The edge is preserved under noise, shrinks under contrast and vanishes under blur.
  All model inputs are mask-derived, so this reflects a shift in the segmenter's failure profile: the
  learned combination's edge over a single topology rule is not robust to acquisition shift.
- **L, independent-lumen burden.** Pooled Spearman correlation between score and the independent burden
  0.26 [0.13, 0.37]; top-20% minus rest +0.047 [0.025, 0.065]. The burden is dominated by a reference
  thickness mismatch, so the pre-specified AUROC is not estimable. Treat as weak support against pure
  label circularity.
- **I, second annotator (negative feasibility).** The original ImageCAS lumen masks are 2-6 times larger
  than the ImageCAS-X lumen labels (median Dice 0.43). They cannot test label robustness, so the
  intended test run was not performed.
- **N, rule stability and feature redundancy (dev and val only).** Dropping any single one of the 22
  inputs changes dev NR@20 by at most 0.05. The features are redundant.

Full tables, confidence intervals and the caveats for every addendum are in `results/`.

## Pre-registration trail

1. [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md): study design and feasibility gate.
2. [`docs/ANALYSIS_PLAN_v1.md`](docs/ANALYSIS_PLAN_v1.md): pre-registered analysis plan with dated
   addenda A to H. The model freeze is Addendum H, made before the sealed test.
3. Sealed test evaluation (`results/RESULTS.md`, Addendum H bundle hash verified at start-up).
4. [`docs/ADDENDA_FROZEN_2026-11-15.md`](docs/ADDENDA_FROZEN_2026-11-15.md): post-test addenda I to N,
   including the errata (1 to 10) and the two one-shot sealed reads (`results/addenda/`).

Frozen documents are not edited. Corrections are recorded as errata in the addenda.

## Repository layout

```
.
├── README.md
├── pyproject.toml
├── configs/
│   ├── taxonomy_v3.yaml        failure taxonomy used by the frozen pipeline
│   └── archive/                taxonomy v1 and v2 (superseded)
├── docs/
│   ├── ANALYSIS_PLAN_v1.md     pre-registered plan with addenda A to H (frozen)
│   ├── ADDENDA_FROZEN_2026-11-15.md   post-test addenda I to N (frozen)
│   ├── RESEARCH_PLAN.md
│   ├── RESEARCH_REPORT.md
│   └── RUNBOOK.md              how each stage was run
├── src/coronarysafe/
│   ├── labels/                 failure taxonomy
│   ├── features/               graph, uncertainty, ostium, evidence and proximal features
│   ├── eval/                   metrics, bootstrap, conformal, triage
│   ├── m4/                     M4 plausibility model, frozen scoring and sealed test
│   ├── m6/                     proximal-connectivity candidate
│   ├── analysis/               sealed reads, label robustness, burden, rule stability, post hoc
│   └── *.py                    graph, geometry, I/O
├── pipeline/                   inference core, labelling core, M5 and CFD runners
├── infra/
│   ├── droplet/                AMD MI300X setup, staging, inference queue, labelling
│   │   └── addenda/            scripts used for addenda K and L/N (summaries of the runs)
│   ├── kaggle/                 CPU labelling bundle for Kaggle
│   └── modal/                  alternative GPU backend used during development
├── tests/                      unit tests on synthetic phantoms
├── viewer/                     offline 3D failure viewer
└── results/
    ├── RESULTS.md              sealed test results and post-test addenda
    ├── POSTHOC.md              descriptive post hoc analyses
    ├── results.json
    ├── figures/
    └── addenda/                aggregate summaries of addenda I to N (no per-scan data)
```

## Data

This repository contains **no images, masks, labels or model weights**. Obtain them from the original sources:

- ImageCAS CT volumes: [Kaggle, xiaoweixumedicalai/imagecas](https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas),
  CC BY-NC 4.0, non-commercial use only.
- ImageCAS-X labels and pretrained segmenter weights: [Zenodo 23894668](https://zenodo.org/records/23894668)
  (paper: arXiv 2608.30404), CC BY 4.0. Benchmark code at [kitbransby/ImageCAS-X](https://github.com/kitbransby/ImageCAS-X).

Per-scan intermediate outputs (predictions, per-unit labels, perturbed volumes) are not included.

## Reproducing

See [`docs/RUNBOOK.md`](docs/RUNBOOK.md) for the stage-by-stage commands: staging, inference, labels,
M4 training, freeze, sealed test, the two sealed reads and the acquisition-shift chain.

Unit tests (synthetic phantoms, no data needed):

```bash
pip install -e ".[dev]"
python -m pytest -q
```

## Status and limitations

- A single source dataset (ImageCAS-X, 160 test patients). There is no external test cohort yet.
- No reader study. The critical-failure labels are rule-based and not adjudicated by clinicians.
- Label circularity is only partly addressed. The failure labels are computed against the ImageCAS-X
  reference masks, so they are not independent of that reference. The second-reference check (Addendum I)
  was not feasible with the original ImageCAS masks.
- The primary result (S1) passes, but the gain over the simple rule (S2) falls just short of its bar.
- The QA model was fitted on the validation split for two segmenters (cas_net, swin_unetr). The primary
  test uses FFR-UNet, which was held out from fitting. Results on the other segmenters are secondary.
- The acquisition-shift results show the learned component is fragile to changes in noise, contrast and blur.
- Not a medical device. Not for clinical decisions.

## Paper

Preprint: link to be added.

## Citation

Citation details to be added on publication.

## Licence

To be decided. No licence is granted until one is chosen.
