# CoronarySafe

Label-free quality control for AI coronary-artery reconstructions from CT angiography. CoronarySafe ranks reconstructions so that reviewers checking a fixed fraction of scans catch the **anatomically critical failures** (missed or broken proximal LM/LAD/LCx/RCA segments, disconnections, leaks and bridges), including failures that overlap scores such as Dice hide. It is a research prototype and has not been clinically validated.

## Result (sealed, pre-registered test)

Evaluated once on the ImageCAS-X test split (160 patients) with a model frozen beforehand (see `docs/ANALYSIS_PLAN_v1.md`, Addendum H). The primary test uses **FFR-UNet as a segmenter the QA model never saw**.

| Criterion | Result | Bar | Verdict |
|---|---|---|---|
| S1: normalised recall of critical failures at a 20% review budget | **0.906** | ≥ 0.75 | pass |
| S2: gain over the best pre-specified simple rule | +12.5 pp [−0.3, +23.9] (AUROC +0.064 [0.007, 0.119]) | ≥ +10 pp, CI > 0 | narrow fail |
| S3: single model vs 5-fold nnU-Net ensemble agreement | +0.41 [0.16, 0.56] | non-inferior | pass |

The model beats the rule on all four reconstruction sets and in all three leave-one-segmenter-out checks. Units it flags carry most of the downstream harm: severe-stenosis category flips occur in 2.1% of flagged units vs 0.3% of unflagged ones, and coverage loss in 17% vs 9%. The full tables and figures are in `results/` (`RESULTS.md`, `POSTHOC.md`, `figures/`).

We also report negative results: a 3D CNN trained on synthetic corruptions does not transfer to real failures, conformal risk control certifies nothing at n = 160, and voxel-derived 1D FFR is unstable under sub-voxel shifts (95th percentile |ΔFFR| ≈ 0.3).

## Repository

| Path | Contents |
|---|---|
| `src/coronarysafe/` | Failure taxonomy (`labels/`), graph, uncertainty, ostium, evidence and proximal features (`features/`), the QA models and sealed evaluation (`m4/`, `m6/`), metrics, bootstrap and conformal (`eval/`), clinical and harm analyses (`analysis/`) |
| `configs/` | Failure taxonomy versions; v3 is the frozen one |
| `pipeline/` | Inference core, labelling and the CFD verification runners |
| `droplet/` | Setup, staging, inference queue and evaluation scripts for an AMD MI300X (ROCm) machine |
| `modal/`, `kaggle/` | Alternative compute backends used during development |
| `docs/ANALYSIS_PLAN_v1.md` | Pre-registered analysis plan with dated addenda A–H (the freeze is in H) |
| `RESEARCH_PLAN.md` | Study design and the G0 feasibility gate |
| `results/` | Sealed test results, post-hoc analyses and figures |
| `tests/` | Unit tests (synthetic phantoms) |

```bash
pip install -e ".[dev]"
python -m pytest -q
```

## Data and licences

This repository contains **no images, labels or model weights**. Obtain them from the original sources:

- ImageCAS CT volumes: [Kaggle](https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas), CC BY-NC 4.0, non-commercial use only.
- ImageCAS-X labels and pretrained segmenter weights: [Zenodo 23894668](https://zenodo.org/records/23894668), CC BY 4.0. Benchmark code at [kitbransby/ImageCAS-X](https://github.com/kitbransby/ImageCAS-X) (MIT).

Code in this repository: MIT.
