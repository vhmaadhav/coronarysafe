# CoronarySafe — audited research plan (v2, 2026-11-13)

**Question.** Using only what a deployed system can see (the CT, one segmenter's output and its probabilities), can we rank coronary CTA reconstructions so that a reviewer who checks 20% of scans catches most of the *anatomically severe* failures, and does this hold for a segmenter the QA model has never seen?

This is a QA / review-triage study built on top of existing segmenters. It is not a new segmentation model, not clinically validated, and not a planning tool.

---

## 0. What the audit changed (v1 → v2)

| # | v1 claim / gap | Finding | v2 change |
|---|---|---|---|
| A1 | 23 h of MI300X at $2.59/h | The MI300X credit goes to PathVG and expires 2026-11-16. The rate is also wrong ($1.99/h). | **All GPU work on Kaggle.** The ImageCAS CT volumes are already a Kaggle dataset (`xiaoweixumedicalai/imagecas`), so nothing big is downloaded to the laptop. |
| A2 | "ImageCAS-X v3 ships 800 CT cases" | Zenodo 23894668 (CC BY 4.0, open) holds **labels, centerlines, meshes and weights only** (0.9 GB + 1.6 GB). The CT volumes come from the Kaggle ImageCAS mirror. | Data = Kaggle ImageCAS volumes + Zenodo ImageCAS-X labels/weights. Code is MIT. |
| A3 | "Run CAS-Net + another baseline" | The weights zip covers 8 methods: CAS-Net, FFR-UNet, Swin UNETR, ImageCAS (5-model cascade), ADE-HTL, nnU-Net, nnU-Net+clDice, TotalSegmentator. ADE-HTL and TotalSegmentator `coronary_arteries` need a TotalSegmentator **non-commercial licence**. | Start with 3 cheap single-step models (nnU-Net, CAS-Net, Swin UNETR). Add the others only if Gate G0 needs more failures or more segmenter diversity. |
| A4 | **Where do QA training labels come from?** (missing in v1) | The pretrained segmenters were trained on the train split, so their train-split predictions are in-sample and nearly perfect. Only the **val + test** predictions show real failure behaviour. | The QA model is trained and evaluated on val+test predictions using **patient-grouped nested CV** and **leave-one-segmenter-out**. It stays small: logistic regression or GBM on about 30 features. Synthetic failures may be used to augment training only, never for evaluation. |
| A5 | "Severe failure" never defined | Without a fixed definition before results are seen, the headline metric can be tuned after the fact. | Severity is **pre-registered** (§3) from ImageCAS-X segment labels plus the benchmark's per-point local Dice along the GT centerline. It is frozen before any QA model is built. |
| A6 | Strongest baseline missing | Per Zenk et al. (MedIA 2025), pairwise Dice between ensemble members is the strongest and most robust failure detector. We already have 8 segmenters. | Two settings are reported separately: **single-model deployment** (main claim) and **ensemble available** (Zenk-style cross-model disagreement as the bar to beat). |
| A7 | "≥60% recall at 20% budget" | Recall at a fixed budget depends on the failure base rate. If failures are 30% of scans, even an oracle reviewing 20% of scans catches only 67%. | The primary metric is **normalised recall@20% = recall / oracle recall**. AURC/E-AURC, AUROC and Spearman are secondary. The bar is defined against the strongest baseline, not as an absolute number. |
| A8 | Failures may be too rare | Strong 2023–2026 segmenters reach Dice ≈ 0.9. Severe failures may cluster in a few poor-quality scans. | **Gate G0** counts failures before anything else is built. Fallbacks, in order: more segmenters → a realistic shift set (noise / low contrast / z-motion on test volumes) → pivot. |
| A9 | ASOCA as external validation | It has 40 cases with lumen masks only (no branch labels or centerlines), a different vessel-extent protocol, and its licence and access status are unclear. | ASOCA is a **descriptive stress test**. Only connectivity and leak failures are used there (segment-level failures can't be defined). Wide CIs are expected and reported as such. Access must be checked first. |
| A10 | Novelty | ImageCAS-X stratifies accuracy but does no failure prediction. The closest work is Zenk 2025 (generic), Robinson 2019 / RCA (cardiac MR) and RepAir 2025 (airway reconnection repair). | The contribution is narrowed to (i) a clinically weighted coronary failure taxonomy calibrated to inter-observer variability, (ii) label-free anatomical graph priors, (iii) **cross-segmenter generalisation**, and (iv) **segment-level localisation** of the failure. |
| A11 | "Paper draft" listed as a deliverable | — | No paper scaffolding until results pass Gate G2. Then propose and wait. |
| A12 | 3D viewer | Scope creep if done early. | Built last: a static three.js/vtk.js page for about 10 curated failure cases. |

### Update 2026-11-13 (after staging)

- **Compute:** Maadhav approved the MI300X credit (shared with PathVG, expires 2026-11-16), Modal (standing OK for this project) and Kaggle. Modal is primary: Volume `coronarysafe-data` plus GPU inference.
- **Splits:** train 560 / val 80 / test 160 / exclude 200. `Descriptors.xlsx` covers all 1000 scans, including the excluded ones.
- **Weights:** 6 methods: CAS-Net, Swin UNETR, FFR-UNet, ImageCAS cascade, ADE-HTL, and **nnU-Net 3d_fullres with all 5 CV folds** (numTraining 640 = train + val). nnU-Net + clDice and TotalSegmentator are not in the zip.
- **This resolves A4.** Each nnU-Net fold never saw ~128 of the 640 train+val cases, so we get **~640 out-of-fold nnU-Net predictions** as QA training data, plus a native 5-member ensemble for the Zenk pairwise-Dice baseline. The fold assignment is reconstructed from nnU-Net's default split and must be verified empirically (out-of-fold Dice < in-fold Dice).
- **Licence:** the ImageCAS volumes on Kaggle are **CC BY-NC 4.0**, so this is research use only.

---

## 1. Data

| Role | Source | Access | Use |
|---|---|---|---|
| CT volumes | Kaggle `xiaoweixumedicalai/imagecas` (ImageCAS, 1000 scans) | Kaggle input mount | All inference |
| Labels | ImageCAS-X v3, Zenodo 23894668 | Open, CC BY 4.0 | 14-segment lumen masks, VTK centerlines, meshes, `filelist/{train,val,test,exclude}.txt`, `Descriptors.xlsx` (image quality, dominance, disease) |
| Weights + code | Zenodo weights zip + github.com/kitbransby/ImageCAS-X (MIT) | Open | Inference + `evaluate.py` + `precompute_centerline_samples` (per-point local Dice by segment) |
| Aorta prior | TotalSegmentator `total` task, aorta class (open, Apache-2.0) | pip | Ostium detection feature |
| External | ASOCA, 40 annotated cases | UK Data Service; licence **UNVERIFIED** | Descriptive stress test only |
| Optional | ImageALCAPA (30 CTA, BIBM 2022) | Access **UNVERIFIED** | Anomalous-origin case study only |

**Open data questions (answer in Phase 0, from the downloaded files):** split sizes; why 200 ImageCAS scans are excluded (if they are poor quality, they are a natural hard set with original ImageCAS labels only); whether second-reader (inter-observer) labels are released.

## 2. Pipeline

```
CT ─▶ segmenter k ─▶ mask + prob map ─┬─▶ uncertainty features (entropy / max-prob aggregated over vessel-skeleton neighbourhood, TTA flip disagreement)
                                      ├─▶ graph features (skeletonise → vessel graph: #components, #ostia touching aorta, total/left/right length,
                                      │    endpoint count, branch depth, radius tapering violations, tortuosity, gaps between collinear endpoints,
                                      │    leaks: blobs with radius ≫ tree prior, distance from heart)
                                      └─▶ anatomical prior features (z-scores vs train-split GT statistics: length per side, #branches, dominance-consistent shape)
                                                   ▼
                                       QA scorer (LR / GBM) ─▶ scan risk + per-region flags
```
Every feature is computable **without GT**. A unit test enforces this: the feature code cannot import the label paths.

## 3. Pre-registered failure taxonomy (freeze before Phase 3)

Unit = (scan, segmenter). Major segments = LM, LAD, LCx, RCA, plus the PDA on the dominant side.

- **F1 Missed / broken segment:** a contiguous run of ≥ 10 mm along a GT centerline (root→leaf, crossing junctions) where no predicted lumen lies within max(1.5 mm, local GT radius). *(Changed 2026-11-13, before any real prediction was seen. The 8 mm-cube local-Dice rule smears a 15 mm gap down to ~9 mm and misses it on phantoms. Local Dice < 0.3 runs are kept as a separate "F1b degraded" flag that doesn't count toward severity.)*
- **F2 Disconnection:** a major tree split into ≥ 2 predicted components, each ≥ 10 mm of skeleton.
- **F3 False connection / leak:** ≥ 10 mm of predicted skeleton more than 2 mm from any GT lumen, or a bridge joining left and right trees or joining a coronary to a non-coronary structure.
- **Severe** = any F1–F3 on a major segment. **Minor** = the same on side branches.
- **Calibration:** thresholds must exceed inter-observer disagreement if second-reader labels exist. Otherwise, a sensitivity analysis is reported at 5/10/20 mm and Dice 0.2/0.3/0.5.

## 4. Experiments

**E1 Baselines (single-model setting):** random; oracle; Betti-0 / component count; skeleton gap count; volume z-score; mean entropy; skeleton-weighted entropy; TTA disagreement; a learned Dice regressor (Robinson-style, small 3D CNN on mask+CT crop) only if E3 beats every other baseline.
**E2 Ensemble setting:** Zenk pairwise-Dice across the other segmenters.
**E3 Proposed:** graph + prior + uncertainty features → LR/GBM. Ablations by feature group.
**E4 Generalisation:** leave-one-segmenter-out; stratification by image quality, disease and dominance from `Descriptors.xlsx`; shift set; ASOCA.
**E5 Localisation:** for flagged scans, the share where the top-ranked region overlaps the true failing segment.

**Statistics:** patient-level bootstrap (2,000 resamples, clustered over segmenters). Paired bootstrap of Δ(normalised recall@20%) and ΔAURC against the strongest baseline. Folds and splits are fixed, seeds logged.

## 5. Gates

- **G0 — is the problem measurable?** (≈3 Kaggle GPU-h, week 1.) Run 3 segmenters on val+test and count severe failures. **Proceed** if there are ≥ 30 severe units across ≥ 15 distinct patients. Otherwise add the remaining segmenters and the shift set, then recount. If still short, **stop** and pivot (pulmonary A/V on AirRC/HiPaS/PARSE, or write up a short negative note).
- **G1 — does structure add anything?** If E3 doesn't beat the best E1 baseline with a paired-bootstrap CI excluding 0 in normalised recall@20% or AURC, the study becomes a benchmark/negative report. There is no "bigger model" escalation unless a specific failure mode demands it.
- **G2 — publishable?** G1 passes, the result holds on leave-one-segmenter-out and at least trends positive on shift/ASOCA, and the numbers have been re-verified. Only then propose a write-up (a MICCAI 2027 UNSURE-type workshop; check the CFP and dates when the time comes).

## 6. Compute and storage

- **Kaggle GPU (T4/P100):** inference for about 3–8 segmenters × (val+test) scans, plus 4-flip TTA for the single-model uncertainty features. Estimated at 10–30 GPU-h total, within the weekly quota. No MI300X, no Modal.
- **Laptop:** graph features and QA training only on small per-case feature tables. Check CPU temperature first (`sensors | grep Tctl`).
- **Kaggle outputs:** uint8 masks (`.nii.gz`) + per-case feature parquet + downsampled float16 entropy maps. No full probability volumes. Outputs are mirrored to a private HF dataset if they exceed Kaggle output limits.

## 7. Phase order

0. Repo + Kaggle kernel skeleton; download weights into a Kaggle dataset; read the filelists and descriptors → **G0**.
1. Reproduce benchmark metrics for the 3 segmenters on test (within benchmark ± SD) → sanity check.
2. Freeze the taxonomy (§3) and compute failure labels.
3. E1/E2 baselines.
4. E3 + ablations + E5 → **G1**.
5. E4 generalisation → **G2**.
6. Viewer + report.

## 8. Known risks

- Single-architecture segmenters may share failure modes (e.g. all miss distal RCA in poor scans). Leave-one-segmenter-out results may then look better than true novel-model generalisation. TotalSegmentator is the most different segmenter (if licensed).
- Ostium detection depends on a correct aorta mask. TotalSegmentator failures propagate into the features, so its failure rate must be measured.
- Probability-map formats differ across methods: sigmoid vs softmax, and the ImageCAS cascade averages several outputs. Uncertainty features are normalised per method.
- The ImageCAS Kaggle mirror is third-party. Check its scan IDs and header geometry against the ImageCAS-X labels before any metric is trusted.

---

## G0 result (2026-11-14): PASS

Test split, CAS-Net + Swin UNETR (320 units, 160 patients). Taxonomy v3 was frozen before any QA scoring.
- Reproduction sanity: CAS-Net Dice 0.909 (published 0.912 ± 0.028); Swin UNETR 0.877.
- v2 "severe" = 206/320 (64%), mostly distal (R-PDA/R-PLA, distal LCx/LAD), with weak link to image quality. Too common to triage on, so it is kept as a secondary target only.
- **v3 "critical"** = an F1/F2 event on LM/LAD/LCx/RCA ≤ 50 mm from the ostium or at GT radius ≥ 1.25 mm, or a leak/bridge, or a thick detached false positive. It covers **80/320 units (25%) in 52 patients**: CAS-Net 21%, Swin 29%; diseased 33% vs non-diseased 18%; the two methods agree on 28 scans and either flags 52. Variants range 56–89 units (run 20 mm → 56; 60 mm → 89), so the count is robust.
- With a 25% base rate, the oracle catches 80% of critical units at a 20% review budget. That leaves room for a meaningful normalised-recall target.
- Continuous target: `burden` (missed major length × r²), median 3.5, 39% zero; `missed_major_mm` p75 13 mm.
- **Open:** the nnU-Net fold reconstruction is NOT verified (out-of-fold was lowest-Dice in only 5/16 scans). Out-of-fold training data is on hold until a confidence-based check passes.
- **Primary target from here on: critical @ 20% budget (normalised recall); secondary: severe, burden (Spearman/AURC).**
