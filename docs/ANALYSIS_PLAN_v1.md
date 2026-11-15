# CoronarySafe pre-registered analysis plan v1

Frozen 2026-11-14, before any FFR-UNet test-set labels or features are inspected. Changes after this point need a new version with a dated reason. Results that deviate are reported as exploratory.

## Data and units
- Taxonomy: configs/taxonomy_v3.yaml (frozen 2026-11-14). Unit = (segmenter, patient).
- **Development (QA fitting/selection):** ImageCAS-X val, 80 patients × {CAS-Net, Swin UNETR} reconstructions, plus synthetic corruptions of the 560 TRAIN ground-truth trees. No test patients, and no FFR-UNet output of any split.
- **Primary evaluation:** FFR-UNet reconstructions of the 160 ImageCAS-X test patients. FFR-UNet is never seen by the QA model.
- **Secondary evaluation (pre-specified):** leave-one-segmenter-out on test over {CAS-Net, Swin UNETR, FFR-UNet}, 480 units. For each held-out segmenter, the QA model is fit on dev data of the other two only.
- nnU-Net folds are used only as the ensemble comparator, never for QA fitting (their train/val membership is unverified).

## Endpoints
- **Primary:** recall of v3-critical units at a 20% per-segmenter review budget (32 of 160), reported as actual recall, oracle recall, normalised recall (actual/oracle), prevalence and chance recall.
- **Secondary:** AURC with burden as the loss; Spearman with burden; AUROC for critical; recall of critical units with Dice ≥ 0.90; v2-severe recall at 20%; segment localisation hit rate (top-flagged segment overlaps a true critical event).

## Comparators
1. Random (expected = budget).
2. Best simple rule. Candidates: n_components ≥10 mm, n_gaps, total-length z-score, max-radius z (leak). **Selected on dev only.**
3. Single-model uncertainty: TTA disagreement, skeleton-weighted entropy (selected on dev).
4. Ensemble (Zenk-style): mean pairwise Dice across nnunet_f0..f4 on test, ranking the majority-vote nnU-Net reconstruction (its own v3 labels). Cross-architecture disagreement is reported separately.

## Candidate QA models (fit on dev; final choice made on dev by normalised recall + AURC with 5-fold patient-grouped CV)
- M1: L2 logistic on graph + prior z-score features.
- M2: M1 + uncertainty features.
- M3: M2 + ostium/aorta features (TotalSegmentator `total` task aorta, Apache-2.0).
- M4: anatomical plausibility model trained on synthetic corruptions of the train GT trees (graph/point-cloud or 3D CNN), with its score used alone and stacked into M3.
Exactly ONE model is frozen (weights + preprocessing hash) before the test evaluation runs.

## Success criteria (strong win)
- S1: primary normalised recall ≥ 0.75 on FFR-UNet test.
- S2: (frozen model − best simple rule) normalised recall ≥ +10 pp, with a paired patient-bootstrap 95% CI (2000 resamples) excluding 0.
- S3: the frozen single-model QA is non-inferior to the 5-fold nnU-Net ensemble comparator (difference ≥ −5 pp normalised recall, CI reported).
- Story: S4, recall of Dice ≥ 0.90 critical units ≥ 50%; or S5, localisation hit rate ≥ 70%.
All criteria are reported whether met or not. Secondary LOSO results are reported alongside and are not substituted for the primary.

## Integrity
- The test split is evaluated once with the frozen model. Any re-run after a bug fix is logged in docs/RUNLOG.md with the reason and both results.
- The code and config hash of the frozen model go into results/triage_v1.json.

## Addendum A (2026-11-14, before any test evaluation): stress-test responses
- **Label/feature circularity.** v3 labels use topology, and so do the graph features. So every result is also reported against `missed_major_mm` (non-topological; Spearman and recall of its top 25%) and within the Dice ≥ 0.90 subset.
- **Ensemble comparator setting.** S3 compares rankings of DIFFERENT reconstructions (FFR-UNet vs the nnU-Net majority vote). It is reported as a separate deployment setting, with prevalence and oracle shown for each.
- **Shift stress set** (noise, z-blur, low contrast on test CTs). Exploratory, evaluated only with the frozen model, never used for selection.
- **Possible amendment v2 (dev data only).** If the nnU-Net fold assignment is verified on all 640 train+val cases (held-out fold has the worst confidence loss in ≥ 90% of cases, permutation p < 0.001), the out-of-fold nnU-Net predictions MAY be added to the DEVELOPMENT set via a dated v2 before freezing. The test protocol is unchanged.

## Addendum B (2026-11-14, before any test evaluation): additional candidate components
- **M5 image-evidence features (label-free).** For each predicted-skeleton endpoint pair or gap (from the n_gaps logic), take the minimal-cost path through the CT between the endpoints, with cost = inverse vesselness (multi-scale Frangi/Sato). Features along it: path length, the mean/min of normalised HU (scan-normalised by the aorta lumen HU median from TotalSegmentator), mean vesselness, and the fraction of the path above a contrast threshold. The scan-level score is the max over gaps, and a weighted sum over gaps near major-vessel endpoints. The same features are computed along each predicted tree's distal tips, to catch missed continuations. M5 joins the candidate list. Selection stays dev-only.
- **Conformal risk control (secondary analysis).** On dev, calibrate a score threshold so that the expected miss rate of critical units among non-flagged units is ≤ α (α = 0.10, 0.05), via split-conformal/CRC with patient-level grouping. On test, report the empirical miss rate and review fraction. Report it explicitly as a guarantee under exchangeability, which an unseen segmenter violates, so the coverage observed on test is the result.
- **Downstream-harm analysis (descriptive).** For LM/LAD/LCx/RCA on GT vs predicted reconstructions: centerline length, minimum lumen diameter and a Poiseuille resistance proxy (Σ 8μL/(πr⁴) along the vessel). Compare the error distributions for critical vs non-critical units, and within Dice ≥ 0.90. Label this as a proxy, not FFR.

## Addendum C (2026-11-14, before any test evaluation): reconstruction-induced haemodynamic error (1D CFD)
- **Purpose (descriptive, secondary).** Quantify how segmentation failures change a clinically used haemodynamic quantity. This is NOT a validation against invasive FFR, because ImageCAS has no measured FFR.
- **Model.** A reduced-order steady hyperaemic model on the rooted coronary tree (skeleton + EDT radius, rooted at ostia from the TotalSegmentator aorta).
  - Primary solver: svOneDSolver (SimVascular) if it builds on the droplet. Fallback (pre-specified): a lumped network with Poiseuille segment resistance plus a Young–Tsai stenosis term (viscous + expansion loss) at local radius minima.
  - Inlet aortic pressure 93 mmHg. Total hyperaemic flow scaled by tree volume (allometric, exponent 0.75) and split at bifurcations by Murray's law (r³). Blood μ = 3.5 mPa·s, ρ = 1060 kg/m³.
  - Output: FFR_sim = P_distal/P_aorta at the distal end of LAD, LCx and RCA (on segments ≥ 10 mm), and the minimum along each vessel.
- **Comparison.** The same model with identical parameters on the GT tree vs each reconstruction. Endpoints:
  - |ΔFFR_sim| per major vessel;
  - "decision flip" = GT and reconstruction fall on opposite sides of 0.80;
  - "unmeasurable" = vessel distal end disconnected from any ostium in the reconstruction.
  - Reported for v3-critical vs non-critical, within Dice ≥ 0.90, and for the QA-flagged (top 20%) vs unflagged sets with the frozen model on test.
- **Verification before use.** Analytic check on a straight tube (Poiseuille ΔP within 1%); a synthetic stenosis series (FFR decreasing monotonically with % diameter stenosis); mesh-free convergence vs skeleton step size; GT-vs-GT test-retest on a resampled mask (|ΔFFR| < 0.01).
- **Integrity.** No QA-model selection uses these outputs. They are computed after the model freeze, for test units.

## Addendum D (2026-11-14, before any CFD result on val/test and before any test evaluation): CFD verification revision
- **Observed (reported in full).** With the pre-specified test-retest (nearest-neighbour 0.45 mm round trip), the coupled fallback solver gave |ΔFFR| < 0.01 in 41% of 58 vessels (median 0.016, p95 0.126), failing the ≥ 90% criterion. The diagnosed cause is that nearest-neighbour resampling of binary masks systematically shrinks lumen volume by 8–10% (resistance ∝ r⁻⁴), so the perturbation changes the geometry, not just the discretisation. Tube, stenosis-series and element-length checks pass.
- **Revisions.**
  - (i) Radius estimator: sub-voxel effective radius r = √(A/π), from the cross-sectional lumen area on a plane orthogonal to the smoothed centerline (partial-volume aware, using the linear-interpolated mask). The EDT is kept only as a fallback.
  - (ii) Geometry-preserving test-retest: a linear-interpolation resample of the mask as float, thresholded at 0.5, plus ±0.5-voxel rigid sub-voxel shifts. Both the old and the new test-retest are reported.
- **Noise floor.** The achieved test-retest |ΔFFR| distribution (p95) defines a measurement noise floor. Only reconstruction-induced |ΔFFR| above the noise-floor p95 counts as a meaningful error, and a decision flip counts only when both FFR values are more than the noise-floor p95 away from 0.80. If (i)+(ii) achieve ≥ 90% < 0.01, the original criterion is met; otherwise the noise floor is reported and used as stated.
- **Cross-check.** svOneDSolver vs the fallback on 20 GT trees; |ΔFFR| between solvers is reported.

## Addendum E (2026-11-14, before any CFD result on val/test): centerline robustness, final attempt
- **Observed in pass 4 (K = 7.89, sub-voxel radius).** Tube, stenosis-series and convergence PASS. Linear round-trip median |ΔFFR| 0.009 (p95 0.097). ±0.5-voxel shifts: median 0.072, p95 0.334. Pooled noise-floor p95 = 0.32. That is far too large for decision-level claims. Cause: voxel-skeleton topology instability.
- **Final attempt.** Replace voxel skeletonisation with TEASAR centerlines (kimimaro), with spline-smoothed paths and radius from orthogonal cross-sections (as in Addendum D). Use the provided VTK centerlines for GT trees, mapped to the same pipeline. Recalibrate K so the median LAD reference flow is 3.0 mL/s.
- **Gate.** If the pooled shift + round-trip p95 |ΔFFR| ≤ 0.05, run the CFD analysis as planned, using that noise floor. Otherwise the CFD component is reported ONLY as a negative methods finding ("voxel-derived 1D FFR is unstable to sub-voxel perturbations at ImageCAS resolution"), and no ΔFFR or decision-flip results are claimed. The length and volume deficits of major vessels (from harm.py) remain as the descriptive harm measures.
- **Addendum E clarification (2026-11-14, before any CFD result on val/test).** The pass-5 gate (pooled p95 0.312) was run on TRAIN GT trees, which have no aorta masks, so every tree used fallback rooting (largest-radius endpoint). That root can jump under sub-voxel shifts, which confounds the test. The method as specified roots at aorta-touching ostia. The gate is therefore evaluated once more, unchanged in every other respect (TEASAR, K recalibrated on the same trees), on 20 VAL GT trees WITH aorta rooting. Val GT is dev data, and no predictions or ΔFFR are involved. Both results are reported. If p95 ≤ 0.05 → proceed. Otherwise the negative finding stands, with no further attempts.

## Addendum F (2026-11-14, before the freeze and before any test evaluation): M6 proximal-connectivity candidate
- **Rationale (dev observation).** The GT masks themselves average ~35 connected components (distal fragments), so global fragmentation features are noisy. The v3 critical label concerns proximal/major vessels (≤ 50 mm from the ostium or radius ≥ 1.25 mm). M4 (synthetic-trained CNN) was uninformative on dev, a genuine domain gap; it is reported as a negative result.
- **M6 features (label-free, aorta-rooted).** Per tree (left/right):
  - ostium attachment (a component touching the dilated aorta);
  - skeleton length reachable from the ostium within 50 and 80 mm geodesic;
  - the number of gaps/breaks whose endpoints lie ≤ 60 mm from the ostium;
  - the max M5 image-evidence score over those proximal gaps;
  - proximal radius continuity (max drop in smoothed radius over the first 60 mm, among branches with radius ≥ 1.25 mm);
  - the share of total skeleton length in ostium-attached components.
  Scan-level = the worse of the two trees, plus the sum.
- **Model.** An L1-logistic with ≤ 3 features (C by nested CV), and also the best single M6 feature with a fixed sign. Both join the candidate list. Selection stays dev-only by the existing rule (normalised recall@20% + AURC, 5-fold patient-grouped CV), on the dev set as it stands at freeze time (including out-of-fold units if Addendum A's verification passes).
- **Comparators are unchanged.** S2 is still judged against the pre-specified simple-rule list.

## Addendum G (2026-11-14, before the freeze and before any test evaluation): deep-research revisions
- **Audit (done).** nnU-Net leakage: ImageCAS-X models/nnunet/data_prep.py (dbc7343) puts train+val (640) in imagesTr and test (160) in imagesTs, so no test scan is in nnU-Net training. Normalised recall = recall@20% / oracle recall@20%, per segmenter, ties averaged (eval/metrics.py). Dev positives: 21 critical in 160 val units. FFR-UNet test labels exist but stay sealed.
- **Power: pooled development set.** Dev = val × {CAS-Net, Swin UNETR} plus (if the Addendum A verification passes) out-of-fold nnU-Net units. Model selection uses scan-grouped CV over the pooled dev set. The 160 test scans stay sealed for the single confirmatory evaluation. Test-time power is reported honestly: the primary analysis uses ~40 FFR-UNet positives, and the pre-specified LOSO secondary uses 480 test units.
- **Second ensemble comparator (S3 setting).** Mean pairwise **clDice** agreement and mean pairwise **Betti-0 error** across nnU-Net folds f0–f4, alongside pairwise Dice (Zenk). All three are reported; S3 is judged against the strongest of them.
- **M6 constraint.** M6 candidates are limited to ≤3 features with a FIXED, anatomically justified sign (monotone logistic: less ostium-reachable length → riskier; more proximal gaps → riskier; larger proximal radius drop → riskier; tree unattached → riskier).
- **Sharpening checks (descriptive, dev now; test after the freeze).** (i) Critical units by Dice bin (<0.85, 0.85–0.90, ≥0.90) and the fraction caught at 20% per ranker. (ii) Whether the fragmentation rule tracks critical failures or distal fragments: AUROC for critical vs severe-minor-only units, and the rule recomputed on a proximal-only (≤ 60 mm from ostium) subtree.
- **Clinical measures replace FFR as headline.** Per major vessel (LM/LAD/LCx/RCA): (a) centerline minimum lumen diameter (MLD) error; (b) %-diameter-stenosis CAD-RADS band (0, 1–24, 25–49, 50–69, 70–99, 100) computed from a reference diameter (median radius over the healthy proximal/distal 10 mm), and band flips GT vs prediction; (c) SCCT segment coverage (fraction of GT centerline per segment covered within tolerance). FFR is reported as a documented negative methods finding (Addenda C–E).

## Addendum H (2026-11-15 ~02:00 UTC): FINAL FREEZE, before any test-set ranker evaluation
- **Dev expansion (Addendum A).** NOT available. fold_verification_full.json: best candidate mapping held-out-worst 0.32 (chance 0.20; bar 0.90), not verified. Dev stays at val × {CAS-Net, Swin UNETR} = 160 units, 21 critical.
- **Frozen model (by the pre-registered dev rule: normalised recall@20% + AURC, 5-fold patient-grouped CV).** M1+M3+M4 stacked L2-logistic (dev NR 0.571, AURC 3.745), files in /root/data/m4/frozen/:
  - frozen_model.json sha256 56a666ef39ed652b85e65fafb18101841dc8963cecff4b27df5ef18de0f7aa5d;
  - m4_model.pt sha256 baf841cfba4d0f08b21cbead0239e930d1f8db19a744720733b94afbfb30a266;
  - lr_convention.json sha256 53ea24bfdfc8978c6ec02b979789bda1a068dad527457cc1fad7b0d182a3f6db.
  - Known caveat, stated before the test: M4 alone was uninformative on dev (domain gap), so the stack's dev gain over M1+M3 (2 units) is within noise. M1+M3 is reported as a pre-declared sensitivity analysis.
- **Best pre-specified simple rule (chosen on dev from the Addendum-v1 list).** n_components ≥ 10 mm (dev NR 0.476, AUROC 0.782). Exploratory comparator, labelled as such: g_share_top2 (dev AUROC 0.838; not on the pre-specified list).
- **Unchanged.** Test protocol, primary endpoint (FFR-UNet test, NR@20%), S1–S5 and the comparators. The test split is evaluated ONCE with these files.

## Addendum I (2026-11-15, after the sealed test; secondary/exploratory): label robustness to a second annotator
- **Purpose.** Validate the v3 failure labels without a clinician by recomputing them against an INDEPENDENT reference: the original ImageCAS lumen masks (a different annotation team from ImageCAS-X).
- **Method.** For each unit (segmenter × scan; val and test), segment names are transferred onto the original ImageCAS mask by nearest-label mapping from the ImageCAS-X 14-class mask (within 3 mm; unmapped voxels are left unlabelled). The same frozen taxonomy_v3 code is run with the original mask as GT, with centerlines derived from its skeleton (the code's fallback). Outputs: critical_alt, severe_alt and burden_alt.
- **Reported.** Agreement critical vs critical_alt: percent agreement, Cohen's kappa, Gwet's AC1 and prevalence-adjusted bias-adjusted kappa (PABAK), with patient-bootstrap CIs. The frozen model's and the rule's NR@20%, AUROC and paired difference re-scored against critical_alt, per segmenter. The frozen model, features and thresholds are unchanged; this is a robustness analysis, not a new test of S1–S3.

- **Outcome (val only, 2026-11-15; no test run).** The original ImageCAS masks are not a second annotation of the same lumen. On 6 checked val cases they share the grid and affine (no orientation error), have 2–6× the voxels of ImageCAS-X, and contain 80–99% of each ImageCAS-X mask; Dice is 0.29–0.56 (median 0.43 over all val). Full-v3 alt prevalence was 0.93 vs 0.13 (kappa 0.02). A restricted variant (F1/F2 on LM/LAD/LCx/RCA within 50 mm only) still gave 0.91 vs 0.13 (kappa 0.03). Centerline choice is not the cause (VTK vs skeleton on ImageCAS-X: 98.8% agreement, AC1 0.98). Conclusion: this reference cannot test label robustness. Addendum I is reported as a negative feasibility finding, and the test-split run is not performed.

---

## DRAFT — not frozen (2026-11-15): Addenda J and K

Status: text only, written after the sealed test (S2 FAIL) and before any J or K run. Nothing here changes the frozen model, the rule, the primary endpoint or S1–S5. These addenda become binding only after a dated freeze.

### Addendum J (DRAFT): frozen-model replication on additional segmenters, same 160 test patients

- **Purpose.** Test whether the frozen model's gain over the rule holds for segmenters the model was not fitted on. The patients are the same 160 ImageCAS-X test scans, so this is robustness across segmenters, NOT independent-patient validation.
- **Frozen, unchanged.** Model M1+M3+M4 (sha256 as in Addendum H), rule g_n_components ≥ 10 mm, NR@20% = recall@20% / oracle recall@20% (ties averaged), 32-unit budget, test IDs of filelist/test.txt.
- **Segmenters and eligibility (fixed before outcomes are seen).**
  - Excluded: CAS-Net, Swin UNETR (the model was fitted on their val units), FFR-UNet (primary), and nnunet_mv (a majority vote derived from the folds; counting it would double-count them).
  - Family A, nnU-Net 3d_fullres 5-fold f0–f4. ONE correlated family. Eligible because the test scans are in imagesTs and never in nnU-Net training (audit, Addendum G: ImageCAS-X models/nnunet/data_prep.py, dbc7343). The five folds share training data and architecture, so they are treated as one family.
  - Family B, ImageCAS cascade (stage 1 coarse, stage 2 dilated, stage 3 patch 16/32/64, released weights). Eligible ONLY IF documentation of the released training (configuration, run log or authors' statement) shows training on the ImageCAS-X train split (560 IDs) with no test ID. The checkpoints carry no split metadata. If that cannot be verified before the J freeze, Family B is excluded from the primary and reported only as a labelled sensitivity.
- **Units.** For each family and each of the 160 test scans: a reconstruction, the v3 labels (frozen labeler, ImageCAS-X GT), and the frozen model and rule scores. Fold-level results are computed per fold, then averaged (below).
- **Primary.** Per segmenter k, D_k = NR@20 of the frozen model minus NR@20 of the rule, on that segmenter's 160 units. For Family A, D_A = mean over folds f0–f4 of the per-fold D_f (the five folds count as one segmenter). Primary estimate Δ_J = equal-weight mean of D_k over the eligible families (D_A alone, or the mean of D_A and D_B if Family B is eligible).
- **Uncertainty.** Patient-block bootstrap, 2000 resamples, seed 0. In each resample, draw the 160 patient IDs once with replacement; take every segmenter's outputs and labels for the drawn patients together; within each segmenter, rerank and recompute NR@20 for the model and the rule; compute D_k*; then Δ_J* = mean of the D_k*. The 95% CI is the 2.5th–97.5th percentile of Δ_J*. Duplicate draws are kept (they are part of the resample).
- **Success (same bar as S2).** Δ_J ≥ +10 pp AND the CI lower bound > 0.
- **Secondary.**
  - Random-effects over families: REML τ², Hartung–Knapp interval with t(K−1) df, using the bootstrap covariance of the D_k*. With K < 3 families this is not estimable and is reported as descriptive only.
  - AUROC difference (model − rule) per family, with the bootstrap interval.
  - Raw counts per segmenter: P = critical units among 160, K = review budget (32), C = critical units among the K flagged; recall = C/P, precision = C/K, oracle recall = min(P, K)/P, NR = recall/oracle recall.
  - g_share_top2: reported side by side as EXPLORATORY (not on the pre-specified comparator list; its dev AUROC was chosen after the freeze).
- **Disclosure written before the run.** The nnunet_mv test result (model 0.600 vs rule 0.420 NR) has already been seen. The individual folds are strongly correlated with their majority vote, so a Family A pass is partly anticipated and is weak, non-independent evidence. Family B: the stage-1 coarse checkpoint is absent from the released weights folder and the training split is unverified, so Family B is EXCLUDED from J (decided before any outcome).
- **Interpretation.** A pass means the frozen gain is not specific to FFR-UNet reconstructions on these 160 patients. A fail, or a pass with a family-level disagreement, is reported as such. The original S2 verdict (FAIL) stands regardless of any J result.

### Addendum K (DRAFT): controlled acquisition-shift stress test

- **Purpose.** Measure how the frozen model's gain over the rule responds to three controlled CT changes on the same 160 test patients. This is a robustness stress test, NOT external validation, and no result changes a frozen criterion.
- **Perturbations (parameters fixed here).** Applied to the raw CT, then the segmenter is run on the perturbed CT.
  1. Noise: add zero-mean Gaussian noise with SD σ_add = √(1.5² − 1)·σ0 = 1.118·σ0, so that the image noise SD rises by about 50%. σ0 is the per-scan SD in a fixed blood-pool ROI (aorta lumen from TotalSegmentator, central 50% of the cross-section, median over slices), computed on the unperturbed CT.
  2. Blur: isotropic Gaussian blur with σ = 0.6 mm in physical units (σ_vox = 0.6 / spacing per axis).
  3. Contrast: HU' = m + 0.75·(HU − m), where m is the median HU in the aorta lumen (TotalSegmentator aorta), per scan.
- **Segmenter.** ONE real segmenter, the cheapest eligible one by the timing pilot in the budget below. Candidates: nnU-Net fold f0 (no new training) or CAS-Net. The choice is recorded in the run log before any perturbed-CT label is computed.
- **Labels and scoring.** The frozen v3 labeler against ImageCAS-X GT (unperturbed GT, since the perturbation changes only the image). The frozen model and rule are scored on the perturbed reconstructions with unchanged features, thresholds and weights.
- **Primary.** For each perturbation p, the paired NR@20 difference D_p = frozen model − rule, with a patient-block bootstrap (2000, seed 0) as in J. Reported per perturbation; no pooling.
- **Pre-existing files.** Perturbed CT volumes in imagecas/volumes_shift/{noise,zblur,contrast} and segmentations preds/cas_net@noise (159 of 160) and preds/ffr_unet@noise (1 of 160) already exist from an earlier chain (2026-11-14/15). Their parameters are not recorded in this plan. Before any use they must be checked against the specification above; if they do not match, they are regenerated and the old files are kept as a labelled deviation. The existing outputs are not used in any selection.
- **GPU and CPU budget (estimates; the timing pilot on VAL scans sets the final numbers).**
  - Perturbation: CPU only, 3 × 160 volumes, a few minutes each on 8 workers.
  - Segmentation: 480 CT inferences. Assumed 20–60 s GPU per case for the chosen segmenter, giving about 2.7–8 GPU-hours. Free VRAM at the check was about 165 GB of 192 GB, with one unrelated vLLM process holding about 40 GB, so memory is not the limit.
  - Labelling: about 480 units at roughly 15 CPU-seconds each (the val run took about 300 s for 160 units with 8 workers), so about 15 wall-minutes on 8 workers.
  - Family J (nnU-Net folds): existing test predictions and v3 labels are reused, so the GPU cost is zero if the qa/v3 fold files are complete on test. Family B (cascade): about 160 inferences, 0.5–3 GPU-hours (estimate, pending pilot), plus about 5 min of labelling.
- **Integrity.** No K or J output feeds model selection, the rule or any threshold. Any perturbation that fails its specification is reported as a deviation, not corrected silently.

## DRAFT — not frozen (2026-11-15): Addenda L and M

Status: text only, written after the sealed test (S2 FAIL) and the Addendum I negative (original ImageCAS annotations are binary lumen; segment-level inter-annotator agreement is unrecoverable). Nothing here changes the frozen model, the rule, the primary endpoint or S1–S5. These addenda become binding only after a dated freeze and, for any TEST-side computation, an explicit user approval.

### Addendum L (DRAFT): independent-lumen burden — feature-disjoint label-robustness analysis

- **Purpose.** Both the frozen model's features and the v3 labels are derived from the predicted reconstruction through related geometric machinery (feature-side graph/ostium statistics vs the labeler's F1/F2/F3 run detection). Addendum I attempted the second-annotator defense and degenerated: the original ImageCAS lumen masks carry no segment labels, so `critical_alt` is near-uninformative (val: pooled %agree 0.229, Gwet AC1 −0.530, median mask Dice 0.427). L asks instead: does the frozen triage still rank severity when the label is an independent-lumen coverage burden built by pure voxel set operations, with no v3 labeler machinery?
- **Frozen, unchanged.** Frozen model M1+M3+M4 (sha256 as in Addendum H), rule `g_n_components_ge_10mm`, NR@20%, patient-block bootstrap (2000, seed 0), frozen labeler + taxonomy_v3, image data, splits.
- **Label (new; computed once, code frozen at freeze time).** Per unit (segmenter × scan):
  - Gate G: ImageCAS-X GT voxels of the base major segments {1 LM, 2 LAD, 3 LCx, 9 RCA}, dilated by PAD = 2.0 mm (spacing-aware). PDA is excluded from the gate (dominance-dependent, and dominance assignment is part of the v3 machinery).
  - orig_mj: original ImageCAS binary lumen voxels ∩ dilated G.
  - pred_lumen: predicted voxels minus background (any of the 14 classes).
  - **B_alt = 1 − |orig_mj ∩ pred_lumen| / |orig_mj|** — the fraction of independently annotated major-segment lumen the prediction does not cover (0 = perfect coverage).
  - Sensitivity B_alt5: same with PAD = 5.0 mm.
  - Exclusion: units with |orig_mj| < 50 voxels are excluded from B_alt statistics (counted, reported).
- **Why this is disjoint (construction argument).** B_alt is computed from (i) the original ImageCAS lumen — a mask the frozen model never sees: the model is label-free at training and inference (inputs: predicted mask, CT, aorta from CT only; training used synthetic corruptions of the GT, never the original lumen) — and (ii) pure voxel set operations: no centerline runs, no F-class rules, no radius or ostium-distance thresholds, no segment identity beyond the GT gate. The v3 machinery can enter B_alt only through the ImageCAS-X gate, and that overlap is quantified by controls C1–C2 below.
- **Units.** val: all val units with both original lumen and a frozen score. Phase 1 (already scoreable): 160 dev-scope units (cas_net, swin_unetr). Phase 2 (val-side score extension, dev scope, no sealed data): frozen-model scores computed for val × {ffr_unet, nnunet_f0..f4} by running the frozen M4 weights + frozen ostium features + frozen LR on these 480 units (CPU), giving 640 val units. test: FFR-UNet × 160 (primary endpoint population only), via ONE canonical sealed read of the frozen scores and of the per-unit qa/v3 label files; scores are never recomputed on test.
- **Statistics.** Per segmenter and pooled:
  1. Spearman ρ(frozen score, B_alt), patient-cluster bootstrap CI (2000, seed 0);
  2. AUROC of the frozen score for high burden = {B_alt ≥ 0.10}, CI;
  3. paired mean B_alt difference, flagged top-20% (frozen score) vs the rest, bootstrap CI.
  Controls (reported, pre-specified):
  - C1: Spearman(burden_v3, B_alt) and percent agreement(critical_v3, B_alt ≥ 0.10) — documents how much of the independent label is v3-adjacent (cheat-sheet view of the overlap);
  - C2: residual correlation — regress B_alt on (burden_v3, critical_v3), take residual r; report Spearman(frozen score, r) with CI — the part of the score's burden association not explained by the v3 label machinery.
- **Decision rule (pre-specified, descriptive; no pass/fail bar).** (i) ρ CI > 0 AND C2 CI > 0 → "transfer to the independent label source; the circularity concern is not supported." (ii) ρ CI > 0 AND C2 CI includes 0 → "transfer is mediated by v3-adjacent information." (iii) ρ CI ≤ 0 → "independent-label transfer fails; reported as a limitation." No re-run, no threshold change, no unit-set change after outcomes.
- **Budget.** val Phase 1 + Phase 2: CPU-only (features exist in qa/v3; M4 inference ≈ 480 × 96³ ≈ 0.5–3 s CPU each). test: one canonical read + 160 voxel ops + bootstrap; CPU minutes.
- **Interpretation.** L does not re-test S1–S3 and cannot change their verdicts. It bounds how far the frozen triage's ranking extends beyond the v3 label construction, using the only independent lumen source available.

### Addendum M (DRAFT): S2 estimator diagnostics (re-analysis of sealed outputs only)

- **Purpose.** S2 result: (frozen model − best simple rule) NR@20 = +12.5 pp, 95% CI [−0.3, +23.9] (paired patient-cluster bootstrap, percentile, 2000 resamples, seed 14, exactly as pre-registered). The verdict FAIL stands unchanged. M documents, on the already-sealed outputs, whether the verdict is an artifact of the estimator or robust, and states the intrinsic power limit. No new data, no new model, no threshold, no selection of any kind.
- **Frozen, unchanged.** S2 definition, bar (+10 pp AND CI excludes 0), the frozen model, the rule, the labels, the split, the verdict FAIL.
- **Reports (all on the sealed FFR-UNet test units, one canonical read):**
  - m1: the identical frozen estimator at 20,000 resamples (Monte-Carlo stability of the −0.3 boundary);
  - m2: BCa and t-approximate CIs of the same statistic (percentile vs BCa vs t);
  - m3: one-sided 97.5% lower bound and one-sided p of (model − rule);
  - m4: complementary lower-variance contrasts on the same units: AURC difference (model − rule) and Spearman rank difference of the two scores, each with paired patient bootstrap CI;
  - m5: nonparametric power: simulate the paired difference under the observed per-patient critical incidence pattern; report the achievable two-sided 95% CI half-width and the power at the observed effect size, as a function of review budget (10/20/30/50%).
- **Interpretation.** No m* output can change the S2 verdict. M's role is disclosure: show all estimators agree on direction and near-zero bound (robust near-miss) or expose an estimator artifact (Monte-Carlo or distributional), and fix the sample-size math for the discussion section.
- **Budget.** CPU seconds-to-minutes; one canonical sealed read.
