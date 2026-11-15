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
