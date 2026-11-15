# Sealed test evaluation (split=test), frozen M1+M3+M4

Frozen bundle sha256 verified. Refit-vs-frozen max coefficient diff: 3.61e-16.

| check | value | CI / note | pass |
|---|---|---|---|
| S1 NR ffr_unet frozen (>=0.75) | 0.906 | | PASS |
| S2 frozen - rule(components>=10mm) | 0.125 | [-0.003, 0.239] | FAIL |
| S3 frozen(ffr) - best ensemble (pairwise_cldice) | 0.406 | [0.15625, 0.5625] | True |
| S4 Dice>=0.90 critical recall (ffr frozen) | 0.000 | n=2 | - |
| S5 | not implemented | scan-level model | - |

| segmenter | model NR | rule NR | model AUROC | prevalence |
|---|---|---|---|---|
| ffr_unet | 0.906 | 0.781 | 0.845 | 0.400 |
| cas_net | 0.625 | 0.523 | 0.813 | 0.212 |
| swin_unetr | 0.719 | 0.556 | 0.810 | 0.281 |
| nnunet_mv | 0.600 | 0.420 | 0.814 | 0.188 |

LOSO (refit on the other two segmenters' dev units):

| held-out | model NR | rule NR |
|---|---|---|
| ffr_unet | 0.906 | 0.781 |
| cas_net | 0.594 | 0.523 |
| swin_unetr | 0.781 | 0.556 |

---

# Post-test addenda (2026-11-15; specs in docs/ADDENDA_FROZEN_2026-11-15.md incl. Errata 1–10)

No result below changes S1–S3. S2 remains FAIL. Outputs: /root/data/addenda/{read1_test,read2_test,L,N,K} (mirrored to HF vhmaadhav/coronarysafe-results and the laptop).

## Addendum I: second annotator (val only), negative feasibility
The original ImageCAS masks label a 2–6× larger lumen than ImageCAS-X and contain 80–99% of it. Median Dice is 0.43. There is no orientation error, and centerlines are not the cause (VTK vs skeleton agreement 98.8%). Alt-critical prevalence is 0.93 vs 0.13 (restricted variant 0.91). This reference cannot test label robustness, so the test run was not performed.

## Addendum M: S2 diagnostics (FFR-UNet test, disclosure only)
| P | K | both | model only | rule only | neither | C model | C rule |
|---|---|---|---|---|---|---|---|
| 64 | 32 | 20 | 9 | 5 | 30 | 29 | 25 |
- The original bootstrap is reproduced (2,000 resamples, seed 0): [−0.003, +0.239]. At 20,000 resamples: +0.125 [−0.007, +0.235]. Still FAIL.
- Exact McNemar (9 vs 5): p = 0.42. ΔAUROC +0.064 [+0.007, +0.119]. ΔAURC −0.029 [−0.075, +0.020].
- Power (optimistic) to clear the S2 bar: 0.40 at N=160, 0.56 at 240, 0.64 at 320, 0.67 at 480. An effect of ~+12 pp against a +10 pp bar is near-unwinnable at realistic N.
- Plain-language headline: reviewing 32 of 160 scans, the model catches 29 of 64 critical reconstructions (precision 0.91, raw recall 0.45) vs 25 for the rule.

## Addendum J: replication on nnU-Net folds f0–f4 (same 160 test patients), PASS
Per-fold D (model − rule NR@20): f0 +0.040, f1 +0.171, f2 +0.118, f3 +0.156, f4 +0.086. **D_A = +0.114 [+0.030, +0.207]: meets the S2 bar.** ΔAUROC is positive on every fold. Disclosed caveats: these are the same patients; the folds track the nnunet_mv result already seen; this is not independent-patient validation. The exploratory g_share_top2 is within 0–0.07 of the model on every fold.

## Addendum K: acquisition-shift stress test (CAS-Net on perturbed test CTs)
Unperturbed CAS-Net reference: model 0.625 vs rule 0.523 (prevalence 0.21).
| perturbation | P | C model / rule | NR model / rule | D [95% CI] | AUROC model / rule |
|---|---|---|---|---|---|
| noise (+50% SD) | 35 | 20 / 16.2 | 0.625 / 0.506 | +0.119 [−0.023, +0.275] | 0.779 / 0.789 |
| contrast (×0.75) | 46 | 25 / 22.5 | 0.781 / 0.703 | +0.078 [−0.079, +0.195] | 0.795 / 0.778 |
| blur (σ 0.6 mm) | 38 | 18 / 17.35 | 0.562 / 0.542 | +0.020 [−0.135, +0.131] | 0.719 / 0.718 |
The model's edge is preserved under noise, shrinks under contrast, and vanishes under blur; its AUROC advantage disappears under all three. Every frozen-model input is derived from the predicted mask: graph features, ostium features against the aorta mask from the UNPERTURBED CT, and an M4 CNN that reads only the binary lumen mask. So the shrinking edge reflects a change in the segmenter's failure profile under degraded images, not image features failing. Under shift, the learned combination's advantage over the single topology rule is not robust.

## Addendum L: independent-lumen burden B_alt (test: cas_net, swin_unetr, ffr_unet × 160)
| key | Spearman(score, B_alt) | top-20% minus rest | C2 (residual) | branch |
|---|---|---|---|---|
| pooled | 0.26 [0.13, 0.37] | +0.047 [0.025, 0.065] | 0.13 [0.03, 0.22] | (i) |
| ffr_unet | 0.39 [0.24, 0.52] | +0.045 | 0.22 [0.10, 0.33] | (i) |
| swin_unetr | 0.24 [0.08, 0.39] | +0.033 | 0.12 [−0.02, 0.25] | (ii) |
| cas_net | 0.15 [−0.02, 0.31] | +0.018 | 0.07 [−0.08, 0.21] | (iii) |
Caveat (major): B_alt is dominated by the reference-thickness mismatch. Mean B_alt is ~0.65 for EVERY segmenter on val, so the pre-specified "high burden ≥ 0.10" AUROC is not estimable: every unit is above it. The association is real but small, and it holds for 2 of 3 segmenters. Treat it as weak support against pure label circularity.

## Addendum N (val/dev only)
- N2 (in-sample ablation): dropping any single one of the 22 inputs changes dev NR@20 by at most 0.05 (largest: g_max_branch_depth, m4_logit). The features are redundant.
- N1: the rule-threshold perturbation in its "flag" form was degenerate for n_components (all flagged). The ranking form is unaffected (val NR 0.48, AUROC 0.76). Reported as an implementation limitation.
