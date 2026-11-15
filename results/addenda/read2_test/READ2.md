# Sealed Read 2 (test): Addendum L-test + Addendum K-eval

Code sealed-read2-v1; frozen M1+M3+M4; rule g_n_components_ge_10mm; NR@20 floor k; patient-block bootstrap 2000, seed 0. Units: test, 160 sealed scans; K perturbations scored by the frozen pipeline.

## K: controlled acquisition shift (per perturbation, NO pooling)

| perturbation | status | labelled | missing/failed | P | K | C_model | C_rule | prec m/r | recall m/r | NR m | NR r | D (m-r) | 95% CI | AUROC m/r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noise (sealed K perturbation) | COMPLETE | 160 | 0 | 35 | 32 | 20.00 | 16.20 | 0.625/0.506 | 0.571/0.463 | 0.625 | 0.506 | +0.119 | [-0.023, +0.275] | 0.779/0.789 |
| contrast (sealed K perturbation) | COMPLETE | 160 | 0 | 46 | 32 | 25.00 | 22.50 | 0.781/0.703 | 0.543/0.489 | 0.781 | 0.703 | +0.078 | [-0.079, +0.195] | 0.795/0.778 |
| blur (sealed K perturbation) | COMPLETE | 160 | 0 | 38 | 32 | 18.00 | 17.35 | 0.562/0.542 | 0.474/0.457 | 0.562 | 0.542 | +0.020 | [-0.135, +0.131] | 0.719/0.718 |


## L: independent-lumen burden (B_alt), per segmenter and pooled

Requested 480, ok 480, excluded low support 0, errors 0.

| key | n | patients | n critical v3 | Spearman(score,B_alt) [CI] | AUROC high B_alt [CI] | top-20 minus rest B_alt [CI] | C1 Spearman(v3,B_alt) | C2 Spearman(score, resid) [CI] | branch |
|---|---|---|---|---|---|---|---|---|---|
| pooled | 480 | 160 | 143 | 0.258 [0.131, 0.371] | nan [nan, nan] | 0.047 [0.025, 0.065] | 0.182 | 0.126 [0.029, 0.215] | (i) |
| cas_net | 160 | 160 | 34 | 0.153 [-0.020, 0.309] | nan [nan, nan] | 0.018 [-0.004, 0.047] | 0.119 | 0.073 [-0.078, 0.210] | (iii) |
| ffr_unet | 160 | 160 | 64 | 0.394 [0.243, 0.524] | nan [nan, nan] | 0.045 [0.024, 0.072] | 0.256 | 0.222 [0.103, 0.326] | (i) |
| swin_unetr | 160 | 160 | 45 | 0.240 [0.082, 0.387] | nan [nan, nan] | 0.033 [0.011, 0.061] | 0.151 | 0.119 [-0.020, 0.245] | (ii) |

## Reference: unperturbed cas_net test numbers (READ from sealed summary, not recomputed)

    | cas_net | 0.625 | 0.523 | 0.813 | 0.212 |
    | cas_net | 0.594 | 0.523 |

## Assumptions

- Part L label and statistics are burden_indep.score_unit and burden_indep.stats_block, unchanged; B_alt is not reimplemented.
- L scores: test cas_net/swin_unetr/ffr_unet read from test_eval features/flags (flags.csv score); a method without sealed rows is computed once by the frozen pipeline (hash logged).
- K scores: frozen pipeline (sealed_read1.compute_units) on preds/cas_net@K_<p> with qa/v3 labels; the rule is qa graph n_components_ge_10mm.
- K D_p and L use patient-block (scan_id) bootstrap, 2,000 resamples, seed 0; no pooling across K perturbations.
- Perturbations with fewer than 150 labelled units are INCOMPLETE and carry their count.
- Missing or failed units are excluded and counted per method; they are never imputed.
