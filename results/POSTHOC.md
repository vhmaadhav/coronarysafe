# Post-freeze descriptive analyses

Descriptive only. Frozen model, rules and thresholds are as sealed.

## 1 Risk-coverage (AURC, E-AURC, AUGRC)

| segmenter | ranker | AURC | E-AURC | AUGRC | E-AUGRC |
|---|---|---|---|---|---|
| ffr_unet | frozen | 0.1913 | 0.0966 | 0.1185 | 0.0373 |
| ffr_unet | rule_components_ge_10mm | 0.2244 | 0.1296 | 0.1338 | 0.0525 |
| ffr_unet | exploratory_share_top2 | 0.1738 | 0.079 | 0.1179 | 0.0366 |
| cas_net | frozen | 0.085 | 0.06 | 0.0545 | 0.0313 |
| cas_net | rule_components_ge_10mm | 0.0782 | 0.0532 | 0.0557 | 0.0325 |
| cas_net | exploratory_share_top2 | 0.0679 | 0.0429 | 0.0489 | 0.0256 |
| swin_unetr | frozen | 0.1436 | 0.0988 | 0.0789 | 0.0385 |
| swin_unetr | rule_components_ge_10mm | 0.1369 | 0.0921 | 0.0907 | 0.0503 |
| swin_unetr | exploratory_share_top2 | 0.1219 | 0.0771 | 0.0786 | 0.0381 |
| nnunet_mv | ensemble_1_minus_dice | 0.0907 | 0.0713 | 0.0543 | 0.0362 |

## 2 Decision curves (net benefit at threshold 0.10)

| segmenter | ranker | threshold | net_benefit |
|---|---|---|---|
| ffr_unet | frozen | 0.1 | 0.1792 |
| ffr_unet | rule_components_ge_10mm | 0.1 | 0.1514 |
| ffr_unet | exploratory_share_top2 | 0.1 | 0.1722 |
| ffr_unet | review_all | 0.1 | 0.3333 |
| cas_net | frozen | 0.1 | 0.1167 |
| cas_net | rule_components_ge_10mm | 0.1 | 0.1097 |
| cas_net | exploratory_share_top2 | 0.1 | 0.1097 |
| cas_net | review_all | 0.1 | 0.125 |
| swin_unetr | frozen | 0.1 | 0.1375 |
| swin_unetr | rule_components_ge_10mm | 0.1 | 0.0889 |
| swin_unetr | exploratory_share_top2 | 0.1 | 0.1306 |
| swin_unetr | review_all | 0.1 | 0.2014 |

## 3 Paired patient bootstrap (2000, seed 0)

| segmenter | comparison | metric | estimate | ci_lo | ci_hi | excludes_zero |
|---|---|---|---|---|---|---|
| ffr_unet | frozen minus rule_components_ge_10mm | NR@20 | 0.125 | -0.003 | 0.239 | False |
| ffr_unet | frozen minus rule_components_ge_10mm | AUROC | 0.064 | 0.007 | 0.119 | True |
| ffr_unet | frozen minus exploratory_share_top2 | NR@20 | 0.031 | -0.125 | 0.125 | False |
| ffr_unet | frozen minus exploratory_share_top2 | AUROC | -0.003 | -0.049 | 0.043 | False |
| ffr_unet | rule_components_ge_10mm minus exploratory_share_top2 | NR@20 | -0.094 | -0.24 | -0.012 | True |
| ffr_unet | rule_components_ge_10mm minus exploratory_share_top2 | AUROC | -0.067 | -0.11 | -0.022 | True |
| cas_net | frozen minus rule_components_ge_10mm | NR@20 | 0.102 | -0.034 | 0.225 | False |
| cas_net | frozen minus rule_components_ge_10mm | AUROC | 0.015 | -0.05 | 0.081 | False |
| cas_net | frozen minus exploratory_share_top2 | NR@20 | 0.031 | -0.094 | 0.148 | False |
| cas_net | frozen minus exploratory_share_top2 | AUROC | -0.032 | -0.089 | 0.022 | False |
| cas_net | rule_components_ge_10mm minus exploratory_share_top2 | NR@20 | -0.071 | -0.205 | 0.067 | False |
| cas_net | rule_components_ge_10mm minus exploratory_share_top2 | AUROC | -0.046 | -0.1 | 0.004 | False |
| swin_unetr | frozen minus rule_components_ge_10mm | NR@20 | 0.163 | 0.032 | 0.318 | True |
| swin_unetr | frozen minus rule_components_ge_10mm | AUROC | 0.073 | 0.01 | 0.133 | True |
| swin_unetr | frozen minus exploratory_share_top2 | NR@20 | 0.031 | -0.125 | 0.156 | False |
| swin_unetr | frozen minus exploratory_share_top2 | AUROC | -0.002 | -0.039 | 0.033 | False |
| swin_unetr | rule_components_ge_10mm minus exploratory_share_top2 | NR@20 | -0.132 | -0.312 | -0.013 | True |
| swin_unetr | rule_components_ge_10mm minus exploratory_share_top2 | AUROC | -0.075 | -0.13 | -0.019 | True |

## 4 Failure atlas and Dice-hides-it

| segmenter | ranker | n_dice_ge90_critical | fraction_flagged |
|---|---|---|---|
| ffr_unet | frozen | 2 | 0.0 |
| ffr_unet | rule_components_ge_10mm | 2 | 0.0 |
| ffr_unet | exploratory_share_top2 | 2 | 0.0 |
| cas_net | frozen | 12 | 0.25 |
| cas_net | rule_components_ge_10mm | 12 | 0.333 |
| cas_net | exploratory_share_top2 | 12 | 0.25 |
| swin_unetr | frozen | 3 | 0.667 |
| swin_unetr | rule_components_ge_10mm | 3 | 0.0 |
| swin_unetr | exploratory_share_top2 | 3 | 0.667 |

## 5 Calibration and subgroups

| segmenter | brier | ece | n |
|---|---|---|---|
| ffr_unet | 0.176 | 0.151 | 160 |
| cas_net | 0.126 | 0.094 | 160 |
| swin_unetr | 0.145 | 0.109 | 160 |

| segmenter | subgroup | n | critical | NR | NR_lo | NR_hi | AUROC | AUROC_lo | AUROC_hi |
|---|---|---|---|---|---|---|---|---|---|
| ffr_unet | Image Quality=1 | 8 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| ffr_unet | Image Quality=2 | 27 | 11 | 1.0 | 0.6 | 1.0 | 0.79 | 0.547 | 0.988 |
| ffr_unet | Image Quality=3 | 34 | 15 | 1.0 | 1.0 | 1.0 | 0.968 | 0.905 | 1.0 |
| ffr_unet | Image Quality=4 | 91 | 34 | 0.778 | 0.556 | 0.944 | 0.796 | 0.697 | 0.884 |
| ffr_unet | Disease=no | 87 | 24 | 0.647 | 0.357 | 0.882 | 0.792 | 0.676 | 0.893 |
| ffr_unet | Disease=yes | 73 | 40 | 0.929 | 0.786 | 1.0 | 0.88 | 0.783 | 0.958 |
| ffr_unet | Dominance=Co | 9 | 2 | 1.0 | 0.0 | 1.0 | 0.786 | 0.286 | 1.0 |
| ffr_unet | Dominance=L | 6 | 4 | 1.0 | 1.0 | 1.0 | 0.75 | 0.25 | 1.0 |
| ffr_unet | Dominance=R | 145 | 58 | 0.931 | 0.793 | 1.0 | 0.863 | 0.794 | 0.921 |
| cas_net | Image Quality=1 | 8 | 4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| cas_net | Image Quality=2 | 27 | 4 | 0.5 | 0.0 | 1.0 | 0.707 | 0.334 | 0.986 |
| cas_net | Image Quality=3 | 34 | 7 | 0.667 | 0.333 | 1.0 | 0.847 | 0.612 | 1.0 |

## 6 Clinical flagged vs unflagged

| group | metric | estimate | ci_lo | ci_hi | vessel_rows | patients |
|---|---|---|---|---|---|---|
| flagged | obstructive_flip_rate | 0.026 | 0.008 | 0.049 | 384 | 56 |
| flagged | severe_flip_rate | 0.021 | 0.005 | 0.043 | 384 | 56 |
| flagged | coverage_below_0.9_rate | 0.172 | 0.128 | 0.216 | 384 | 56 |
| flagged | abs_stenosis_err_pct_median | 8.233 | 6.714 | 9.584 | 384 | 56 |
| unflagged | obstructive_flip_rate | 0.022 | 0.014 | 0.031 | 1533 | 146 |
| unflagged | severe_flip_rate | 0.003 | 0.0 | 0.006 | 1533 | 146 |
| unflagged | coverage_below_0.9_rate | 0.091 | 0.073 | 0.108 | 1533 | 146 |
| unflagged | abs_stenosis_err_pct_median | 6.276 | 5.468 | 6.749 | 1533 | 146 |

