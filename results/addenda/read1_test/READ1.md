# Sealed Read 1 (test) : Addendum M (reduced) + Addendum J

Code sealed-read1-v1; frozen M1+M3+M4; rule g_n_components_ge_10mm; NR@20 floor k; seed/resamples per addendum.

## M, FFR-UNet units (disclosure only; S2 verdict FAIL is unchanged)

### m0 realised 2x2 among critical units

| n | P | K | oracle recall | a both | b model only | c rule only | n neither | C_model | C_rule | NR model | NR rule | dNR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 160 | 64 | 32 | 0.500 | 20.00 | 9.00 | 5.00 | 30.00 | 29.00 | 25.00 | 0.906 | 0.781 | +0.125 |

Tied units (fractional weight): model 0, rule 0.

### m1 paired patient bootstrap (percentile)

- 20,000 resamples, seed 0: dNR +0.1250, CI [-0.0069, +0.2351]

- Check at 2,000 resamples, seed 0: CI [-0.0029, +0.2386] vs posted +0.125 [-0.003, +0.239]; reproduces within 0.002: **True**

### m2 exact McNemar

- b = 9.00, c = 5.00, two-sided exact p = 0.424

### m4 AURC and AUROC (model minus rule), patient bootstrap 2,000, seed 0

| metric | model | rule | difference | 95% CI |
|---|---|---|---|---|
| AURC (critical loss) | 0.1913 | 0.2206 | -0.0293 | [-0.0754, +0.0203] |
| AUROC | 0.8447 | 0.7805 | +0.0642 | [+0.0067, +0.1194] |

### m5 power (planning only, OPTIMISTIC: observed effect and discordance used as truth)

| N | power P(dNR >= 0.10 and CI lo > 0) | mean simulated dNR |
|---|---|---|
| 160 | 0.395 | +0.115 |
| 240 | 0.560 | +0.116 |
| 320 | 0.635 | +0.115 |
| 480 | 0.670 | +0.119 |

## J, nnU-Net f0-f4 (Family A; Family B excluded)

| fold | n | P | K | C_model | C_rule | precision model/rule | recall model/rule | NR model | NR rule | D_f | dAUROC | EXPL: NR share_top2 | EXPL: D share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nnunet_f0 | 160 | 32 | 32 | 18.00 | 16.71 | 0.562 / 0.522 | 0.562 / 0.522 | 0.562 | 0.522 | +0.040 | +0.037 | 0.562 | +0.000 |
| nnunet_f1 | 160 | 30 | 32 | 17.00 | 11.87 | 0.531 / 0.371 | 0.567 / 0.396 | 0.567 | 0.396 | +0.171 | +0.127 | 0.500 | +0.067 |
| nnunet_f2 | 160 | 32 | 32 | 17.00 | 13.21 | 0.531 / 0.413 | 0.531 / 0.413 | 0.531 | 0.413 | +0.118 | +0.049 | 0.500 | +0.031 |
| nnunet_f3 | 160 | 32 | 32 | 19.00 | 14.00 | 0.594 / 0.438 | 0.594 / 0.438 | 0.594 | 0.438 | +0.156 | +0.064 | 0.531 | +0.062 |
| nnunet_f4 | 160 | 37 | 32 | 22.00 | 19.25 | 0.688 / 0.602 | 0.595 / 0.520 | 0.688 | 0.602 | +0.086 | +0.034 | 0.688 | +0.000 |

- D_A (mean of folds) = **+0.1144**, patient-block bootstrap 2,000 seed 0: CI [+0.0297, +0.2070], nan draws 0
- Success (D_A >= +0.10 and CI lower > 0): **True**
- g_share_top2 columns are EXPLORATORY and not part of the verdict.

## Exclusions and inputs

```
{
 "sealed_flag_mismatch_vs_recomputed_top20": 0,
 "J_common_scans": 160,
 "FFR_scored": 160
}
```

Assumptions: see results.json field `assumptions`.
