# Independent-lumen burden (Addendum L, FROZEN 2026-11-15), split=val

Label: B_alt = 1 - |orig_mj ∩ pred| / |orig_mj|; orig_mj = original ImageCAS lumen ∩ EDT ball(2 mm) of the ImageCAS-X GT major gate {LM, LAD, LCx, RCA}; B_alt5 at 5 mm; pure voxel ops, no v3 machinery; units with |orig_mj| < 50 (including 0) excluded.

The frozen model never sees the original lumen (label-free). Secondary, robustness analysis; does not re-test S1–S3.

Units: 640 requested, 640 with B_alt, 0 excluded (low or zero support), 0 errors.

## Per segmenter (label statistics only)

| method | n | mean B_alt(2mm) | mean B_alt(5mm) | mean burden_v3 | n critical v3 |
|---|---|---|---|---|---|
| cas_net | 80 | 0.646 | 0.663 | 5.048 | 9 |
| ffr_unet | 80 | 0.648 | 0.665 | 8.993 | 18 |
| nnunet_f0 | 80 | 0.651 | 0.667 | 2.993 | 10 |
| nnunet_f1 | 80 | 0.652 | 0.668 | 3.212 | 8 |
| nnunet_f2 | 80 | 0.645 | 0.662 | 3.118 | 11 |
| nnunet_f3 | 80 | 0.656 | 0.672 | 3.453 | 11 |
| nnunet_f4 | 80 | 0.659 | 0.675 | 3.096 | 8 |
| swin_unetr | 80 | 0.638 | 0.655 | 4.568 | 12 |

## Frozen score vs B_alt (pre-specified statistics)

### pooled (n=640, patients=80, n_critical_v3=87, prev(B_alt≥0.1)=1.000, mean B_alt=0.649)

- Spearman(score, B_alt) = 0.222 [0.053, 0.391]
- AUROC(score, B_alt≥0.1) = nan [nan, nan] (640 positive)
- top-20%-flagged − rest, mean B_alt = 0.031 [0.002, 0.063]
- C1 Spearman(burden_v3, B_alt) = 0.025 [-0.146, 0.199]; agreement(critical_v3, high B_alt) = 0.136 [0.081, 0.202]
- C2 Spearman(score, residual‖v3) = 0.167 [0.018, 0.320]
- Decision branch: (i)
- Verdict: (i) transfer to the independent label source; circularity concern not supported

### cas_net (n=80, patients=80, n_critical_v3=9, prev(B_alt≥0.1)=1.000, mean B_alt=0.646)

- Spearman(score, B_alt) = 0.228 [0.012, 0.440]
- AUROC(score, B_alt≥0.1) = nan [nan, nan] (80 positive)
- top-20%-flagged − rest, mean B_alt = 0.037 [-0.003, 0.072]
- C1 Spearman(burden_v3, B_alt) = 0.001 [-0.233, 0.237]; agreement(critical_v3, high B_alt) = 0.113 [0.050, 0.188]
- C2 Spearman(score, residual‖v3) = 0.173 [-0.025, 0.372]
- Decision branch: (ii)
- Verdict: (ii) transfer present but mediated / v3-adjacent information

### ffr_unet (n=80, patients=80, n_critical_v3=18, prev(B_alt≥0.1)=1.000, mean B_alt=0.648)

- Spearman(score, B_alt) = 0.215 [0.001, 0.432]
- AUROC(score, B_alt≥0.1) = nan [nan, nan] (80 positive)
- top-20%-flagged − rest, mean B_alt = 0.015 [-0.012, 0.065]
- C1 Spearman(burden_v3, B_alt) = -0.053 [-0.286, 0.205]; agreement(critical_v3, high B_alt) = 0.225 [0.138, 0.325]
- C2 Spearman(score, residual‖v3) = 0.164 [0.001, 0.344]
- Decision branch: (i)
- Verdict: (i) transfer to the independent label source; circularity concern not supported

### nnunet_f0 (n=80, patients=80, n_critical_v3=10, prev(B_alt≥0.1)=1.000, mean B_alt=0.651)

- Spearman(score, B_alt) = 0.225 [-0.005, 0.422]
- AUROC(score, B_alt≥0.1) = nan [nan, nan] (80 positive)
- top-20%-flagged − rest, mean B_alt = 0.024 [-0.015, 0.068]
- C1 Spearman(burden_v3, B_alt) = 0.040 [-0.182, 0.259]; agreement(critical_v3, high B_alt) = 0.125 [0.062, 0.200]
- C2 Spearman(score, residual‖v3) = 0.150 [-0.063, 0.348]
- Decision branch: (iii)
- Verdict: (iii) transfer not shown; reported as a limitation

### nnunet_f1 (n=80, patients=80, n_critical_v3=8, prev(B_alt≥0.1)=1.000, mean B_alt=0.652)

- Spearman(score, B_alt) = 0.247 [0.019, 0.448]
- AUROC(score, B_alt≥0.1) = nan [nan, nan] (80 positive)
- top-20%-flagged − rest, mean B_alt = 0.035 [-0.005, 0.076]
- C1 Spearman(burden_v3, B_alt) = 0.049 [-0.170, 0.266]; agreement(critical_v3, high B_alt) = 0.100 [0.037, 0.175]
- C2 Spearman(score, residual‖v3) = 0.198 [-0.015, 0.393]
- Decision branch: (ii)
- Verdict: (ii) transfer present but mediated / v3-adjacent information

### nnunet_f2 (n=80, patients=80, n_critical_v3=11, prev(B_alt≥0.1)=1.000, mean B_alt=0.645)

- Spearman(score, B_alt) = 0.222 [0.008, 0.418]
- AUROC(score, B_alt≥0.1) = nan [nan, nan] (80 positive)
- top-20%-flagged − rest, mean B_alt = 0.032 [-0.008, 0.070]
- C1 Spearman(burden_v3, B_alt) = -0.022 [-0.246, 0.201]; agreement(critical_v3, high B_alt) = 0.138 [0.062, 0.225]
- C2 Spearman(score, residual‖v3) = 0.155 [-0.046, 0.362]
- Decision branch: (ii)
- Verdict: (ii) transfer present but mediated / v3-adjacent information

### nnunet_f3 (n=80, patients=80, n_critical_v3=11, prev(B_alt≥0.1)=1.000, mean B_alt=0.656)

- Spearman(score, B_alt) = 0.163 [-0.053, 0.369]
- AUROC(score, B_alt≥0.1) = nan [nan, nan] (80 positive)
- top-20%-flagged − rest, mean B_alt = 0.017 [-0.021, 0.061]
- C1 Spearman(burden_v3, B_alt) = -0.020 [-0.243, 0.203]; agreement(critical_v3, high B_alt) = 0.138 [0.062, 0.225]
- C2 Spearman(score, residual‖v3) = 0.152 [-0.058, 0.341]
- Decision branch: (iii)
- Verdict: (iii) transfer not shown; reported as a limitation

### nnunet_f4 (n=80, patients=80, n_critical_v3=8, prev(B_alt≥0.1)=1.000, mean B_alt=0.659)

- Spearman(score, B_alt) = 0.309 [0.084, 0.511]
- AUROC(score, B_alt≥0.1) = nan [nan, nan] (80 positive)
- top-20%-flagged − rest, mean B_alt = 0.034 [0.007, 0.077]
- C1 Spearman(burden_v3, B_alt) = 0.157 [-0.061, 0.359]; agreement(critical_v3, high B_alt) = 0.100 [0.037, 0.175]
- C2 Spearman(score, residual‖v3) = 0.246 [0.037, 0.440]
- Decision branch: (i)
- Verdict: (i) transfer to the independent label source; circularity concern not supported

### swin_unetr (n=80, patients=80, n_critical_v3=12, prev(B_alt≥0.1)=1.000, mean B_alt=0.638)

- Spearman(score, B_alt) = 0.184 [-0.051, 0.413]
- AUROC(score, B_alt≥0.1) = nan [nan, nan] (80 positive)
- top-20%-flagged − rest, mean B_alt = 0.036 [0.002, 0.080]
- C1 Spearman(burden_v3, B_alt) = 0.070 [-0.161, 0.296]; agreement(critical_v3, high B_alt) = 0.150 [0.075, 0.237]
- C2 Spearman(score, residual‖v3) = 0.113 [-0.117, 0.329]
- Decision branch: (iii)
- Verdict: (iii) transfer not shown; reported as a limitation
