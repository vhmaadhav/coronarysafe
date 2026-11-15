# Label robustness to a second annotator (Addendum I), split=val

Label: restricted critical tier (F1/F2 on LM/LAD/LCx/RCA within 50 mm; no thickness terms). Prevalence v3 column = ImageCAS-X reference; alt column = original-mask reference.

Reference: original ImageCAS lumen masks (independent team), segment names transferred from ImageCAS-X within 3 mm, frozen taxonomy_v3 with skeleton-derived centerlines. Secondary, exploratory.

Units: 160 requested, 160 ok, 0 errors.

Inter-annotator mask Dice (ImageCAS-X vs original, per scan): median 0.427, IQR 0.377-0.467. Fraction of original voxels mapped within 3 mm: median 0.8705.

v3 centerline source: {'provided': 160}. Alt: {'gt_skeleton': 160}. The two differ (VTK vs skeleton), a confound that this analysis does not remove.

## Agreement, v3 critical vs critical_alt (patient-cluster bootstrap 95% CI, 2000, seed 0)

| segmenter | n | prev v3 | prev alt | % agree | kappa | Gwet AC1 | PABAK |
|---|---|---|---|---|---|---|---|
| cas_net | 80 | 0.113 | 0.912 | 0.200 [0.113, 0.300] | 0.024 [0.006, 0.055] | -0.599 [-0.773, -0.400] | -0.600 [-0.775, -0.400] |
| swin_unetr | 80 | 0.150 | 0.900 | 0.250 [0.163, 0.338] | 0.038 [0.012, 0.078] | -0.496 [-0.675, -0.299] | -0.500 [-0.675, -0.325] |
| pooled | 160 | 0.131 | 0.906 | 0.225 [0.144, 0.312] | 0.031 [0.011, 0.063] | -0.548 [-0.709, -0.368] | -0.550 [-0.713, -0.375] |

Secondary and exploratory. The frozen model, features and thresholds are unchanged; this checks label robustness, it is not a new test of S1-S3.
