# Label robustness to a second annotator (Addendum I), split=val

Reference: original ImageCAS lumen masks (independent team), segment names transferred from ImageCAS-X within 3 mm, frozen taxonomy_v3 with skeleton-derived centerlines. Secondary, exploratory.

Units: 160 requested, 160 ok, 0 errors.

Inter-annotator mask Dice (ImageCAS-X vs original, per scan): median 0.427, IQR 0.377-0.467. Fraction of original voxels mapped within 3 mm: median 0.8705.

v3 centerline source: {'provided': 160}. Alt: {'gt_skeleton': 160}. The two differ (VTK vs skeleton), a confound that this analysis does not remove.

## Agreement, v3 critical vs critical_alt (patient-cluster bootstrap 95% CI, 2000, seed 0)

| segmenter | n | prev v3 | prev alt | % agree | kappa | Gwet AC1 | PABAK |
|---|---|---|---|---|---|---|---|
| cas_net | 80 | 0.113 | 0.938 | 0.175 [0.100, 0.263] | 0.017 [0.003, 0.042] | -0.646 [-0.800, -0.455] | -0.650 [-0.800, -0.475] |
| swin_unetr | 80 | 0.150 | 0.925 | 0.225 [0.138, 0.325] | 0.028 [0.007, 0.061] | -0.541 [-0.723, -0.337] | -0.550 [-0.725, -0.350] |
| pooled | 160 | 0.131 | 0.931 | 0.200 [0.125, 0.287] | 0.022 [0.006, 0.049] | -0.594 [-0.750, -0.414] | -0.600 [-0.750, -0.425] |

Secondary and exploratory. The frozen model, features and thresholds are unchanged; this checks label robustness, it is not a new test of S1-S3.
