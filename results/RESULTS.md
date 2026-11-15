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
