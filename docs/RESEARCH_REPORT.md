# CoronarySafe pilot report

CoronarySafe ranks coronary CTA reconstructions for human review using measurements available from the segmentation output. This pilot evaluates agreement with a rule-based anatomical failure definition. Reader adjudication and external evaluation remain open.

## Existing ImageCAS-X pilot

Two segmenters produced 320 reconstructions of 160 patients. The ImageCAS-X v3 reference-based taxonomy flagged 80 reconstructions across 52 patients. Sixteen of 146 reconstructions with Dice ≥0.90 received a critical flag. These findings motivate measuring anatomical failure detection alongside volumetric overlap.

| Method | Critical flags | Graph-only recall at 20% review | Graph + uncertainty recall | Graph-only cross-segmenter recall |
| --- | ---: | ---: | ---: | ---: |
| CAS-Net | 34/160 | 55.9% | 55.9% | 58.8% |
| Swin UNETR | 46/160 | 52.2% | 50.0% | 56.5% |

The paired change in oracle-normalized recall from adding uncertainty was 0.0 points (95% CI −9.4 to +6.3) for CAS-Net and −3.1 points (−9.4 to +6.3) for Swin UNETR. Patient bootstrap intervals use fixed out-of-fold predictions. The taxonomy was refined on this cohort before QA scoring, and saved CAS-Net confidence maps predate the current foreground-probability definition. The ranker comparisons are exploratory.

## Third-model test

FFR-UNet completed a three-scan check (Dice range 0.849–0.906, one taxonomy-critical case). The planned transfer experiment uses available CAS-Net validation predictions to train a fixed graph-only ranker and available FFR-UNet test predictions to evaluate it. The FFR-UNet architecture and the evaluation patients are held out from QA fitting. The acquisition cohort is shared with the pilot, and the saved cases came from an interrupted queue. Treat the result as an internal subset analysis.

## External validation

ASOCA has 40 annotated scans and its centerlines and lumen masks permit a pre-specified connectivity and false-positive stress test. The full v3 branch endpoint requires compatible branch labels. The data collection requires owner permission, and the paper describes an ethics approval or waiver for access. External validation is pending. [ASOCA data paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10006074/), [UK Data Service record](https://reshare.ukdataservice.ac.uk/855916/)
