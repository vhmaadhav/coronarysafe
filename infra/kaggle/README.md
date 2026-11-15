# CPU reproduction on Kaggle

The notebook uses the private feature dataset to reproduce the pilot and can use separately staged prediction masks to evaluate an unseen segmenter. Kaggle is configured with the CPU accelerator, private visibility and internet disabled. The notebook publishes aggregate JSON on its private job output.

## Pilot

Install the package in the repository environment and prepare the upload directory.

```sh
python infra/kaggle/build_bundle.py /path/to/table_test.parquet \
  --owner YOUR_KAGGLE_USERNAME --bundle /path/to/private-bundle
kaggle datasets create --keep-tabular --path /path/to/private-bundle/input
kaggle kernels push --path /path/to/private-bundle/notebook --timeout 900
```

A previous pilot uses five patient-grouped folds. The report uses a separate review queue per segmenter and averages score ties at the review cutoff.

## Unseen-segmenter test

Use `--development` and `--third-model` with completed feature tables, or attach the private dataset containing `saved-inputs.bundle` with `--saved-mask-source OWNER/DATASET`. For a mask-based run the archive uses the following structure.

```text
cohort/val_cas_net.txt
cohort/val_swin_unetr.txt           # optional if no validation predictions exist
cohort/test_ffr_unet.txt
preds/<method>/<scan>/mask.nii.gz
imagecas_x/ImageCAS-X_dataset_v3/segmentations/<scan>.coronary.nii.gz
imagecas_x/ImageCAS-X_dataset_v3/centerlines/<scan>.coronary_left_centerline.vtk
imagecas_x/ImageCAS-X_dataset_v3/centerlines/<scan>.coronary_right_centerline.vtk
```

Each manifest lists one benchmark scan ID per line. The notebook uses the frozen v3 taxonomy and a bounded CPU pool. For the larger mask run, allow up to 90 minutes and wait for the existing run to finish before submitting another version. The QA evaluator rejects overlapping training and evaluation patient identifiers. Report the number of saved masks compared with the planned cohort and treat a partially completed, non-random run as exploratory.

## Viewer

Generate an offline viewer using `python viewer/build_cases.py --output viewer.html`. The default examples are synthetic. Use `--reference` and `--prediction` with aligned NIfTI files to inspect a case privately. An optional JSON list passed via `--markers` holds objects with `label` and `world_mm` fields. Keep images, patient-level tables and real-case viewer bundles in approved private storage.
