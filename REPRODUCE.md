# Reproducing the results

There are two routes. Route A recomputes every table and the result figures from the shipped
per-patch predictions in a few minutes, without the images or a GPU. Route B reruns the whole
pipeline from the DICOM files. [TEST.md](TEST.md) explains how to check either route against the
shipped outputs.

## Where each manuscript item comes from

| Manuscript item | Script | Output |
|---|---|---|
| Cohort: 1018 scans, 1010 patients, 1392 nodules, 703 scans, 696 patients | `src/census.py`, `src/consort.py` | `results/census_scans.json`, `results/consort.json` |
| Nodule diameter, slice thickness, pixel spacing and slices obtained per nodule | `src/cohort_stats.py` | `results/cohort_stats.json` |
| Table I, patients, scans, nodules and patches per test fold | `src/folds.py` | `results/fold_counts.csv` |
| Table II, five-fold results | `src/headline.py` | `results/headline_per_fold.csv` |
| Table III, pooled confusion counts | `src/headline.py` | `results/headline.json` |
| Pooled AUC, accuracy, sensitivity, specificity with patient-level intervals | `src/headline.py` | `results/headline.json` |
| Predictive values at assumed prevalences | `src/headline.py` | `results/headline_prevalence.csv` |
| Statistics-only baseline on extracted and source-grid patches, and its feature subsets | `src/probe.py` | `results/probe_gap.csv` |
| Resampling factor by class and by diameter band | `src/stratify.py` | `results/resample_factor_by_class.csv`, `results/resample_factor_by_stratum.csv` |
| Table IV, separability gap within diameter strata | `src/stratify.py` | `results/gap_within_diameter_strata.csv`, `results/strata_counts.csv` |
| CNN results per construction, backbone and resolution grid, hard negatives | `src/train.py`, `src/analyse.py` | `results/runs/`, `results/arm_metrics.csv` |
| Calibration (expected calibration error, score range) and threshold rules | `src/calibration.py`, `src/analyse.py` | `results/calibration.csv`, `results/arm_metrics.csv` |
| Geometric and photometric components over three training seeds | `src/seed_contrast.py` | `results/seed_contrast.csv` |
| Artifact dose-response, statistics-only baseline and EfficientNetB0 | `src/dose.py` | `results/dose_response.csv`, `results/dose_fit.json` |
| Table VI, AUC on unrestricted and hard negatives by backbone and resolution | `src/backbones.py` | `results/backbones/table_vi.csv` |
| Backbones on hard negatives, single split and five folds at 224 pixels | `src/backbones.py` | `results/backbones/g4_128_vs_B0_128.csv`, `results/backbones/g4_128_vs_B0_224.csv`, `results/backbones/g4_224_vs_B0_224.csv`, `results/backbones/g4_kfold_224_pooled.csv`, `results/backbones/g4_kfold_224_paired.csv` |
| Duplicate audit | `src/dedup_audit.py` | `results/dedup_audit.json`, `results/dedup_sweep.csv`, `results/dedup_effect.csv` |
| Cohort coverage audit | `src/audit_coverage.py` | `results/coverage_audit.json` |
| Per-fold patient identifier lists | `src/folds.py` | `results/fold_manifest.json` |
| Figure 1, axial slices through the 31 mm nodule of LIDC-IDRI-0050 | `src/fig_overview.py` | `figures/fig01_overview.png` |
| Figure 2, pipeline | `src/fig_pipeline.py` | `figures/fig02_pipeline.png` |
| Figures 3 and 4, training phases | `src/fig_phases.py` | `figures/fig03_phase1.png`, `figures/fig04_phase2.png` |
| Figure 5, construction chain | `src/fig_construction_chain.py` | `figures/fig05_construction_chain.png` |
| Figure 6, augmentation | `src/fig_augment.py` | `figures/fig06_augmentation.png` |
| Figure 7, construction against performance | `src/fig_construction.py` | `figures/fig07_construction.png` |
| Figure 8, backbone and input resolution | `src/fig_backbone.py` | `figures/fig08_backbone.png` |
| Figures 9, 10 and 11, fold-wise results, pooled ROC, pooled confusion counts | `src/fig_headline.py` | `figures/fig09_folds.png`, `figures/fig10_roc.png`, `figures/fig11_confusion.png` |
| Figure 12, artifact dose-response | `src/fig_dose.py` | `figures/fig12_dose.png` |

Figures 1 and 3 to 6 illustrate the method and need the extraction cache (Figure 1 also needs the
DICOM files), so they belong to Route B.

## Route A: from the shipped predictions (no DICOM, no GPU, a few minutes)

One command runs them all:

```
bash scripts/from_predictions.sh
```

It runs, in order:

```
python src/headline.py
python src/analyse.py
python src/calibration.py
python src/seed_contrast.py
python src/backbones.py
python src/fig_pipeline.py
python src/fig_construction.py
python src/fig_backbone.py
python src/fig_headline.py
python src/fig_dose.py
```

These recompute Tables II and III, the pooled metrics and their intervals, the prevalence table,
the per-run metrics, the calibration and threshold results, the three-seed comparison of the
construction components, Table VI, the backbone comparisons, and Figures 2 and 7 to 12 from
`results/`.

## Route B: the full pipeline

| step | command | time on the reference machine |
|---|---|---|
| census, extraction, audits, cohort statistics, folds | `bash scripts/prepare_data.sh` | about 25 min |
| twenty-five main training runs | `bash scripts/train_all.sh` | about 7.5 h of training plus a 90 s pause between runs |
| 107 backbone ablation runs | `bash scripts/train_backbones.sh` | about 47 h of training plus a 90 s pause between runs |
| baseline, strata, headline, metrics, calibration, dose, backbones, duplicate audit | `bash scripts/analyse.sh` | about 7 min |
| figures | `bash scripts/figures.sh` | about 1 min |

`bash scripts/run_all.sh` runs the five in order.

`train_all.sh` and `train_backbones.sh` skip any run whose predictions already exist, so with the
shipped `results/` they train nothing. To retrain everything and keep the shipped outputs for comparison, direct the
outputs to a new folder:

```
export NODULE_BIAS_RESULTS="$PWD/results_rerun"
bash scripts/run_all.sh
```

Alternatively `FORCE=1 bash scripts/train_all.sh` overwrites the shipped runs in place.

Training runs and their settings are described in [TRAIN.md](TRAIN.md).

## Environment variables

| variable | default | purpose |
|---|---|---|
| `NODULE_BIAS_CACHE` | `cache/` | extraction cache location |
| `NODULE_BIAS_RESULTS` | `results/` | results location, including the census and run outputs |
| `NODULE_BIAS_FIGURES` | `figures/` | figure output location |
| `PYTHON` | `python` | interpreter used by the shell scripts |
| `COOLDOWN` | `90` | seconds between training runs, which keeps a fanless machine from throttling |
| `FORCE` | `0` | set to `1` to retrain runs that already have predictions |
