# Results and output files

Everything in `results/` is produced by the scripts in `src/` and is shipped, so every reported
number can be checked without the images. [REPRODUCE.md](REPRODUCE.md) maps each manuscript
table and figure to the file that holds it.

## Output files

| file | content |
|---|---|
| `results/census_scans.json` | per-scan census of the pylidc database |
| `results/patch_manifest.csv` | one row per extracted patch (15,232 rows), identical to `cache/manifest.csv` after extraction |
| `results/consort.json`, `results/consort_per_arm.csv` | data accounting per stage and per condition |
| `results/cohort_stats.json` | cohort statistics, slices per nodule, positive windows failing the lung rule, counts per diameter band |
| `results/coverage_audit.json` | reconciliation of census, scan log and manifest |
| `results/fold_manifest.json` | patient identifiers per fold and for the single split |
| `results/fold_counts.csv` | patients, scans, nodules and patches per test fold (Table I) |
| `results/runs/<tag>_run.json` | configuration and outcome of each training run |
| `results/runs/<tag>_predictions.csv` | validation and test score of every patch, with its metadata |
| `results/arm_metrics.csv` | AUC with patient-level interval and the four threshold rules, per run |
| `results/arm_summary.json` | baseline and CNN side by side for G0 to G4 |
| `results/calibration.csv`, `results/calibration.json` | calibration per run, with reliability bins |
| `results/seed_contrast.csv` | G0 to G1 and G1 to G2 differences in AUC and calibration error, per seed and averaged |
| `results/dose_response.csv`, `results/dose_fit.json` | dose-response per condition and model; power-law and exponential fits |
| `results/probe_gap.csv`, `results/probe_gap.json` | statistics-only baseline per condition |
| `results/resample_factor_by_class.csv`, `results/resample_factor_by_stratum.csv` | legacy resampling factor |
| `results/gap_within_diameter_strata.csv`, `results/strata_counts.csv`, `results/size_conditioned.json` | diameter strata (Table IV) |
| `results/headline.json`, `results/headline_per_fold.csv`, `results/headline_prevalence.csv`, `results/headline_pooled_predictions.csv` | five-fold results |
| `results/backbones/runs/<tag>_run.json`, `results/backbones/runs/<tag>_predictions.csv` | the 107 backbone ablation runs |
| `results/backbones/table_vi.csv` | mean AUC on unrestricted and hard negatives per backbone and resolution (Table VI) |
| `results/backbones/g4_<res>_vs_<reference>.csv` | backbones on hard negatives: metrics and paired AUC differences from a reference |
| `results/backbones/g4_kfold_224_pooled.csv`, `results/backbones/g4_kfold_224_paired.csv` | five-fold results on hard negatives at 224 pixels |
| `results/dedup_audit.json`, `results/dedup_sweep.csv`, `results/dedup_distance_distribution.csv`, `results/dedup_effect.csv` | duplicate audit |

The tags of the shipped training runs are listed in [TRAIN.md](TRAIN.md#shipped-runs-and-their-tags).

## Expected values

Census, extraction, the coverage audit, the baseline, the strata analysis, the duplicate audit
and every table computed from stored predictions are deterministic and reproduce exactly. Key
values:

| quantity | value |
|---|---|
| pooled out-of-fold AUC, five folds, 696 patients | 0.990 (95% CI 0.988 to 0.992) |
| accuracy, sensitivity, specificity | 0.957 (0.952 to 0.963), 0.951 (0.942 to 0.960), 0.964 (0.958 to 0.970) |
| confusion counts | TP 3620, FP 136, TN 3672, FN 188 |
| per-fold AUC | 0.990, 0.988, 0.994, 0.991, 0.988 (SD 0.002) |
| statistics-only baseline, legacy construction | 0.998 extracted, 0.830 source grid, gap 0.168 |
| statistics-only baseline, fixed intensity construction | 0.831 extracted, gap 0.001 |
| legacy resampling factor | positives median 3.657 (1.438 to 4.923); negatives 1.000 |
| CNN AUC, G0 to G4 at 128 pixels, seed 42 | 1.000, 0.979, 0.980, 0.970, 0.969 |
| CNN AUC over seeds 42, 1234 and 2026, G0, G1, G2 | 1.000 in every seed; 0.977 (0.976 to 0.979); 0.980 (0.980 to 0.981) |
| accuracy over the four threshold rules and three seeds, G0, G1, G2 | 0.996 to 0.999; 0.906 to 0.918; 0.918 to 0.922 |
| expected calibration error over the three seeds, G0, G1, G2 | 0.013, 0.066, 0.064 |
| geometric component, G0 to G1, mean of three seeds | AUC -0.023 (95% CI -0.031 to -0.016) |
| photometric component, G1 to G2, mean of three seeds | AUC +0.0029 (95% CI +0.0007 to +0.0051); expected calibration error -0.0025 (95% CI -0.0106 to +0.0032) |
| EfficientNetB0 at 96, 128, 224 | 0.972, 0.980, 0.994 |
| EfficientNetB3 at 128, 224, 300 | 0.976, 0.991, 0.992 |
| positive predictive value at prevalence 1/100, 1/1000, 1/10000 | 0.212, 0.026, 0.003 |
| statistics-only baseline over the dose conditions, f = 1 to 5 | accuracy 0.755 to 0.995; errors 1868 to 39 of 7616; errors = 2144 f^-2.55 (R2 0.992 on the log scale, exponential 0.947) |
| EfficientNetB0 at 128 at f = 1, 1.5, 2, 3.66 | accuracy 0.882, 0.968, 0.993, 0.997; errors 175, 48, 11, 5 of 1482; AUC 0.954, 0.995, 1.000, 1.000 |
| change in mean AUC from unrestricted to hard negatives, eight backbones at 128 and 224 | -0.0042 to -0.0163 |
| pooled AUC on hard negatives, five folds at 224, B0, VGG16, R50 | 0.984, 0.987, 0.984 |

## Determinism

Training on the Metal plug-in is not bitwise deterministic even with fixed seeds. Retraining in a
fresh environment on the reference machine reproduced thirteen of the runs (G2 to G4 at 128
pixels, the resolution grid and the five folds) to within 5e-6 in every per-patch score.
Retraining `G2_IV3_128_single_clip1` and `G2_VGG16_128_single_clip1` with this code reproduced all
2698 per-patch scores of each exactly. For G0,
G1 and G2 the three training seeds agree to within 0.003 in AUC. The statistics-only baseline, a
logistic regression on fixed features, gives the same result on every run. Other hardware may differ further. The analysis
steps applied to the shipped predictions reproduce every value above exactly.
