# Tests and verification

## Test suite

```
python -m pytest -q tests
```

The tests check that folds and splits are disjoint in patients and cover the cohort, that patients
with two scans stay in one partition, that split assembly refuses a missing patient, that the
cache holds raw Hounsfield units, that the fixed intensity map does not depend on the patch while
min-max scaling does, that every condition reaches the network on the 0 to 255 scale, that the dose conditions change positives only, that the reference construction is not resampled and treats both classes
alike, that negative rules respect their constraints and never share a slice with a nodule, that
every rule is balanced against the positives, that the counts per fold tile the cohort, and that
the manifest covers the whole eligible cohort.

Without the images, the tests read the shipped `results/patch_manifest.csv`, which is identical
to the extraction cache's manifest. On a fresh clone 16 tests pass and 7 are skipped; the skipped
tests inspect the cached tiles or the extraction log. After `bash scripts/prepare_data.sh` all 23
run.

## Checking the shipped results

Route A recomputes the tables and result figures from the shipped predictions and overwrites them
in place:

```
bash scripts/from_predictions.sh
git status --short results figures
```

On the reference platform the second command prints nothing, because every recomputed file is
identical to the shipped one. On another platform the tables should agree in every reported
digit; figures can differ in rendering.

## Checking a full rerun

Direct a rerun to a separate folder so that the shipped outputs stay available for comparison:

```
export NODULE_BIAS_RESULTS="$PWD/results_rerun"
bash scripts/run_all.sh
```

Census, extraction, the coverage audit, the statistics-only baseline, the strata analysis and the
duplicate audit are deterministic and should match exactly. Training on the Metal plug-in is not
bitwise deterministic; [RESULTS.md](RESULTS.md#determinism) states how closely retrained runs
matched the shipped ones.

## Environment check

```
python src/check_environment.py
```

It reports versions, the pylidc configuration, TensorFlow devices, cached weights and the state of
`results/` and `cache/` (see [SETUP.md](SETUP.md)).
