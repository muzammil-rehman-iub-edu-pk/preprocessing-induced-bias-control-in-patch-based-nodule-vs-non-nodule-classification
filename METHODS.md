# Method as implemented

This file describes what the code does, step by step, with the constants it uses. The manuscript
gives the reasoning behind each choice.

## Cohort

`census.py` reads every scan in the pylidc database (1018 scans, 1010 patients; eight patients
have two scans) and records the annotation clusters that pylidc forms from the four readers'
marks (2651 clusters). A cluster qualifies when at least three readers marked it, giving 1392
nodules in 703 scans from 696 patients; every one of these patients is used. Nodule diameter is
the median of the readers' diameters. Manufacturer, model, reconstruction kernel and kVp are read
from the first DICOM header of each series.

## Patch extraction

`extract.py` loads each eligible scan with `pylidc` (`Scan.to_volume()` returns axes in Y, X, Z
order) and stores, for every patch, a 224 x 224 tile of raw Hounsfield units on the native grid,
centred on the patch centre and padded with -1024 HU beyond the slice edge. At the finest spacing
in the cohort (0.461 mm) the tile spans 103 mm, so it contains every window used downstream. No
resampling, normalisation or label-dependent operation is applied at this stage.

**Positives.** For each qualifying nodule, the centre is the centre of the bounding box of the
first annotation in the cluster, and three axial slices are taken near the centroid, stepped by
at least one slice thickness (`ceil(thickness / slice spacing)` slices). Slices outside the nodule's z extent are dropped, so
thin nodules on thick-slice scans give fewer than three: 1122 nodules give three slices, 172 give
two and 98 give one, for 3808 positive patches. That bounding box plus 10 pixels is stored for the
legacy construction.

**Negatives.** Negatives come only from slices outside the z extent of every nodule in the scan,
and each scan contributes as many negatives per rule as it has positives, so each rule yields
exactly 3808 patches (15,232 patches in total). Three rules are sampled from one pool:

- `any`: a uniformly random position (away from the slice border) on a random nodule-free slice,
  with no anatomical constraint;
- `lung`: a centre inside the lung mask with at least 80% of the 64 mm window inside the mask;
- `hard`: a centre inside the lung mask on a pixel above -300 HU (vessels, airway walls,
  fissures), again with at least 80% of the window inside the mask.

The lung mask of a slice is the set of pixels below -400 HU, with components touching the image
border removed, components smaller than 0.5% of the slice discarded, a morphological closing with
a disk of radius 6 and hole filling. Sampling is seeded per scan with `42 * 1000003 + scan_id`.

Positive windows are not subject to the 80% rule; 2847 of the 3808 positive windows (74.8%) lie
less than 80% inside the lung mask, because a 64 mm window around a nodule near the pleura or the
mediastinum extends beyond the lung.

Outputs: one `cache/<patient>__scan_<id>.npz` per scan, `cache/manifest.csv` with one row per
patch and `cache/scan_log.csv` with one row per scan. Extraction is resumable; a scan whose file
exists is skipped. If a run is interrupted, `src/audit_coverage.py` lists any incomplete scan;
delete that scan's `.npz` and its rows, or the whole cache, and run again.

## Construction conditions

All conditions are rendered at load time from the same cached tile (`arms.py`).

| condition | name in the paper | positives | negatives | intensity |
|---|---|---|---|---|
| G0 | Legacy | bounding box plus 10 px on the native grid, resized to the input size | 128 px native window, not resized at 128 | per-crop min-max to [0, 255] |
| G1 | Geometry matched | 64 mm window resampled to a 64 x 64 grid (1 mm in plane), resized to the input size | the same | per-crop min-max to [0, 255] |
| G2 | Fixed intensity | as G1 | as G1, rule `any` | HU clipped to [-1000, 400], fixed affine map to [0, 255] |
| G3 | Parenchyma negatives | as G2 | as G2, rule `lung` | as G2 |
| G4 | Hard negatives | as G2 | as G2, rule `hard` | as G2 |

The **reference construction** used by the statistics-only baseline takes the same 64 mm window
on the native grid in raw HU, without resampling or normalisation and without any operation that
depends on the label; its size varies with pixel spacing (66 to 139 pixels).

Resampling is in plane: each patch is one native axial slice, and slices are taken at their
native z positions.

The **dose conditions** D100 to D500 isolate the resampling asymmetry of the legacy recipe. Both
classes take the same 128-pixel native window and the fixed Hounsfield map of G2; positives alone
are then resampled by a factor f, downsampled to 128/f pixels with anti-aliasing and resized back
to 128 bilinearly. The factors are 1, 1.25, 1.5, 2, 2.5, 3, 3.66, 4 and 5 (D100 to D500); D100
treats both classes identically, and 3.66 is the legacy recipe's median positive factor.
Negatives use rule `any`.

## Splits

`folds.py` sorts the 696 patient identifiers and permutes them with seed 42.

- **Five-fold cross-validation.** The permutation is cut into five test folds (140, 139, 139, 139,
  139 patients). Within each fold, 15% of the remaining patients (83 or 84) form the validation
  set and the rest (473) the training set. Patients with two scans keep both scans in one
  partition. The test folds tile the cohort, so pooling the five test sets gives one out-of-fold
  score per patch for all 696 patients.
- **Single split.** Used for the construction conditions and the backbone and resolution grid:
  139 test, 104 validation and 453 training patients.

The patient lists are in `results/fold_manifest.json`. Split assembly raises an error if any
assigned patient is missing from the manifest.

## Model and training

The network, its training schedule and every setting are described in
[TRAIN.md](TRAIN.md#model-and-training-settings).

## Thresholds

For each run, thresholds are chosen on the validation predictions only. Four rules are reported:
a fixed 0.5; the lowest threshold reaching the best validation accuracy on the grid
`linspace(0.05, 0.95, 91)`; the midpoint of that best interval on the same grid; and the midpoint
of the best interval on an adaptive grid of 2001 quantiles of the validation scores. The adaptive
midpoint is used for every thresholded metric in the paper. A patch is called positive when its
score is strictly above the threshold.

## Statistics-only baseline

`probe.py` computes ten scalar statistics per patch with no spatial information: mean, standard
deviation, 5th and 95th percentiles, mean absolute and standard deviation of the 4-neighbour
Laplacian, mean absolute horizontal and vertical difference, and the fractions of pixels within 2%
of the minimum and of the maximum. A standardised logistic regression is evaluated with five-fold
`GroupKFold` grouped by patient, once on the condition's patches (rendered at 128 pixels) and once
on the reference construction of the same regions. The difference between the two AUCs is the
separability introduced by extraction. The four Laplacian and gradient statistics form the
frequency subset; the other six form the intensity subset. Intervals bootstrap patients 1000
times with seed 42. `stratify.py` repeats the comparison for the legacy construction within
nodule-diameter bands ([0, 6), [6, 8), [8, 10), [10, 15) and 15 mm or more; each negative takes
the median diameter of the positives in its scan).

## Duplicate audit

`dedup_audit.py` renders the G3 patches at 128 pixels and computes an exact content hash and
three perceptual fingerprints: a 16 x 16 average hash (256 bits), a difference hash that compares
horizontal neighbors on a 17 x 17 grid (272 bits) and an 8 x 8 discrete cosine transform hash
without its constant term (63 bits). For every fold it finds each test patch's nearest training patch
by Hamming distance and reports the fraction within distances 0 to 20, the distance percentiles,
and the change in pooled AUC when the test-side member of every pair within distance 0, 4 and 8
of the DCT hash is removed. Nothing is removed from any reported analysis.

## Statistics

The unit of analysis is the patch. Every confidence interval resamples patients with replacement
(2000 resamples for the CNN metrics, 1000 for the baseline, percentile method, seed 42), because
patches from one patient are correlated. Standard deviations across folds use `ddof = 1`.
Calibration uses the test scores with 15 equal-width bins (expected and maximum calibration
error of the positive-class score, Brier score). Predictive values at an assumed prevalence p use Bayes' rule with the pooled
sensitivity and specificity; the evaluation set itself has prevalence 0.5.

`seed_contrast.py` measures the two components of the construction effect on the network. G0, G1
and G2 share the same test patches on the single split, so for each training seed it takes the
difference in AUC and in expected calibration error from G0 to G1 (geometric component) and from
G1 to G2 (photometric component), and the mean of the three seeds, with paired intervals that
resample test patients (2000 resamples, seed 42).

`dose.py` evaluates the statistics-only baseline on every dose condition (positives and rule
`any` negatives, 7616 patches, `GroupKFold` by patient, accuracy at 0.5 on the out-of-fold
probabilities, AUC intervals from 1000 patient resamples) and collects the EfficientNetB0 runs on
the dose conditions from `results/arm_metrics.csv` (adaptive midpoint threshold). For each model
it fits the logarithm of the error count against log f (power law) and against f (exponential)
and reports both, with the ratio of errors at f = 1 to errors at f = 3.66.

`backbones.py` compares eight backbones on rule `any` negatives (G2) and rule `hard` negatives
(G4), at 128 and 224 pixels, over three training seeds on the single split. Every run clips the
gradient norm at 1.0, except the seed-42 G2 runs of EfficientNetB0 and B3, which are the unclipped
runs in `results/runs/`. The two grids use different negatives, so the comparison in Table VI is
unpaired. Paired differences between backbones on G4 average over the three seeds and resample test
patients. The three backbones with the highest mean AUC at 224 pixels were also trained on G4 at
five folds with one seed; their out-of-fold predictions are pooled with per-fold adaptive midpoint
thresholds. Intervals use 2000 patient resamples with seed 42.
