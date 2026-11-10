# Setup

This guide installs the environment, downloads LIDC-IDRI and configures pylidc. Recomputing the
tables and figures from the shipped predictions needs only steps 1 to 3 (see
[REPRODUCE.md](REPRODUCE.md)). Steps 4 to 6 are needed to rebuild the extraction cache or to
train.

## Requirements

**Platform.** The reported results were produced on an Apple M4 (10 cores, 24 GB unified memory)
with TensorFlow 2.14 on the Metal plug-in, under macOS 26.6 for thirteen runs and macOS 27.0.1 for
the G0 and G1 runs, the two additional G2 seeds and the four dose runs. In the backbone ablation
the unrestricted-negative runs used macOS 26.6 and the hard-negative runs macOS 27.0.1. Each
`_run.json` records its platform. Any Apple silicon Mac with 16 GB or
more should run the pipeline. On Linux, `requirements-linux.txt` replaces the two Apple TensorFlow
packages with the standard TensorFlow 2.14 wheel; Linux is not the platform the reported results
were produced on. The analysis steps do not depend on the platform; training gives close but not
identical numbers on other hardware (see [RESULTS.md](RESULTS.md#determinism)).

**Disk.** About 125 GB for the LIDC-IDRI DICOM collection, 1 GB for the extraction cache and
18 MB (EfficientNetB0) to 174 MB (ResNet-101) of weights per training run, about 8.3 GB for all
132 runs.

**Software constraints, and why.**

| constraint | reason |
|---|---|
| Python 3.11 | pylidc 0.2.3 uses `configparser.SafeConfigParser`, which Python 3.12 removed |
| numpy 1.23.5 | pylidc 0.2.3 uses `np.int`, `np.float` and `np.bool`, which numpy 1.24 removed |
| setuptools below 81 | pylidc imports `pkg_resources`, which later setuptools releases deprecate and then remove; 80.10.2 is the tested version |
| scipy 1.15.3 | newest scipy that supports numpy 1.23.5 |
| tensorflow-macos 2.14.0, tensorflow-metal 1.1.0, keras 2.14.0 | versions used for every run; training uses `keras.optimizers.legacy.Adam` |
| scikit-image 0.21.0 | `transform.resize` builds every patch; other versions may change the last bits |

One Python environment runs every step: census, extraction, training, analysis and figures.

## Installation, step by step

**Step 1. Python 3.11.** On macOS with Homebrew:

```
brew install python@3.11
```

**Step 2. Virtual environment and packages.** From the repository root:

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade "pip<27"
pip install -r requirements.txt
```

Use `requirements-lock.txt` instead to install the exact versions of every package on the
reference machine, or `requirements-linux.txt` on Linux.

**Step 3. macOS 27 only.** The macOS 27 dynamic loader rejects eight of the Fortran extensions
shipped in the scipy 1.15.3 wheel, which stops `scipy.ndimage`, `skimage.transform`, `sklearn` and
`pylidc` from importing. None of the eight (ODE solvers, COBYLA and PROPACK) is used here. Run

```
python tools/patch_scipy_macos27.py
```

The tool first checks whether the imports already work and exits without changes if they do.
Otherwise it replaces three import sites with stand-ins that raise only if called, keeps the
originals beside them as `*.orig`, and verifies the import chain. Re-run it after reinstalling
scipy.

**Step 4. LIDC-IDRI.** Download the LIDC-IDRI collection from The Cancer Imaging Archive
(https://www.cancerimagingarchive.net/collection/lidc-idri/) with the NBIA Data Retriever, choosing
the "Classic Directory Name" option. The result is
`LIDC-IDRI/LIDC-IDRI-dddd/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm`, about 125 GB. Other layouts
also work, because pylidc falls back to a recursive search, but it is slower.

**Step 5. Configure pylidc.** Create `~/.pylidcrc` (on Windows `%USERPROFILE%\pylidc.conf`):

```
[dicom]
path = /absolute/path/to/LIDC-IDRI
warn = True
```

The annotation database ships inside the pylidc package, so the census needs no DICOM access.

**Step 6. ImageNet weights.** Keras downloads `efficientnetb0_notop.h5` and
`efficientnetb3_notop.h5` into `~/.keras/models/` on first use. On a machine without internet
access, copy them there beforehand from https://storage.googleapis.com/keras-applications/. The
backbone ablation fetches the ImageNet weights of its other six networks in the same way.

**Step 7. Check.**

```
python src/check_environment.py
```

It reports the Python and package versions against the pins, whether `pylidc` and `scipy` import,
the configured DICOM folder, the TensorFlow devices, the cached weights and the state of `results/`
and `cache/`. Every line should read `ok` before a full run; the cache line reads `check` until
extraction has run.

## What is not shipped

- **LIDC-IDRI images.** The collection is distributed by The Cancer Imaging Archive and must be
  downloaded as in step 4.
- **The extraction cache** (`cache/`, about 1 GB). `bash scripts/prepare_data.sh` rebuilds it
  from the DICOM files in about 25 minutes.
- **Trained weights** (`*.weights.h5`, about 8.3 GB for all 132 runs). They are excluded by
  `.gitignore`. Every run can be retrained as described in
  [TRAIN.md](TRAIN.md), and the shipped predictions are enough to recompute every reported number.
