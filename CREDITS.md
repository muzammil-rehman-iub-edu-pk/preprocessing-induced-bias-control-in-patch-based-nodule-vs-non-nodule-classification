# Credits

This work builds on a public dataset, open-source software and widely shared example code. The
code in this repository was written for this study, and parts of it follow public examples and
documentation closely. For each component this file names the closest public implementation we
found, with its licence and the ways our code differs, so that readers can compare and so that
those authors receive credit.

The match column uses three terms:

| term | meaning |
|---|---|
| follows | the source shows the same calls or expressions in the same structure |
| same method | the source implements the same algorithm with different code or parameters |
| related | the source covers part of what our code does, or is the nearest public reference we found |

All links were checked on 9 November 2026. Where a live page has changed since the code was
written, the link is pinned to the version the code targets: keras-io commit 2495b548, TensorFlow
r2.14, scikit-learn 1.3 and the pylidc v0.2.3 tag. Kaggle notebooks are cited by their URLs, and
their terms are those shown on each notebook page.

## 1. Data

The LIDC-IDRI collection is distributed by The Cancer Imaging Archive (TCIA) under the Creative
Commons Attribution 3.0 licence (CC BY 3.0). TCIA asks every user of the data to cite the data set,
the reference publication and, when referring to TCIA in general, the TCIA publication:

- Armato III SG, McLennan G, Bidaut L, McNitt-Gray MF, Meyer CR, Reeves AP, Zhao B, Aberle DR,
  Henschke CI, Hoffman EA, Kazerooni EA, MacMahon H, Van Beek EJR, Yankelevitz D, Biancardi AM,
  Bland PH, Brown MS, Engelmann RM, Laderach GE, Max D, Pais RC, Qing DPY, Roberts RY, Smith AR,
  Starkey A, Batra P, Caligiuri P, Farooqi A, Gladish GW, Jude CM, Munden RF, Petkovska I, Quint LE,
  Schwartz LH, Sundaram B, Dodd LE, Fenimore C, Gur D, Petrick N, Freymann J, Kirby J, Hughes B,
  Casteele AV, Gupte S, Sallam M, Heath MD, Kuhn MH, Dharaiya E, Burns R, Fryd DS, Salganicoff M,
  Anand V, Shreter U, Vastagh S, Croft BY, Clarke LP. Data From LIDC-IDRI [Data set]. The Cancer
  Imaging Archive, 2015. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX
- Armato SG III, McLennan G, Bidaut L, et al. The Lung Image Database Consortium (LIDC) and Image
  Database Resource Initiative (IDRI): a completed reference database of lung nodules on CT scans.
  Medical Physics. 2011;38(2):915-931. https://doi.org/10.1118/1.3528204
- Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D,
  Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): maintaining and operating a
  public information repository. Journal of Digital Imaging. 2013;26(6):1045-1057.
  https://doi.org/10.1007/s10278-013-9622-7

Acknowledgement requested by TCIA: "The authors acknowledge the National Cancer Institute and the
Foundation for the National Institutes of Health, and their critical role in the creation of the
free publicly available LIDC/IDRI Database used in this study."

Collection page: https://www.cancerimagingarchive.net/collection/lidc-idri/ . The download layout
("Classic Directory Name") is documented at
https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images . The older wiki page
for the collection, still linked from pylidc's documentation, no longer loads; the collection page
and the DOI replace it.

The rule that a nodule must be marked by at least three of the four readers is the LUNA16
reference standard (https://luna16.grand-challenge.org/Data/):

- Setio AAA, Traverso A, de Bel T, et al. Validation, comparison, and combination of algorithms
  for automatic detection of pulmonary nodules in computed tomography images: the LUNA16
  challenge. Medical Image Analysis. 2017;42:1-13. https://doi.org/10.1016/j.media.2017.06.015

## 2. Software

| package | version | licence | reference |
|---|---|---|---|
| pylidc | 0.2.3 | MIT | Hancock and Magnan 2016; https://github.com/notmatthancock/pylidc |
| TensorFlow (tensorflow-macos) | 2.14.0 | Apache-2.0 | Abadi et al. 2015 |
| tensorflow-metal | 1.1.0 | MIT (Apple Inc.) | https://developer.apple.com/metal/tensorflow-plugin/ |
| Keras | 2.14.0 | Apache-2.0 | Chollet et al. 2015 |
| scikit-learn | 1.3.2 | BSD-3-Clause | Pedregosa et al. 2011 |
| scikit-image | 0.21.0 | BSD-3-Clause | van der Walt et al. 2014 |
| NumPy | 1.23.5 | BSD-3-Clause | Harris et al. 2020 |
| SciPy | 1.15.3 | BSD-3-Clause | Virtanen et al. 2020 |
| pandas | 2.1.4 | BSD-3-Clause | McKinney 2010 |
| Matplotlib | 3.8.2 | Matplotlib licence (PSF based) | Hunter 2007 |
| pydicom | 2.4.3 | MIT | https://github.com/pydicom/pydicom |
| Pillow | 10.1.0 | HPND | https://github.com/python-pillow/Pillow |
| SQLAlchemy | 2.0.46 | MIT | used by pylidc |
| h5py | 3.15.1 | BSD-3-Clause | weight files |
| pytest | 9.1.1 | MIT | tests |

The environment pins follow from documented breakages in pylidc 0.2.3, the last release on PyPI.
It uses `np.int`, `np.float` and `np.bool`, which NumPy 1.24 removed
(https://numpy.org/doc/stable/release/1.24.0-notes.html; pylidc pull request 64), and
`configparser.SafeConfigParser`, which Python 3.12 removed
(https://docs.python.org/3/whatsnew/3.12.html). Both are fixed on pylidc's master branch (pull
request 70) but not in a release. pylidc also imports `pkg_resources`, which setuptools removed in
version 82 (https://setuptools.pypa.io/en/latest/history.html; pylidc pull request 74).
`tools/patch_scipy_macos27.py` is our own.

## 3. Code components and their closest public implementations

### 3.1 LIDC-IDRI access (`census.py`, `extract.py`, `check_environment.py`)

| source | licence | match | notes and differences |
|---|---|---|---|
| pylidc documentation, https://pylidc.github.io/ (tutorials for Scan and Annotation, installation page), and the v0.2.3 source, https://github.com/notmatthancock/pylidc/blob/v0.2.3/pylidc/Scan.py | MIT | follows | `pl.query(pl.Scan)`, `cluster_annotations(verbose=False)`, `to_volume()` with slices on the last axis, `bbox()`, the `~/.pylidcrc` file and its `[dicom] path` key |
| AIM-Harvard, foundation-cancer-image-biomarker, `annotations_generator.py`, https://github.com/AIM-Harvard/foundation-cancer-image-biomarker/blob/master/data/preprocessing/luna16/annotations_generator.py | MIT | follows | keeps clusters with at least three annotations; centres on a consensus box, where we use the first reader's box |
| diku-dk, credanno, `extract_LIDC_IDRI_nodules.py`, https://github.com/diku-dk/credanno/blob/main/extract_LIDC_IDRI_nodules.py | Apache-2.0 | follows | the same reader filter with medians of diameter, malignancy and subtlety; crops with `bbox(pad=...)` |
| felixpeters, lung-cancer-detection, `preprocessing.py`, https://github.com/felixpeters/lung-cancer-detection/blob/master/lung_cancer_detection/data/preprocessing.py | no licence | follows | reads acquisition fields from the first sorted DICOM file; we also read the kernel and kVp and skip pixel data |
| apjohndim, LIDC-IDRI-Extract-64x64x16-nodules, https://github.com/apjohndim/LIDC-IDRI-Extract-64x64x16-nodules | no licence | same method | first annotation of each cluster as the crop centre |
| lyricsyee, MLSL-Net, https://github.com/lyricsyee/MLSL-Net ; lmz123321, Causal_alignment, https://github.com/lmz123321/Causal_alignment | no licence | same method | per-cluster medians; multi-slice extraction |
| jaeho3690, LIDC-IDRI-Preprocessing, https://github.com/jaeho3690/LIDC-IDRI-Preprocessing | no licence | related | the most used pylidc preprocessing repository; no reader filter, consensus masks |

### 3.2 Lung mask, intensity window and patch construction (`extract.py`, `arms.py`)

| source | licence | match | notes and differences |
|---|---|---|---|
| Arnav Jain, "Candidate Generation and LUNA16 preprocessing", Kaggle, https://www.kaggle.com/code/arnavkj95/candidate-generation-and-luna16-preprocessing | see notebook page | same method | per-slice threshold, connected components, border removal, closing and hole filling. Our `lung_mask` differs in every parameter: 4-connectivity, border components removed by label, every component of at least 0.5% of the slice kept instead of the two largest, no erosion, closing with a disk of radius 6 instead of 10, holes filled on the mask itself, and the image is never masked. Its idea of candidates as in-lung pixels above a threshold is related to our `hard` rule |
| Julian de Wit, kaggle_ndsb2017, `helpers.py` and `step1_preprocess_luna16.py`, https://github.com/juliandewit/kaggle_ndsb2017 | MIT | follows (window); same method (mask, rescaling) | the fixed map of conditions G2 to G4 equals its `normalize` on [-1000, 400] times 255; the same lung mask family written in HU; rescaling to 1 mm. Random in-lung negatives with a margin around nodules are related to our `lung` rule |
| Guido Zuidhof, "Full Preprocessing Tutorial", Kaggle, https://www.kaggle.com/code/gzuidhof/full-preprocessing-tutorial ; gzuidhof/luna16, https://github.com/gzuidhof/luna16 (BSD-2-Clause) | see notebook page | same method (window, resampling); related (mask) | the [-1000, 400] window constants and resampling to 1 mm; its 3D lung segmentation is a different algorithm |
| Swetha Subramanian, LungCancerDetection, https://github.com/swethasubramanian/LungCancerDetection | MIT | same method (window); related (patches) | `normalizePlanes` on [-1000, 400]; fixed-size patches for both classes, without per-crop min-max |
| grt123 (Liao et al.), DSB2017, https://github.com/lfz/DSB2017 | MIT | related | 4-connected labelling and removal of components touching the image corners; a [-1200, 600] window in 3D |
| Wentao Zhu, DeepLung, https://github.com/wentaozhu/DeepLung | Apache-2.0 | related | [-1200, 600] window and 1 mm resampling in 3D |
| ankasor, "Improved Lung Segmentation using Watershed", Kaggle, https://www.kaggle.com/code/ankasor/improved-lung-segmentation-using-watershed | see notebook page | related | -400 HU threshold, border clearing and closing, without hole filling |
| credanno (above) | Apache-2.0 | related | positive patches cut from the annotation box and resized, as in the legacy condition G0 |

We resample in plane only, cutting a 64 mm window and resizing it with `skimage.transform.resize`,
where the sources above resample whole volumes.

### 3.3 Model and training (`train.py`, `fig_augment.py`, `fig_phases.py`)

| source | licence | match | notes and differences |
|---|---|---|---|
| Yixing Fu, "Image classification via fine-tuning with EfficientNet", Keras 2 version: https://github.com/keras-team/keras-io/blob/2495b54868feaf6a6f286e563a3539beedcfda14/examples/vision/image_classification_efficientnet_fine_tuning.py (current Keras 3 page: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/) | Apache-2.0 | follows | an in-model augmentation block with `RandomRotation(0.15)`, a frozen `EfficientNetB0(include_top=False, input_tensor=x)` with a batch normalization, dropout and dense head, then upper layers unfrozen and recompiled at 1e-4 |
| TensorFlow tutorial "Transfer learning and fine-tuning", https://www.tensorflow.org/tutorials/images/transfer_learning | Apache-2.0 code, CC BY 4.0 text | follows | the phase-two loop: `base.trainable = True`, then the lower layers frozen again |
| F. Chollet, "Transfer learning and fine-tuning" guide, Keras 2 version: https://github.com/keras-team/keras-io/blob/2a645d3ed77474fe46bc17a34d53696c4a46f4aa/guides/transfer_learning.py | Apache-2.0 | same method | the freeze, train, unfreeze and recompile workflow |
| TensorFlow tutorial "Data augmentation", https://www.tensorflow.org/tutorials/images/data_augmentation | Apache-2.0 code, CC BY 4.0 text | follows | `RandomFlip("horizontal_and_vertical")` and augmentation layers inside the model |
| Keras 2 application pages, https://keras.io/2/api/applications/ | Apache-2.0 | follows | the eight backbones and their `preprocess_input`; EfficientNet expects inputs in [0, 255] |
| TensorFlow r2.14 API pages, https://www.tensorflow.org/versions/r2.14/api_docs/python/tf | Apache-2.0 code, CC BY 4.0 text | follows | `BinaryFocalCrossentropy`, `CosineDecay`, `optimizers.legacy.Adam` with `clipnorm`, `ModelCheckpoint`, `EarlyStopping(restore_best_weights=True)`, `metrics.AUC`, `utils.set_random_seed`, `tf.data` |
| TensorFlow tutorial "Classification on imbalanced data", https://www.tensorflow.org/tutorials/structured_data/imbalanced_data | Apache-2.0 | same method | an AUC metric with max-mode early stopping that restores the best weights |
| Keras 2.14.0 source, https://github.com/keras-team/keras/tree/v2.14.0 | Apache-2.0 | related | its warning that the new Adam runs slowly on Apple silicon explains the legacy optimizer |
| Apple, "Get started with tensorflow-metal", https://developer.apple.com/metal/tensorflow-plugin/ | site terms | follows | installation of tensorflow-macos and tensorflow-metal |

Our training differs from these examples in a sigmoid output trained with focal loss, an extra
256-unit swish layer with L2 regularization, a first-phase learning rate of 1e-3, freezing by a
fraction of 70% of the backbone layers, cosine decay in the second phase, per-family
`preprocess_input` and, in the second phase, trainable batch normalization layers in the unfrozen
part of the backbone, which the examples keep frozen. `fig_augment.py` redraws the Keras transforms
with `scipy.ndimage.rotate` and `shift` for illustration.

### 3.4 Evaluation (`calibration.py`, `analyse.py`, `headline.py`, `seed_contrast.py`, `folds.py`, `dose.py`)

| source | licence | match | notes and differences |
|---|---|---|---|
| Geoff Pleiss, temperature_scaling, `_ECELoss`, https://github.com/gpleiss/temperature_scaling | MIT | follows | the binning loop with 15 equal-width bins; ours bins the positive-class score, closes bins on the left and also returns the maximum calibration error |
| torchmetrics, `binary_calibration_error(n_bins=15, norm="l1")` and `norm="max"`, https://github.com/Lightning-AI/torchmetrics | Apache-2.0 | same method | gives the same numbers except for scores of exactly 1.0 |
| scikit-learn 1.3, `calibration_curve(strategy="uniform", n_bins=15)` and `brier_score_loss`, https://scikit-learn.org/1.3/modules/generated/sklearn.calibration.calibration_curve.html | BSD-3-Clause | same method | the reliability bins and the Brier score |
| Olivier Grisel, Stack Overflow answer on ROC curve confidence intervals, https://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals | CC BY-SA | follows | the bootstrap loop with seed 42, rejection of single-class resamples and percentile bounds; ours resamples patients, not predictions |
| SciPy, `scipy.stats.bootstrap(paired=True, method="percentile")`, https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html | BSD-3-Clause | same method | paired differences; `seed_contrast.py` reuses one patient draw for every condition and seed |
| scikit-learn 1.3, `GroupKFold`, `make_pipeline`, and the "Common pitfalls" guidance on keeping the scaler inside the pipeline, https://scikit-learn.org/1.3/modules/generated/sklearn.model_selection.GroupKFold.html , https://scikit-learn.org/stable/common_pitfalls.html | BSD-3-Clause | follows | the statistics-only baseline in `probe.py`, `stratify.py` and `dose.py` |
| scikit-learn, `permutation_test_score(groups=...)`, https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html | BSD-3-Clause | same method | the optional within-patient permutation null in `probe.py` |
| NumPy `Generator.permutation` and `array_split`; scikit-learn `GroupShuffleSplit` | BSD-3-Clause | related | `folds.py` permutes patient identifiers and splits them, balancing patients rather than patches |
| NumPy `polyfit`, https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html | BSD-3-Clause | follows | power-law and exponential fits of the error counts in `dose.py` |
| yandexdataschool, roc_comparison, https://github.com/yandexdataschool/roc_comparison ; pROC, https://github.com/xrobin/pROC | MIT; GPL-3.0 | related | DeLong intervals, the analytic alternative to our patient bootstrap |

Predictive values at an assumed prevalence use Bayes' rule as given by Altman and Bland (1994).

### 3.5 Duplicate audit and statistics-only baseline (`dedup_audit.py`, `probe.py`)

| source | licence | match | notes and differences |
|---|---|---|---|
| Johannes Buchner, imagehash, https://github.com/JohannesBuchner/imagehash | BSD-2-Clause | follows | the difference hash comparison `pixels[:, 1:] > pixels[:, :-1]` and a two-pass DCT hash with a median threshold. Ours resizes by exact block means on floating-point values, uses a 16 x 16 average hash, a 17 x 17 grid for the difference hash (272 bits), drops the constant DCT term (63 bits) and uses the orthonormal DCT |
| Neal Krawetz, "Looks Like It" (average and perceptual hash), https://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html , and "Kind of Like That" (difference hash), https://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html | all rights reserved | same method | the original algorithm descriptions and the Hamming distance guidance |
| pHash library, https://github.com/aetilius/pHash | GPL-3.0 | same method | DCT hash thresholded at the median of the AC coefficients |
| Python `hashlib` (BLAKE2b), https://docs.python.org/3/library/hashlib.html | PSF | follows | the exact content hash |
| idealo, imagededup, https://github.com/idealo/imagededup ; cleanlab, cleanvision, https://github.com/cleanlab/cleanvision | Apache-2.0 | related | general duplicate finders; neither sweeps folds or measures the effect of removal on AUC |
| Barz and Denzler, ciFAIR, https://github.com/cvjena/cifair | no licence found | related | the nearest training image of each test image as a duplicate test |
| nis-research, nn-frequency-shortcuts, https://github.com/nis-research/nn-frequency-shortcuts | MIT | related | frequency shortcuts; background for the frequency feature group of the baseline |
| He et al., AdvFrequency, https://github.com/C0notSilly/AdvFrequency | no licence | related | frequency-domain shortcut mitigation |
| Lee Lab, cxr_covid, https://github.com/suinleelab/cxr_covid | MIT-style | related | acquisition shortcuts in radiographs |
| AIM-Harvard, PyRadiomics, https://github.com/AIM-Harvard/pyradiomics | BSD-3-Clause | related | first-order image statistics such as mean, standard deviation and percentiles |

### 3.6 Reporting standards that shaped the design

The patient-level partitions, the duplicate audit and the leakage tests follow REFORMS item 6b and
CLAIM 2024 item 20, which ask that all data from one patient stay in one partition, and Kapoor and
Narayanan's leakage taxonomy. These are checklists, not code.

## 4. Original to this work

We found no public implementation of the following, and present them as our own: the construction
conditions G0 to G4 as a controlled contrast, the geometry-matched 64 mm field of view, the dose
conditions that resample the positive class alone, the 80% in-lung rule for negative windows and
the scan-by-scan matching of negative counts, the statistics-only baseline that compares extracted
patches with the same regions on the native grid, the adaptive midpoint threshold rule, the paired
patient-level bootstrap across training seeds, and the per-fold duplicate sweep with the change in
AUC after removal. Kaggle notebooks could not be searched systematically, so a similar public
implementation may exist.

## 5. Licence notes

A repository without a licence keeps all rights with its authors. Such repositories, the GPL-3.0
pHash library and the HackerFactor posts are cited here for comparison and attribution only.

## 6. References

**Data**

- Armato SG III, et al. Data From LIDC-IDRI [Data set]. The Cancer Imaging Archive, 2015.
  doi:10.7937/K9/TCIA.2015.LO9QL9SX (full author list in section 1).
- Armato SG III, McLennan G, Bidaut L, et al. Medical Physics. 2011;38(2):915-931.
  doi:10.1118/1.3528204.
- Clark K, et al. Journal of Digital Imaging. 2013;26(6):1045-1057. doi:10.1007/s10278-013-9622-7.
- Hancock MC, Magnan JF. Lung nodule malignancy classification using only radiologist-quantified
  image features as inputs to statistical learning algorithms: probing the Lung Image Database
  Consortium dataset with two statistical learning methods. Journal of Medical Imaging.
  2016;3(4):044504. doi:10.1117/1.JMI.3.4.044504. (pylidc)
- Setio AAA, Traverso A, de Bel T, et al. Medical Image Analysis. 2017;42:1-13.
  doi:10.1016/j.media.2017.06.015.

**Software**

- Abadi M, et al. TensorFlow: Large-scale machine learning on heterogeneous systems. 2015.
  https://www.tensorflow.org/about/bib
- Chollet F, et al. Keras. 2015. https://keras.io
- Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine learning in Python. Journal of
  Machine Learning Research. 2011;12(85):2825-2830.
- van der Walt S, Schönberger JL, Nunez-Iglesias J, et al. scikit-image: image processing in Python.
  PeerJ. 2014;2:e453. doi:10.7717/peerj.453.
- Harris CR, Millman KJ, van der Walt SJ, et al. Array programming with NumPy. Nature.
  2020;585:357-362. doi:10.1038/s41586-020-2649-2.
- Virtanen P, Gommers R, Oliphant TE, et al. SciPy 1.0: fundamental algorithms for scientific
  computing in Python. Nature Methods. 2020;17:261-272. doi:10.1038/s41592-019-0686-2.
- McKinney W. Data structures for statistical computing in Python. Proceedings of the 9th Python in
  Science Conference. 2010:56-61. doi:10.25080/Majora-92bf1922-00a.
- Hunter JD. Matplotlib: a 2D graphics environment. Computing in Science and Engineering.
  2007;9(3):90-95. doi:10.1109/MCSE.2007.55.

**Networks and training**

- Tan M, Le QV. EfficientNet: rethinking model scaling for convolutional neural networks. ICML
  2019, PMLR 97:6105-6114. arXiv:1905.11946.
- Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition.
  ICLR 2015. arXiv:1409.1556.
- He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. CVPR 2016, pp. 770-778.
  doi:10.1109/CVPR.2016.90.
- Huang G, Liu Z, van der Maaten L, Weinberger KQ. Densely connected convolutional networks. CVPR
  2017, pp. 2261-2269. doi:10.1109/CVPR.2017.243.
- Szegedy C, Vanhoucke V, Ioffe S, Shlens J, Wojna Z. Rethinking the Inception architecture for
  computer vision. CVPR 2016, pp. 2818-2826. doi:10.1109/CVPR.2016.308.
- Russakovsky O, Deng J, Su H, et al. ImageNet Large Scale Visual Recognition Challenge.
  International Journal of Computer Vision. 2015;115(3):211-252. doi:10.1007/s11263-015-0816-y.
- Lin TY, Goyal P, Girshick R, He K, Dollár P. Focal loss for dense object detection. ICCV 2017,
  pp. 2999-3007. doi:10.1109/ICCV.2017.324.
- Kingma DP, Ba J. Adam: a method for stochastic optimization. ICLR 2015. arXiv:1412.6980.
- Loshchilov I, Hutter F. SGDR: stochastic gradient descent with warm restarts. ICLR 2017.
  arXiv:1608.03983.
- Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal
  covariate shift. ICML 2015, PMLR 37:448-456.
- Srivastava N, Hinton G, Krizhevsky A, Sutskever I, Salakhutdinov R. Dropout: a simple way to
  prevent neural networks from overfitting. Journal of Machine Learning Research.
  2014;15(56):1929-1958.
- Ramachandran P, Zoph B, Le QV. Searching for activation functions. arXiv:1710.05941, 2017.
- Pascanu R, Mikolov T, Bengio Y. On the difficulty of training recurrent neural networks. ICML 2013,
  PMLR 28(3):1310-1318.

**Evaluation and statistics**

- Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. ICML 2017,
  PMLR 70.
- Naeini MP, Cooper GF, Hauskrecht M. Obtaining well calibrated probabilities using Bayesian
  binning. AAAI 2015;29(1):2901-2907. doi:10.1609/aaai.v29i1.9602.
- Brier GW. Verification of forecasts expressed in terms of probability. Monthly Weather Review.
  1950;78(1):1-3.
- Efron B, Tibshirani RJ. An Introduction to the Bootstrap. Chapman and Hall/CRC; 1993.
  doi:10.1201/9780429246593.
- Field CA, Welsh AH. Bootstrapping clustered data. Journal of the Royal Statistical Society
  Series B. 2007;69(3):369-390. doi:10.1111/j.1467-9868.2007.00593.x.
- DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated
  receiver operating characteristic curves: a nonparametric approach. Biometrics.
  1988;44(3):837-845. doi:10.2307/2531595.
- Ojala M, Garriga GC. Permutation tests for studying classifier performance. Journal of Machine
  Learning Research. 2010;11:1833-1863.
- Phipson B, Smyth GK. Permutation P-values should never be zero. Statistical Applications in
  Genetics and Molecular Biology. 2010;9(1):39. doi:10.2202/1544-6115.1585.
- Altman DG, Bland JM. Diagnostic tests 2: predictive values. BMJ. 1994;309(6947):102.
  doi:10.1136/bmj.309.6947.102.
- Youden WJ. Index for rating diagnostic tests. Cancer. 1950;3(1):32-35.
- Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science.
  Patterns. 2023;4(9):100804. doi:10.1016/j.patter.2023.100804.
- Kapoor S, Cantrell EM, Peng K, et al. REFORMS: consensus-based recommendations for
  machine-learning-based science. Science Advances. 2024;10(18):eadk3452.
  doi:10.1126/sciadv.adk3452.
- Tejani AS, Klontzas ME, Gatti AA, et al. Checklist for Artificial Intelligence in Medical Imaging
  (CLAIM): 2024 update. Radiology: Artificial Intelligence. 2024;6(4):e240300.
  doi:10.1148/ryai.240300.

**Hashing and shortcut learning**

- Krawetz N. Looks Like It. The Hacker Factor Blog, 26 May 2011.
- Krawetz N. Kind of Like That. The Hacker Factor Blog, 21 January 2013.
- Ahmed N, Natarajan T, Rao KR. Discrete cosine transform. IEEE Transactions on Computers.
  1974;C-23(1):90-93. doi:10.1109/T-C.1974.223784.
- Aumasson JP, Neves S, Wilcox-O'Hearn Z, Winnerlein C. BLAKE2: simpler, smaller, fast as MD5.
  Lecture Notes in Computer Science, 2013, pp. 119-135. doi:10.1007/978-3-642-38980-1_8.
- Barz B, Denzler J. Do we train on test data? Purging CIFAR of near-duplicates. Journal of
  Imaging. 2020;6(6):41. doi:10.3390/jimaging6060041.
- Geirhos R, Jacobsen JH, Michaelis C, et al. Shortcut learning in deep neural networks. Nature
  Machine Intelligence. 2020;2(11):665-673. doi:10.1038/s42256-020-00257-z.
- DeGrave AJ, Janizek JD, Lee SI. AI for radiographic COVID-19 detection selects shortcuts over
  signal. Nature Machine Intelligence. 2021;3(7):610-619. doi:10.1038/s42256-021-00338-7.
- Zech JR, Badgeley MA, Liu M, Costa AB, Titano JJ, Oermann EK. Variable generalization
  performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional
  study. PLOS Medicine. 2018;15(11):e1002683. doi:10.1371/journal.pmed.1002683.
- Wang S, Veldhuis R, Brune C, Strisciuglio N. What do neural networks learn in image
  classification? A frequency shortcut perspective. ICCV 2023. arXiv:2307.09829.
- He X, Hu J, Lin Q, et al. Towards combating frequency simplicity-biased learning for domain
  generalization. NeurIPS 2024. arXiv:2410.16146.
- Torralba A, Efros AA. Unbiased look at dataset bias. CVPR 2011, pp. 1521-1528.
  doi:10.1109/CVPR.2011.5995347.
- van Griethuysen JJM, Fedorov A, Parmar C, et al. Computational radiomics system to decode the
  radiographic phenotype. Cancer Research. 2017;77(21):e104-e107.
  doi:10.1158/0008-5472.CAN-17-0339.
