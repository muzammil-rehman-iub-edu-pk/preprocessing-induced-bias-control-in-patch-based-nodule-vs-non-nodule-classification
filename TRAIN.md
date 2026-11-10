# Training

Every training run is one call to `src/train.py`. The run lists in `scripts/train_all.sh` (the
twenty-five main runs) and `scripts/train_backbones.sh` (the 107 runs of the backbone ablation)
call it with the settings used in the manuscript. Both scripts skip any run whose predictions
already exist, so with the shipped `results/` they train nothing; see
[REPRODUCE.md](REPRODUCE.md#route-b-the-full-pipeline) to retrain into a separate folder.
Training needs the extraction cache (`bash scripts/prepare_data.sh`, see [SETUP.md](SETUP.md)).

## Running a single training run

```
python src/train.py --arm G3 --backbone B0 --res 224 --split kfold --fold 1
python src/train.py --arm G0 --backbone B0 --res 128 --split single
```

| option | values | meaning |
|---|---|---|
| `--arm` | G0 to G4, D100 to D500 | construction or dose condition ([METHODS.md](METHODS.md#construction-conditions)) |
| `--backbone` | B0, B3, B4, VGG16, R50, R101, D121, IV3 | EfficientNetB0, B3 or B4, VGG-16, ResNet-50, ResNet-101, DenseNet-121 or Inception V3 |
| `--res` | pixels | network input size |
| `--split` | single, kfold | single patient-level split or five-fold cross-validation |
| `--fold` | 1 to 5 | fold for `kfold` |
| `--seed` | integer, default 42 | training seed; the split stays at 42 |
| `--clipnorm` | float, default 0 | gradient norm clipping; 0 disables it; the backbone ablation uses 1.0, which adds `_clip1` to the tag |
| `--force` | flag | retrain even if predictions exist |

Each run writes `results/runs/<tag>_predictions.csv` (validation and test scores for every
patch), `results/runs/<tag>_run.json` (configuration, sizes, epochs, wall clock and versions)
and `results/runs/<tag>.weights.h5` (the best phase-two checkpoint on validation AUC).
`train_backbones.sh` adds `backbones` to the results path, so its runs go to
`results/backbones/runs/`.

## Run lists and time

| script | runs | time on the reference machine |
|---|---|---|
| `scripts/train_all.sh` | 25: the five construction conditions at 128 pixels, three training seeds for G0 to G2, the EfficientNetB0 and B3 resolution grid, the five folds of the corrected protocol and four dose conditions | about 7.5 h of training plus a 90 s pause between runs |
| `scripts/train_backbones.sh` | 107: eight backbones on unrestricted (G2) and hard (G4) negatives at 128 and 224 pixels with seeds 42, 1234 and 2026, and five folds on G4 at 224 pixels for EfficientNetB0, VGG-16 and ResNet-50, all with gradient norms clipped at 1.0 | about 47 h of training plus a 90 s pause between runs |

The four seed-42 G2 runs of EfficientNetB0 and B3 at 128 and 224 pixels belong to the main runs
and are not clipped; `train_backbones.sh` therefore does not repeat them.

## Shipped runs and their tags

Run tags read `<condition>_<backbone>_<resolution>_<split>[_fold<k>][_seed<s>][_clip1]`; a seed
appears only when it is not 42, and `_clip1` marks a gradient norm clipped at 1.0. The shipped
runs:

| tag | role in the paper |
|---|---|
| `G0_B0_128_single` to `G4_B0_128_single` | construction conditions, calibration, threshold rules, hard negatives (Figure 7) |
| `G0_B0_128_single_seed1234`, `G0_B0_128_single_seed2026`, and the same for G1 and G2 | training-seed replicates of the first three conditions |
| `G2_B0_96_single`, `G2_B0_128_single`, `G2_B0_224_single` | EfficientNetB0 by resolution (Figure 8) |
| `G2_B3_128_single`, `G2_B3_224_single`, `G2_B3_300_single` | EfficientNetB3 by resolution (Figure 8) |
| `G3_B0_224_kfold_fold1` to `G3_B0_224_kfold_fold5` | corrected protocol, five-fold (Tables II and III, Figures 9 to 11) |
| `D100_B0_128_single`, `D150_B0_128_single`, `D200_B0_128_single`, `D366_B0_128_single` | artifact dose-response, network points (Figure 12) |
| `G4_<backbone>_<res>_single[_seed<s>]_clip1` in `results/backbones/runs/` | eight backbones on hard negatives at 128 and 224 pixels, three seeds (Table VI) |
| `G2_<backbone>_<res>_single[_seed<s>]_clip1` in `results/backbones/runs/` | the same grid on unrestricted negatives (Table VI), apart from the four seed-42 runs of B0 and B3, which are `G2_B0_128_single`, `G2_B0_224_single`, `G2_B3_128_single` and `G2_B3_224_single` |
| `G4_B0_224_kfold_fold<k>_clip1`, and the same for VGG16 and R50, in `results/backbones/runs/` | five folds on hard negatives at 224 pixels |

## Model and training settings

`train.py` builds the backbone from ImageNet weights with `include_top=False` and global average
pooling. The main experiments use EfficientNetB0 and EfficientNetB3; the backbone ablation adds
EfficientNetB4, VGG-16, ResNet-50, ResNet-101, DenseNet-121 and Inception V3. Input patches are float32 in [0, 255]
for every condition, replicated to three channels. EfficientNet's preprocessing (rescaling by
1/255 and ImageNet normalisation) lives inside the Keras graph and expects that scale, so its
input passes unchanged. The other families receive their own Keras `preprocess_input` as a layer
after augmentation: caffe mode for VGG-16 and the ResNets, torch mode for DenseNet-121 and tf mode
for Inception V3. Augmentation during training: random horizontal and vertical
flips, rotation up to 0.15 of a full turn (54 degrees) and translation up to 8% of the patch
width, with the Keras default reflect filling. The head is
batch normalisation, dropout 0.4, a 256-unit dense layer with swish activation and L2 1e-4,
dropout 0.2 and one sigmoid unit.

| setting | phase one | phase two |
|---|---|---|
| trainable | head only | head and the upper 30% of backbone layers: the first `int(0.7 * len(base.layers))` layers of the backbone model stay frozen (168 of 240 for B0, 270 of 387 for B3; the count includes the input and augmentation layers) |
| epochs | 12 | up to 25 |
| optimiser | Adam (legacy Keras), learning rate 1e-3, beta1 0.9, beta2 0.999, epsilon 1e-7 | Adam (legacy Keras), cosine decay from 1e-4 to 5% of it over 25 epochs |
| loss | binary focal cross-entropy, gamma 2.0 | the same |
| batch size | 32 | 32 |
| gradient clipping | none (norm 1.0 in the backbone ablation) | none (norm 1.0 in the backbone ablation) |
| callbacks | checkpoint on best validation AUC; early stopping on validation AUC, patience 8 | the same |

In phase two every layer in the upper 30% of the backbone becomes trainable, batch normalization
layers included, so those layers update their statistics during training. The Keras fine-tuning
examples credited in [CREDITS.md](CREDITS.md) keep batch normalization frozen instead.

Early stopping restores the weights of the best epoch of the phase when it triggers. If phase
two runs all 25 epochs, predictions use the last epoch's weights. The weights of the best
phase-two epoch are saved to `<tag>.weights.h5` either way. The split is seeded at 42 and training at 42
(`keras.utils.set_random_seed`); the training set is shuffled through an 8192-patch buffer seeded
with the training seed. G0, G1 and G2 at 128 pixels are also trained at seeds 1234 and 2026 on the
same split.
