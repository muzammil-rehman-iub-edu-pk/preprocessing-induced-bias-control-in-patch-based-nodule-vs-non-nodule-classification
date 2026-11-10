import argparse
import json
import os
import platform
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

from arms import ARM_NEG_RULE, build_patch
from folds import assemble_split, make_folds, single_split
from paths import MANIFEST, RUNS, scan_file

PHASE1_EPOCHS = 12
PHASE2_EPOCHS = 25
PHASE1_LR = 1e-3
PHASE2_LR = 1e-4
COSINE_ALPHA = 0.05
PATIENCE = 8
BATCH = 32
DROPOUT = 0.4
L2 = 1e-4
FOCAL_GAMMA = 2.0
FROZEN_FRACTION = 0.7
ROTATION = 0.15
TRANSLATION = 0.08
SPLIT_SEED = 42
K_FOLDS = 5

BACKBONES = {
    "B0": ("EfficientNetB0", None),
    "B3": ("EfficientNetB3", None),
    "B4": ("EfficientNetB4", None),
    "VGG16": ("VGG16", "vgg16"),
    "R50": ("ResNet50", "resnet"),
    "R101": ("ResNet101", "resnet"),
    "D121": ("DenseNet121", "densenet"),
    "IV3": ("InceptionV3", "inception_v3"),
}


def run_tag(arm: str, backbone: str, res: int, split: str, fold: int,
            seed: int = SPLIT_SEED, clipnorm: float = 0.0) -> str:
    return (f"{arm}_{backbone}_{res}_{split}" + (f"_fold{fold}" if fold else "")
            + ("" if seed == SPLIT_SEED else f"_seed{seed}")
            + ("" if not clipnorm else f"_clip{clipnorm:g}"))


def build_model(backbone: str, res: int):
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = layers.Input(shape=(res, res, 3))
    x = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(ROTATION),
        layers.RandomTranslation(TRANSLATION, TRANSLATION),
    ], name="augmentation")(inputs)

    app_name, prep_module = BACKBONES[backbone]
    if prep_module is not None:
        prep = getattr(keras.applications, prep_module).preprocess_input
        x = layers.Lambda(prep, name="imagenet_preprocess")(x)
    app = getattr(keras.applications, app_name)
    base = app(include_top=False, weights="imagenet", input_tensor=x, pooling="avg")
    base.trainable = False

    h = layers.BatchNormalization()(base.output)
    h = layers.Dropout(DROPOUT)(h)
    h = layers.Dense(256, activation="swish",
                     kernel_regularizer=keras.regularizers.l2(L2))(h)
    h = layers.Dropout(DROPOUT / 2)(h)
    out = layers.Dense(1, activation="sigmoid", dtype="float32")(h)
    return keras.Model(inputs, out), base


def materialise(rows: pd.DataFrame, tiles: np.ndarray, idx: np.ndarray, arm: str, res: int):
    X = np.zeros((len(idx), res, res), dtype=np.float32)
    for k, i in enumerate(idx):
        X[k] = build_patch(tiles[i], rows.iloc[i], arm, out_px=res)
    return X


def load_tiles(manifest: pd.DataFrame):
    tiles = np.zeros((len(manifest), 224, 224), dtype=np.int16)
    order, pos = [], 0
    for (pid, sid), grp in manifest.groupby(["patient_id", "scan_id"], sort=False):
        d = np.load(scan_file(pid, sid), allow_pickle=True)
        t = d["tiles"]
        meta = pd.DataFrame({k[5:]: d[k] for k in d.files if k.startswith("meta_")})
        lookup = {(int(r.z), int(r.y), int(r.x), str(r["rule"])): i
                  for i, r in meta.iterrows()}
        for _, r in grp.iterrows():
            i = lookup[(int(r.z), int(r.y), int(r.x), str(r["rule"]))]
            tiles[pos] = t[i]
            order.append(r.name)
            pos += 1
    return tiles[:pos], manifest.loc[order].reset_index(drop=True)


def optimizer(keras, learning_rate, clipnorm: float):
    kwargs = {"clipnorm": clipnorm} if clipnorm else {}
    return keras.optimizers.legacy.Adam(learning_rate, **kwargs)


def run(arm: str, backbone: str, res: int, split_name: str, fold: int, force: bool,
        seed: int = SPLIT_SEED, clipnorm: float = 0.0):
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score
    from tensorflow import keras

    tag = run_tag(arm, backbone, res, split_name, fold, seed, clipnorm)
    RUNS.mkdir(parents=True, exist_ok=True)
    pred_path = RUNS / f"{tag}_predictions.csv"
    meta_path = RUNS / f"{tag}_run.json"
    if pred_path.exists() and not force:
        print(f"{tag}: already done, skipping")
        return

    t0 = time.time()
    keras.utils.set_random_seed(seed)

    man = pd.read_csv(MANIFEST)
    if split_name == "single":
        sp = single_split(man, seed=SPLIT_SEED)
    else:
        sp = make_folds(man, k=K_FOLDS, seed=SPLIT_SEED)[fold - 1]
    parts = {k: assemble_split(man, sp[k], arm=arm) for k in ("train", "val", "test")}
    sp_sizes = {k: len(sp[k]) for k in ("train", "val", "test")}
    print(f"{tag}: train {len(parts['train'])}  val {len(parts['val'])}  test {len(parts['test'])}"
          f"  (patients {sp_sizes['train']}/{sp_sizes['val']}/{sp_sizes['test']})", flush=True)

    rows_all = pd.concat([parts["train"], parts["val"], parts["test"]], ignore_index=True)
    tiles, rows_all = load_tiles(rows_all)
    bounds = np.cumsum([0, len(parts["train"]), len(parts["val"]), len(parts["test"])])
    idx = {k: np.arange(bounds[i], bounds[i + 1]) for i, k in enumerate(("train", "val", "test"))}

    data = {}
    for k in ("train", "val", "test"):
        X = materialise(rows_all, tiles, idx[k], arm, res)
        y = rows_all.label.to_numpy()[idx[k]].astype("float32")
        data[k] = (X, y)
    del tiles

    def ds(k, shuffle=False):
        X, y = data[k]
        d = tf.data.Dataset.from_tensor_slices((X[..., None], y))
        if shuffle:
            d = d.shuffle(min(8192, len(y)), seed=seed)
        d = d.map(lambda a, b: (tf.repeat(a, 3, axis=-1), b), num_parallel_calls=tf.data.AUTOTUNE)
        return d.batch(BATCH).prefetch(tf.data.AUTOTUNE)

    model, base = build_model(backbone, res)
    loss = keras.losses.BinaryFocalCrossentropy(gamma=FOCAL_GAMMA)
    metrics = [keras.metrics.AUC(name="auc")]

    ckpt = RUNS / f"{tag}.weights.h5"

    def callbacks():
        return [
            keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_auc", mode="max",
                                            save_best_only=True, save_weights_only=True),
            keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=PATIENCE,
                                          restore_best_weights=True),
        ]

    model.compile(optimizer=optimizer(keras, PHASE1_LR, clipnorm), loss=loss, metrics=metrics)
    h1 = model.fit(ds("train", True), validation_data=ds("val"),
                   epochs=PHASE1_EPOCHS, callbacks=callbacks(), verbose=2)

    base.trainable = True
    n_freeze = int(len(base.layers) * FROZEN_FRACTION)
    for layer in base.layers[:n_freeze]:
        layer.trainable = False
    steps = max(1, len(data["train"][1]) // BATCH)
    schedule = keras.optimizers.schedules.CosineDecay(PHASE2_LR, steps * PHASE2_EPOCHS,
                                                      alpha=COSINE_ALPHA)
    model.compile(optimizer=optimizer(keras, schedule, clipnorm), loss=loss, metrics=metrics)
    h2 = model.fit(ds("train", True), validation_data=ds("val"),
                   epochs=PHASE2_EPOCHS, callbacks=callbacks(), verbose=2)

    preds = {}
    for k in ("val", "test"):
        preds[k] = model.predict(ds(k), verbose=0).reshape(-1)
        n_nan = int(np.isnan(preds[k]).sum())
        if n_nan:
            raise RuntimeError(f"{tag}: {n_nan}/{len(preds[k])} predictions on the {k} split "
                               f"are NaN; the run diverged and no predictions were written.")

    out = []
    for k in ("val", "test"):
        sub = rows_all.iloc[idx[k]].copy()
        sub["split"] = k
        sub["score"] = preds[k]
        out.append(sub)
    pd.concat(out, ignore_index=True).to_csv(pred_path, index=False)

    y_test = data["test"][1]
    test_auc = (float(roc_auc_score(y_test, preds["test"]))
                if len(np.unique(y_test)) > 1 else float("nan"))

    with open(meta_path, "w") as fh:
        json.dump({
            "tag": tag, "arm": arm, "negative_rule": ARM_NEG_RULE[arm],
            "backbone": backbone, "resolution": res, "split": split_name, "fold": fold,
            "seed": seed, "split_seed": SPLIT_SEED, "batch": BATCH, "dropout": DROPOUT,
            "l2": L2, "augment": True, "clipnorm": clipnorm,
            "loss": f"BinaryFocalCrossentropy(gamma={FOCAL_GAMMA})",
            "phase1_epochs": PHASE1_EPOCHS, "phase1_lr": PHASE1_LR,
            "phase2_epochs": PHASE2_EPOCHS, "phase2_lr": PHASE2_LR,
            "cosine_alpha": COSINE_ALPHA, "patience": PATIENCE,
            "freeze_rule": f"layer_fraction_{FROZEN_FRACTION}",
            "backbone_layers": len(base.layers), "frozen_in_phase2": n_freeze,
            "backbone_application": BACKBONES[backbone][0],
            "backbone_preprocess": BACKBONES[backbone][1] or "identity",
            "backbone_params": int(base.count_params()),
            "n_train": int(len(data["train"][1])), "n_val": int(len(data["val"][1])),
            "n_test": int(len(y_test)),
            "patients_train": sp_sizes["train"], "patients_val": sp_sizes["val"],
            "patients_test": sp_sizes["test"],
            "epochs_ran_phase1": len(h1.history["loss"]),
            "epochs_ran_phase2": len(h2.history["loss"]),
            "test_auc": test_auc,
            "seconds": round(time.time() - t0, 1),
            "python": platform.python_version(), "tensorflow": tf.__version__,
            "platform": platform.platform(),
        }, fh, indent=2)

    print(f"{tag}: test AUC {test_auc:.4f}  ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="G2", choices=sorted(ARM_NEG_RULE))
    ap.add_argument("--backbone", default="B0", choices=sorted(BACKBONES))
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--split", default="single", choices=["single", "kfold"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--clipnorm", type=float, default=0.0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.split == "kfold" and not 1 <= a.fold <= K_FOLDS:
        ap.error(f"--split kfold needs --fold between 1 and {K_FOLDS}")
    if a.split == "single" and a.fold:
        ap.error("--split single takes no --fold")
    run(a.arm, a.backbone, a.res, a.split, a.fold, a.force, seed=a.seed, clipnorm=a.clipnorm)


if __name__ == "__main__":
    main()
