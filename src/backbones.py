import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analyse import metrics_at, threshold_rules
from calibration import ece_mce
from headline import adaptive_midpoint
from paths import BACKBONE_RESULTS, RUNS

BACKBONES = ["B0", "B3", "B4", "VGG16", "R50", "R101", "D121", "IV3"]
UNCLIPPED_AT_SEED_42 = ("B0", "B3")
KFOLD_BACKBONES = ["B0", "VGG16", "R50"]
KFOLD_CONTRASTS = (("VGG16", "B0"), ("R50", "B0"), ("VGG16", "R50"))
RESOLUTIONS = (128, 224)
COMPARISONS = ((128, "B0_128"), (128, "B0_224"), (224, "B0_224"))
SEEDS = (42, 1234, 2026)
N_BOOT = 2000
BOOT_SEED = 42
KEY = ["patient_id", "scan_id", "label", "rule", "z", "y", "x"]
GRID_RUNS = BACKBONE_RESULTS / "runs"


def clipped_tag(arm: str, backbone: str, res: int, seed: int) -> str:
    return f"{arm}_{backbone}_{res}_single" + ("" if seed == 42 else f"_seed{seed}") + "_clip1"


def run_auc(path) -> float:
    with open(path) as fh:
        return json.load(fh)["test_auc"]


def unrestricted_aucs(backbone: str, res: int) -> list[float]:
    out = []
    for seed in SEEDS:
        if seed == 42 and backbone in UNCLIPPED_AT_SEED_42:
            out.append(run_auc(RUNS / f"G2_{backbone}_{res}_single_run.json"))
        else:
            out.append(run_auc(GRID_RUNS / f"{clipped_tag('G2', backbone, res, seed)}_run.json"))
    return out


def hard_aucs(backbone: str, res: int) -> list[float]:
    return [run_auc(GRID_RUNS / f"{clipped_tag('G4', backbone, res, seed)}_run.json") for seed in SEEDS]


def load_hard(backbone: str, res: int, seed: int):
    pred = pd.read_csv(GRID_RUNS / f"{clipped_tag('G4', backbone, res, seed)}_predictions.csv")
    return pred[pred.split == "val"].reset_index(drop=True), pred[pred.split == "test"].reset_index(drop=True)


def verdict(lo: float, hi: float) -> str:
    return "above" if lo > 0 else ("below" if hi < 0 else "tie")


def comparison(res: int, ref: str) -> pd.DataFrame:
    ref_backbone, ref_res = ref.split("_")
    scores, rows, base = {}, [], None
    for backbone in BACKBONES:
        for seed in SEEDS:
            val, test = load_hard(backbone, res, seed)
            if base is None:
                base = test
            assert test[KEY].equals(base[KEY])
            scores[backbone, seed] = test.score.to_numpy(float)
            m = metrics_at(test, threshold_rules(val)["adaptive_midpoint"])
            rows.append({"backbone": backbone, "seed": seed,
                         "auc": roc_auc_score(test.label, test.score),
                         "acc": m["accuracy"], "sens": m["sensitivity"], "spec": m["specificity"],
                         "fp": m["fp"], "fn": m["fn"],
                         "ece": ece_mce(test.label.to_numpy(float), test.score.to_numpy(float))[0]})
    for seed in SEEDS:
        if int(ref_res) == res:
            scores["REF", seed] = scores[ref_backbone, seed]
        else:
            _, test = load_hard(ref_backbone, int(ref_res), seed)
            assert test[KEY].equals(base[KEY])
            scores["REF", seed] = test.score.to_numpy(float)

    runs = pd.DataFrame(rows)
    y = base.label.to_numpy(int)
    patients = base.patient_id.unique()
    index = base.groupby("patient_id").indices

    def differences(ix):
        reference = [roc_auc_score(y[ix], scores["REF", s][ix]) for s in SEEDS]
        return {b: np.mean([roc_auc_score(y[ix], scores[b, s][ix]) - reference[j]
                            for j, s in enumerate(SEEDS)]) for b in BACKBONES}

    point = differences(np.arange(len(y)))
    rng = np.random.default_rng(BOOT_SEED)
    boot = {b: [] for b in BACKBONES}
    for _ in range(N_BOOT):
        pick = rng.choice(patients, len(patients), replace=True)
        d = differences(np.concatenate([index[p] for p in pick]))
        for b in BACKBONES:
            boot[b].append(d[b])

    out = []
    for b in BACKBONES:
        g = runs[runs.backbone == b]
        lo, hi = np.percentile(boot[b], [2.5, 97.5])
        unrestricted = unrestricted_aucs(b, res)
        out.append({"backbone": b, "auc_mean": g.auc.mean(), "auc_min": g.auc.min(), "auc_max": g.auc.max(),
                    "acc": g.acc.mean(), "sens": g.sens.mean(), "spec": g.spec.mean(),
                    "fp": g.fp.mean(), "fn": g.fn.mean(),
                    "fp_seeds": "/".join(str(int(v)) for v in g.fp),
                    "fn_seeds": "/".join(str(int(v)) for v in g.fn),
                    "ece": g.ece.mean(),
                    f"diff_vs_{ref}": point[b], "lo": lo, "hi": hi, "verdict": verdict(lo, hi),
                    "g2_auc_mean": np.mean(unrestricted), "g2_n": len(unrestricted)})
    return pd.DataFrame(out).sort_values("auc_mean", ascending=False)


def pooled_point(y, pred, score, ix) -> dict:
    yy, pp, ss = y[ix], pred[ix], score[ix]
    tp = int(((pp == 1) & (yy == 1)).sum())
    fp = int(((pp == 1) & (yy == 0)).sum())
    tn = int(((pp == 0) & (yy == 0)).sum())
    fn = int(((pp == 0) & (yy == 1)).sum())
    return {"auc": roc_auc_score(yy, ss), "acc": (tp + tn) / len(yy), "sens": tp / (tp + fn),
            "spec": tn / (tn + fp), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def five_fold() -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled, per_fold = {}, []
    for backbone in KFOLD_BACKBONES:
        parts = []
        for fold in range(1, 6):
            pred = pd.read_csv(GRID_RUNS / f"G4_{backbone}_224_kfold_fold{fold}_clip1_predictions.csv")
            val, test = pred[pred.split == "val"], pred[pred.split == "test"].copy()
            threshold = adaptive_midpoint(val)
            test["pred"] = (test.score > threshold).astype(int)
            parts.append(test)
            per_fold.append({"backbone": backbone, "fold": fold, "threshold": threshold,
                             "auc": roc_auc_score(test.label, test.score)})
        pooled[backbone] = pd.concat(parts).sort_values(KEY).reset_index(drop=True)

    ref = pooled[KFOLD_BACKBONES[0]]
    for backbone in KFOLD_BACKBONES[1:]:
        assert pooled[backbone][KEY].equals(ref[KEY]), backbone
    y = ref.label.to_numpy()
    score = {b: pooled[b].score.to_numpy(float) for b in KFOLD_BACKBONES}
    pred = {b: pooled[b].pred.to_numpy() for b in KFOLD_BACKBONES}

    full = np.arange(len(y))
    estimate = {b: pooled_point(y, pred[b], score[b], full) for b in KFOLD_BACKBONES}
    patients = ref.patient_id.unique()
    index = ref.groupby("patient_id").indices
    rng = np.random.default_rng(BOOT_SEED)
    boot = {b: {k: [] for k in ("auc", "acc", "sens", "spec")} for b in KFOLD_BACKBONES}
    contrast = {pair: [] for pair in KFOLD_CONTRASTS}
    for _ in range(N_BOOT):
        ix = np.concatenate([index[p] for p in rng.choice(patients, len(patients), replace=True)])
        m = {b: pooled_point(y, pred[b], score[b], ix) for b in KFOLD_BACKBONES}
        for b in KFOLD_BACKBONES:
            for k in boot[b]:
                boot[b][k].append(m[b][k])
        for a, b in contrast:
            contrast[a, b].append(m[a]["auc"] - m[b]["auc"])

    folds = pd.DataFrame(per_fold)
    rows = []
    for b in KFOLD_BACKBONES:
        e = estimate[b]
        ci = {k: np.percentile(boot[b][k], [2.5, 97.5]) for k in boot[b]}
        f = folds[folds.backbone == b]
        rows.append({"model": b, "n_patients": len(patients), "n_patches": len(y),
                     "auc": e["auc"], "auc_lo": ci["auc"][0], "auc_hi": ci["auc"][1],
                     "acc": e["acc"], "acc_lo": ci["acc"][0], "acc_hi": ci["acc"][1],
                     "sens": e["sens"], "sens_lo": ci["sens"][0], "sens_hi": ci["sens"][1],
                     "spec": e["spec"], "spec_lo": ci["spec"][0], "spec_hi": ci["spec"][1],
                     "tp": e["tp"], "fp": e["fp"], "tn": e["tn"], "fn": e["fn"],
                     "ece": ece_mce(y.astype(float), score[b])[0],
                     "fold_aucs": "/".join(f"{v:.4f}" for v in f.auc), "fold_sd": f.auc.std(ddof=1),
                     "thresholds": "/".join(f"{v:.3f}" for v in f.threshold)})
    paired = []
    for (a, b), values in contrast.items():
        lo, hi = np.percentile(values, [2.5, 97.5])
        paired.append({"contrast": f"{a} minus {b}", "auc_diff": estimate[a]["auc"] - estimate[b]["auc"],
                       "lo": lo, "hi": hi, "verdict": verdict(lo, hi)})
    return pd.DataFrame(rows), pd.DataFrame(paired)


def table_vi() -> pd.DataFrame:
    rows = []
    for backbone in BACKBONES:
        for res in RESOLUTIONS:
            unrestricted, hard = unrestricted_aucs(backbone, res), hard_aucs(backbone, res)
            rows.append({"backbone": backbone, "resolution": res,
                         "unrestricted_auc": np.mean(unrestricted), "hard_auc": np.mean(hard),
                         "change": np.mean(hard) - np.mean(unrestricted),
                         "unrestricted_min": min(unrestricted), "hard_max": max(hard),
                         "every_hard_run_below": max(hard) < min(unrestricted)})
    return pd.DataFrame(rows)


def main():
    BACKBONE_RESULTS.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    fmt = lambda v: f"{v:.4f}"

    table = table_vi()
    table.to_csv(BACKBONE_RESULTS / "table_vi.csv", index=False)
    print(table.to_string(index=False, float_format=fmt))

    for res, ref in COMPARISONS:
        result = comparison(res, ref)
        result.to_csv(BACKBONE_RESULTS / f"g4_{res}_vs_{ref}.csv", index=False)
        print(f"\nhard negatives at {res} against {ref}")
        print(result.to_string(index=False, float_format=fmt))

    pooled, paired = five_fold()
    pooled.to_csv(BACKBONE_RESULTS / "g4_kfold_224_pooled.csv", index=False)
    paired.to_csv(BACKBONE_RESULTS / "g4_kfold_224_paired.csv", index=False)
    print("\nhard negatives, five folds at 224")
    print(pooled.T.to_string())
    print(paired.to_string(index=False, float_format=fmt))
    print(f"\nwrote {BACKBONE_RESULTS}")


if __name__ == "__main__":
    main()
