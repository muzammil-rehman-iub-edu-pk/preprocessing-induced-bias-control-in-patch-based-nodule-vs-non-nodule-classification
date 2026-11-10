import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from paths import RESULTS, RUNS

N_BOOT = 2000
SEED = 42
ARM, BACKBONE, RES, K = "G3", "B0", 224, 5
PREVALENCES = [0.5, 0.10, 0.01, 0.001, 0.0001]


def adaptive_midpoint(val: pd.DataFrame) -> float:
    grid = np.unique(np.quantile(val.score, np.linspace(0, 1, 2001)))
    acc = np.array([((val.score > t).astype(int) == val.label).mean() for t in grid])
    hits = grid[acc >= acc.max() - 1e-12]
    return float((hits[0] + hits[-1]) / 2)


def point_metrics(d, thr):
    p = (d.score > thr).astype(int)
    tp = int(((p == 1) & (d.label == 1)).sum())
    fp = int(((p == 1) & (d.label == 0)).sum())
    tn = int(((p == 0) & (d.label == 0)).sum())
    fn = int(((p == 0) & (d.label == 1)).sum())

    def ratio(a, b):
        return a / b if b else float("nan")

    return {"accuracy": ratio(tp + tn, len(d)), "sensitivity": ratio(tp, tp + fn),
            "specificity": ratio(tn, tn + fp), "ppv": ratio(tp, tp + fp),
            "npv": ratio(tn, tn + fn), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def load_folds():
    per_fold, pooled = [], []
    for f in range(1, K + 1):
        tag = f"{ARM}_{BACKBONE}_{RES}_kfold_fold{f}"
        with open(RUNS / f"{tag}_run.json") as fh:
            meta = json.load(fh)
        pred = pd.read_csv(RUNS / f"{tag}_predictions.csv")
        val, test = pred[pred.split == "val"], pred[pred.split == "test"]
        thr = adaptive_midpoint(val)

        t = test.copy()
        t["fold"] = f
        t["threshold"] = thr
        pooled.append(t)
        per_fold.append({"fold": f, "auc": meta["test_auc"], "threshold": thr,
                         "n": len(test), "patients": test.patient_id.nunique(),
                         "seconds": meta["seconds"], **point_metrics(test, thr)})
    return pd.DataFrame(per_fold), pd.concat(pooled, ignore_index=True)


def patient_boot(df, fn, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    pats = df.patient_id.unique()
    by = {p: g.index.to_numpy() for p, g in df.groupby("patient_id")}
    vals = []
    for _ in range(n):
        pick = rng.choice(pats, len(pats), replace=True)
        v = fn(df.loc[np.concatenate([by[p] for p in pick])])
        if v == v:
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bayes_ppv_npv(sens, spec, prev):
    tp, fn = sens * prev, (1 - sens) * prev
    tn, fp = spec * (1 - prev), (1 - spec) * (1 - prev)

    def ratio(a, b):
        return a / b if b else float("nan")

    return ratio(tp, tp + fp), ratio(tn, tn + fn)


def pooled_auc(d):
    return roc_auc_score(d.label, d.score) if d.label.nunique() > 1 else float("nan")


def pooled_accuracy(d):
    return (d.pred == d.label).mean()


def pooled_sensitivity(d):
    return d.pred[d.label == 1].mean() if (d.label == 1).any() else np.nan


def pooled_specificity(d):
    return 1 - d.pred[d.label == 0].mean() if (d.label == 0).any() else np.nan


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    per_fold, pooled = load_folds()

    print(f"HEADLINE PROTOCOL: {ARM}, EfficientNet{BACKBONE} at {RES}, {K}-fold patient-level\n")
    cols = ["fold", "n", "patients", "auc", "threshold", "accuracy", "sensitivity", "specificity"]
    print(per_fold[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    a = per_fold.auc.values
    print(f"\n  mean AUC {a.mean():.3f}   sample SD (ddof=1) {a.std(ddof=1):.3f}")

    auc_pooled = roc_auc_score(pooled.label, pooled.score)
    lo, hi = patient_boot(pooled, pooled_auc)
    print(f"  pooled out-of-fold AUC {auc_pooled:.3f}  95% CI [{lo:.3f}, {hi:.3f}]"
          f"   ({len(pooled)} patches, {pooled.patient_id.nunique()} patients)")

    pooled["pred"] = (pooled.score > pooled.threshold).astype(int)
    pm = point_metrics(pooled.assign(score=pooled.pred + 0.0), 0.5)
    prev = float(pooled.label.mean())
    intervals = {}
    for name, fn in (("accuracy", pooled_accuracy), ("sensitivity", pooled_sensitivity),
                     ("specificity", pooled_specificity)):
        intervals[name] = patient_boot(pooled, fn)
        print(f"  {name:12s} {pm[name]:.3f}  95% CI [{intervals[name][0]:.3f}, "
              f"{intervals[name][1]:.3f}]")
    print(f"  confusion: TP {pm['tp']}  FP {pm['fp']}  TN {pm['tn']}  FN {pm['fn']}")

    print(f"\nPREVALENCE. Evaluated at pool prevalence {prev:.3f}, which is not a")
    print("screening prevalence. Sensitivity and specificity transfer; PPV and NPV are")
    print("recomputed by Bayes' rule at each assumed prevalence:")
    print(f"  {'assumed prevalence':>20s}  {'PPV':>7s}  {'NPV':>7s}")
    rows = []
    for p in PREVALENCES:
        ppv, npv = bayes_ppv_npv(pm["sensitivity"], pm["specificity"], p)
        label = f"{p:.4f}" + ("  (pool)" if abs(p - prev) < 1e-6 else "")
        print(f"  {label:>20s}  {ppv:7.4f}  {npv:7.4f}")
        rows.append({"prevalence": p, "ppv": ppv, "npv": npv})

    per_fold.to_csv(RESULTS / "headline_per_fold.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULTS / "headline_prevalence.csv", index=False)
    with open(RESULTS / "headline.json", "w") as fh:
        json.dump({"arm": ARM, "backbone": BACKBONE, "resolution": RES, "k": K,
                   "mean_auc": float(a.mean()), "sd_auc_ddof1": float(a.std(ddof=1)),
                   "pooled_auc": float(auc_pooled), "pooled_auc_ci": [lo, hi],
                   "pool_prevalence": prev, "pooled_metrics": pm,
                   "pooled_metric_ci": {k: list(v) for k, v in intervals.items()},
                   "n_patches": int(len(pooled)),
                   "n_patients": int(pooled.patient_id.nunique())}, fh, indent=2)
    pooled.to_csv(RESULTS / "headline_pooled_predictions.csv", index=False)
    print(f"\nwrote {RESULTS / 'headline.json'}, headline_per_fold.csv, "
          f"headline_prevalence.csv, headline_pooled_predictions.csv")


if __name__ == "__main__":
    main()
