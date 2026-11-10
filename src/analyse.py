import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from paths import RESULTS, RUNS

N_BOOT = 2000
SEED = 42
CONSTRUCTION_ARMS = ("G0", "G1", "G2", "G3", "G4")
CONSTRUCTION_TAG = "{arm}_B0_128_single"


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


def _auc(d):
    return roc_auc_score(d.label, d.score) if d.label.nunique() > 1 else float("nan")


def _sweep(val: pd.DataFrame, grid: np.ndarray):
    acc = np.array([((val.score > t).astype(int) == val.label).mean() for t in grid])
    hits = grid[acc >= acc.max() - 1e-12]
    return float(hits[0]), float((hits[0] + hits[-1]) / 2)


def threshold_rules(val: pd.DataFrame):
    fixed = np.linspace(0.05, 0.95, 91)
    adaptive = np.unique(np.quantile(val.score, np.linspace(0, 1, 2001)))
    low_fixed, mid_fixed = _sweep(val, fixed)
    _, mid_adaptive = _sweep(val, adaptive)
    return {"fixed_0.5": 0.5,
            "original_lowest": low_fixed,
            "midpoint": mid_fixed,
            "adaptive_midpoint": mid_adaptive}


def metrics_at(d: pd.DataFrame, thr: float):
    p = (d.score > thr).astype(int)
    tp = int(((p == 1) & (d.label == 1)).sum())
    fp = int(((p == 1) & (d.label == 0)).sum())
    tn = int(((p == 0) & (d.label == 0)).sum())
    fn = int(((p == 0) & (d.label == 1)).sum())

    def ratio(a, b):
        return a / b if b else float("nan")

    return {"threshold": thr, "accuracy": ratio(tp + tn, len(d)),
            "sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
            "ppv": ratio(tp, tp + fp), "npv": ratio(tn, tn + fn),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def load_runs():
    rows = []
    for meta_path in sorted(RUNS.glob("*_run.json")):
        with open(meta_path) as fh:
            meta = json.load(fh)
        pred = pd.read_csv(RUNS / f"{meta['tag']}_predictions.csv")
        test, val = pred[pred.split == "test"], pred[pred.split == "val"]
        if not len(test):
            continue
        lo, hi = patient_boot(test, _auc)
        r = {"tag": meta["tag"], "arm": meta["arm"], "backbone": meta["backbone"],
             "resolution": meta["resolution"], "split": meta["split"], "fold": meta["fold"],
             "negative_rule": meta["negative_rule"],
             "n_test": len(test), "patients_test": test.patient_id.nunique(),
             "auc": _auc(test), "auc_lo": lo, "auc_hi": hi,
             "epochs": meta["epochs_ran_phase1"] + meta["epochs_ran_phase2"],
             "seconds": meta["seconds"]}
        for name, t in threshold_rules(val).items():
            for k, v in metrics_at(test, t).items():
                r[f"{name}__{k}"] = v
        rows.append(r)
    return pd.DataFrame(rows)


def construction_summary(runs: pd.DataFrame, probe: dict) -> list[dict]:
    by_tag = runs.set_index("tag")
    summary = []
    for arm in CONSTRUCTION_ARMS:
        tag = CONSTRUCTION_TAG.format(arm=arm)
        if tag not in by_tag.index or arm not in probe:
            continue
        r, p = by_tag.loc[tag], probe[arm]
        summary.append({
            "arm": arm, "tag": tag, "negative_rule": r["negative_rule"],
            "probe_extracted": p["probe_extracted"], "probe_source": p["probe_source"],
            "fabrication_gap": p["fabrication_gap"],
            "probe_frequency_only": p["extracted_frequency_only"],
            "probe_photometric_only": p["extracted_photometric_only"],
            "cnn_auc": r["auc"], "cnn_auc_lo": r["auc_lo"], "cnn_auc_hi": r["auc_hi"],
            "cnn_over_probe": r["auc"] - p["probe_extracted"],
        })
    return summary


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not len(runs):
        print("no runs found")
        return
    runs.to_csv(RESULTS / "arm_metrics.csv", index=False)

    print("CNN RUNS")
    cols = ["tag", "n_test", "patients_test", "auc", "auc_lo", "auc_hi",
            "midpoint__accuracy", "adaptive_midpoint__threshold",
            "adaptive_midpoint__accuracy", "adaptive_midpoint__sensitivity",
            "adaptive_midpoint__specificity"]
    print(runs[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    probe_path = RESULTS / "probe_gap.json"
    if probe_path.exists():
        with open(probe_path) as fh:
            probe = {p["arm"]: p for p in json.load(fh)}
        summary = construction_summary(runs, probe)
        if summary:
            with open(RESULTS / "arm_summary.json", "w") as fh:
                json.dump(summary, fh, indent=2)
            print("\nCONSTRUCTION CONDITIONS (statistics-only baseline and CNN)")
            print(pd.DataFrame(summary).drop(columns=["tag"])
                  .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nwrote {RESULTS / 'arm_metrics.csv'}")


if __name__ == "__main__":
    main()
