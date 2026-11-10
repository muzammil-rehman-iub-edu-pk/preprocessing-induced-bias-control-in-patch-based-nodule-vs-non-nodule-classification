import json

import numpy as np
import pandas as pd

from paths import RESULTS, RUNS

N_BINS = 15
THRESHOLD_COLS = ["tag", "auc", "fixed_0.5__accuracy", "original_lowest__accuracy",
                  "midpoint__accuracy", "adaptive_midpoint__accuracy",
                  "adaptive_midpoint__threshold"]


def ece_mce(y, p, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    ece, mce, rows = 0.0, 0.0, []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc, w = p[m].mean(), y[m].mean(), m.mean()
        gap = abs(acc - conf)
        ece += w * gap
        mce = max(mce, gap)
        rows.append({"bin": b, "lo": edges[b], "hi": edges[b + 1], "n": int(m.sum()),
                     "mean_score": float(conf), "observed_rate": float(acc)})
    return float(ece), float(mce), rows


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def analyse_run(tag: str):
    pred = pd.read_csv(RUNS / f"{tag}_predictions.csv")
    test = pred[pred.split == "test"]
    y = test.label.to_numpy().astype(float)
    p = test.score.to_numpy().astype(float)
    ece, mce, bins = ece_mce(y, p)
    return {
        "tag": tag,
        "brier": brier(y, p), "ece": ece, "mce": mce,
        "score_min": float(p.min()), "score_max": float(p.max()),
        "score_range": float(p.max() - p.min()),
        "score_median_neg": float(np.median(p[y == 0])),
        "score_median_pos": float(np.median(p[y == 1])),
        "frac_above_0.95": float((p > 0.95).mean()),
        "bins": bins,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    tags = sorted(m.stem.replace("_run", "") for m in RUNS.glob("*_run.json"))
    if not tags:
        print("no runs found")
        return

    results = [analyse_run(t) for t in tags]
    flat = pd.DataFrame([{k: v for k, v in r.items() if k != "bins"} for r in results])

    metrics_path = RESULTS / "arm_metrics.csv"
    if metrics_path.exists():
        m = pd.read_csv(metrics_path)
        keep = [c for c in THRESHOLD_COLS if c in m.columns]
        flat = flat.merge(m[keep], on="tag", how="left")

    flat.to_csv(RESULTS / "calibration.csv", index=False)
    with open(RESULTS / "calibration.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print("CALIBRATION (test split)")
    cols = [c for c in ["tag", "auc", "brier", "ece", "mce", "score_range",
                        "frac_above_0.95"] if c in flat.columns]
    print(flat[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    thr_cols = [c for c in flat.columns if c.endswith("__accuracy")]
    if thr_cols:
        print("\nTHRESHOLD RULES (test accuracy)")
        print(flat[["tag"] + thr_cols].to_string(index=False,
                                                 float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {RESULTS / 'calibration.csv'}, {RESULTS / 'calibration.json'}")


if __name__ == "__main__":
    main()
