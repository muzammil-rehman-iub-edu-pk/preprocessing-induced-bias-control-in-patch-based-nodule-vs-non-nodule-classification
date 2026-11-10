import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from calibration import ece_mce
from paths import RESULTS, RUNS

N_BOOT = 2000
SEED = 42
SEEDS = (42, 1234, 2026)
ARMS = ("G0", "G1", "G2")
CONTRASTS = (("G0", "G1"), ("G1", "G2"))
KEY = ["patient_id", "scan_id", "label", "rule", "z", "y", "x"]


def run_tag(arm, seed):
    tag = f"{arm}_B0_128_single"
    return tag if seed == 42 else f"{tag}_seed{seed}"


def load():
    scores, ref = {}, None
    for arm in ARMS:
        for seed in SEEDS:
            tag = run_tag(arm, seed)
            pred = pd.read_csv(RUNS / f"{tag}_predictions.csv")
            test = pred[pred.split == "test"].reset_index(drop=True)
            if ref is None:
                ref = test
            elif not test[KEY].equals(ref[KEY]):
                raise ValueError(f"{tag} test rows differ from {run_tag(ARMS[0], SEEDS[0])}")
            scores[arm, seed] = test.score.to_numpy(float)
    return ref, scores


def contrasts(y, scores, ix):
    m = {k: (roc_auc_score(y[ix], s[ix]), ece_mce(y[ix], s[ix])[0]) for k, s in scores.items()}
    out = {}
    for a, b in CONTRASTS:
        for j, metric in enumerate(("auc", "ece")):
            d = [m[b, s][j] - m[a, s][j] for s in SEEDS]
            out[f"{a}_to_{b}", metric] = d + [float(np.mean(d))]
    return out


def main():
    ref, scores = load()
    y = ref.label.to_numpy(float)
    pats = ref.patient_id.unique()
    by = ref.groupby("patient_id").indices
    point = contrasts(y, scores, np.arange(len(y)))

    rng = np.random.default_rng(SEED)
    boot = {k: [] for k in point}
    for _ in range(N_BOOT):
        pick = rng.choice(pats, len(pats), replace=True)
        ix = np.concatenate([by[p] for p in pick])
        for k, v in contrasts(y, scores, ix).items():
            boot[k].append(v)

    rows = []
    for (name, metric), est in point.items():
        b = np.array(boot[name, metric])
        for j, seed in enumerate([str(s) for s in SEEDS] + ["mean"]):
            lo, hi = np.percentile(b[:, j], [2.5, 97.5])
            rows.append({"contrast": name, "metric": metric, "seed": seed,
                         "difference": est[j], "lo": float(lo), "hi": float(hi)})
    out = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS / "seed_contrast.csv", index=False)

    for arm in ARMS:
        auc = [roc_auc_score(y, scores[arm, s]) for s in SEEDS]
        ece = [ece_mce(y, scores[arm, s])[0] for s in SEEDS]
        print(f"{arm}  AUC {np.mean(auc):.4f} ({min(auc):.4f} to {max(auc):.4f})  "
              f"ECE {np.mean(ece):.4f} ({min(ece):.4f} to {max(ece):.4f})")
    print()
    print(out.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"\nwrote {RESULTS / 'seed_contrast.csv'}")


if __name__ == "__main__":
    main()
