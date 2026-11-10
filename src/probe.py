import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from arms import ARM_NEG_RULE, build_patch, reference_patch
from paths import MANIFEST, RESULTS, scan_file

FEATURES = ("mean", "std", "p05", "p95", "lap_abs", "lap_std",
            "grad_x", "grad_y", "frac0", "frac1")
FREQUENCY = ("lap_abs", "lap_std", "grad_x", "grad_y")
PHOTOMETRIC = ("mean", "std", "p05", "p95", "frac0", "frac1")

PROBE_PX = 128
N_BOOT = 1000
N_PERM = 1000
SEED = 42


def features_one(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    if a.shape[0] < 3 or a.shape[1] < 3:
        a = np.pad(a, 1, mode="edge")
    lap = (a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:])
    lo, hi = float(a.min()), float(a.max())
    rng = hi - lo + 1e-8
    return np.array([
        a.mean(), a.std(),
        np.percentile(a, 5), np.percentile(a, 95),
        np.abs(lap).mean(), lap.std(),
        np.abs(np.diff(a, axis=1)).mean(), np.abs(np.diff(a, axis=0)).mean(),
        float((a < lo + 0.02 * rng).mean()), float((a > hi - 0.02 * rng).mean()),
    ], dtype=np.float64)


def _classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def grouped_scores(F, y, g, cols=None):
    X = F if cols is None else F[:, cols]
    oof = np.zeros(len(y), dtype=float)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=g):
        model = _classifier().fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    return oof


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def boot_auc_ci(y, s, g, rng, n=N_BOOT):
    pats = np.unique(g)
    idx_by_pat = {p: np.where(g == p)[0] for p in pats}
    vals = []
    for _ in range(n):
        pick = rng.choice(pats, size=len(pats), replace=True)
        idx = np.concatenate([idx_by_pat[p] for p in pick])
        if len(np.unique(y[idx])) > 1:
            vals.append(auc(y[idx], s[idx]))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def perm_null_gap(Fe, Fs, y, g, rng, n=N_PERM):
    gaps = []
    idx_by_pat = {p: np.where(g == p)[0] for p in np.unique(g)}
    for _ in range(n):
        yp = y.copy()
        for _, idx in idx_by_pat.items():
            yp[idx] = rng.permutation(y[idx])
        if len(np.unique(yp)) < 2:
            continue
        gaps.append(auc(yp, grouped_scores(Fe, yp, g)) - auc(yp, grouped_scores(Fs, yp, g)))
    return np.array(gaps)


def load_cohort(manifest: pd.DataFrame):
    tiles = np.zeros((len(manifest), 224, 224), dtype=np.int16)
    pos = 0
    order = []
    for (pid, sid), grp in manifest.groupby(["patient_id", "scan_id"], sort=False):
        d = np.load(scan_file(pid, sid), allow_pickle=True)
        t = d["tiles"]
        meta = pd.DataFrame({k[5:]: d[k] for k in d.files if k.startswith("meta_")})
        key = meta.apply(lambda r: (int(r.z), int(r.y), int(r.x), str(r["rule"])), axis=1)
        lookup = {k: i for i, k in enumerate(key)}
        for _, r in grp.iterrows():
            i = lookup.get((int(r.z), int(r.y), int(r.x), str(r["rule"])))
            if i is None:
                raise KeyError(f"patch not found in cache: {pid} {sid} z{r.z}")
            tiles[pos] = t[i]
            order.append(r.name)
            pos += 1
    return tiles[:pos], manifest.loc[order].reset_index(drop=True)


def run_arm(arm, tiles, rows, rng, n_perm):
    sub = rows.index[(rows.label == 1) | (rows.rule == ARM_NEG_RULE[arm])].to_numpy()
    y = rows.label.to_numpy()[sub].astype(int)
    g = rows.patient_id.to_numpy()[sub]

    Fe = np.stack([features_one(build_patch(tiles[i], rows.iloc[i], arm, out_px=PROBE_PX))
                   for i in sub])
    Fs = np.stack([features_one(reference_patch(tiles[i], rows.iloc[i])) for i in sub])

    se, ss = grouped_scores(Fe, y, g), grouped_scores(Fs, y, g)
    ae, a_s = auc(y, se), auc(y, ss)

    cols_f = [FEATURES.index(c) for c in FREQUENCY]
    cols_p = [FEATURES.index(c) for c in PHOTOMETRIC]

    out = {
        "arm": arm, "negative_rule": ARM_NEG_RULE[arm],
        "n_patches": int(len(sub)), "n_patients": int(len(np.unique(g))),
        "probe_extracted": ae, "probe_extracted_ci": boot_auc_ci(y, se, g, rng),
        "probe_source": a_s, "probe_source_ci": boot_auc_ci(y, ss, g, rng),
        "fabrication_gap": ae - a_s,
        "extracted_frequency_only": auc(y, grouped_scores(Fe, y, g, cols_f)),
        "extracted_photometric_only": auc(y, grouped_scores(Fe, y, g, cols_p)),
        "source_frequency_only": auc(y, grouped_scores(Fs, y, g, cols_f)),
        "source_photometric_only": auc(y, grouped_scores(Fs, y, g, cols_p)),
    }
    if n_perm:
        null = perm_null_gap(Fe, Fs, y, g, rng, n=n_perm)
        out["null_gap_mean"] = float(null.mean())
        out["null_gap_p975"] = float(np.percentile(null, 97.5))
        out["gap_p_value"] = float(((null >= out["fabrication_gap"]).sum() + 1) / (len(null) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="G0,G1,G2,G3,G4")
    ap.add_argument("--perm", type=int, default=0)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(MANIFEST)
    print(f"loading {len(man)} tiles from {man.patient_id.nunique()} patients ...", flush=True)
    tiles, rows = load_cohort(man)
    print(f"loaded {tiles.shape}, {tiles.nbytes / 1e9:.2f} GB\n", flush=True)

    rng = np.random.default_rng(SEED)
    results = []
    header = (f"{'arm':4s} {'rule':5s} {'n':>6s} {'extracted':>19s} {'source':>19s} "
              f"{'gap':>8s}  {'freq':>6s} {'photo':>6s}")
    print(header)
    print("-" * len(header), flush=True)
    for arm in args.arms.split(","):
        r = run_arm(arm, tiles, rows, rng, args.perm)
        results.append(r)
        ce, cs = r["probe_extracted_ci"], r["probe_source_ci"]
        print(f"{r['arm']:4s} {r['negative_rule']:5s} {r['n_patches']:6d} "
              f"{r['probe_extracted']:.4f} [{ce[0]:.4f},{ce[1]:.4f}] "
              f"{r['probe_source']:.4f} [{cs[0]:.4f},{cs[1]:.4f}] "
              f"{r['fabrication_gap']:+8.4f}  "
              f"{r['extracted_frequency_only']:.4f} {r['extracted_photometric_only']:.4f}",
              flush=True)

    with open(RESULTS / "probe_gap.json", "w") as fh:
        json.dump(results, fh, indent=2)
    pd.DataFrame(results).to_csv(RESULTS / "probe_gap.csv", index=False)
    print(f"\nwrote {RESULTS / 'probe_gap.json'}, {RESULTS / 'probe_gap.csv'}")


if __name__ == "__main__":
    main()
