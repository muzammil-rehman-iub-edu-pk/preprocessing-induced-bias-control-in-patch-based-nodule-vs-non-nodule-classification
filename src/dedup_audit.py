import hashlib
import json

import numpy as np
import pandas as pd
from scipy.fftpack import dct
from sklearn.metrics import roc_auc_score

from arms import build_patch
from folds import make_folds
from paths import MANIFEST, RESULTS
from probe import load_cohort

ARM, RENDER_PX, K, SEED = "G3", 128, 5, 42
HAMMING_MAX = 20
REMOVAL_THRESHOLDS = (0, 4, 8)
BITS = {"aHash(16x16)": 256, "dHash(17x16)": 272, "pHash(8x8 DCT)": 63}
KEY_COLS = ["patient_id", "scan_id", "z", "y", "x", "rule"]


def exact_hash(img: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(img, np.float32).tobytes(),
                           digest_size=16).hexdigest()


def _resize_mean(img: np.ndarray, n: int) -> np.ndarray:
    h, w = img.shape
    ys = (np.arange(h) * n // h)
    xs = (np.arange(w) * n // w)
    out = np.zeros((n, n), np.float64)
    cnt = np.zeros((n, n), np.float64)
    np.add.at(out, (ys[:, None], xs[None, :]), img)
    np.add.at(cnt, (ys[:, None], xs[None, :]), 1.0)
    return out / np.maximum(cnt, 1)


def ahash(img, n=16):
    b = _resize_mean(img, n)
    return (b > b.mean()).ravel()


def dhash(img, n=16):
    b = _resize_mean(img, n + 1)
    return (b[:, 1:] > b[:, :-1]).ravel()


def phash(img, n=8, size=32):
    b = _resize_mean(img, size)
    d = dct(dct(b, axis=0, norm="ortho"), axis=1, norm="ortho")[:n, :n]
    lo = d.ravel()[1:]
    return lo > np.median(lo)


def pack(bits: np.ndarray) -> np.ndarray:
    return np.packbits(bits.astype(np.uint8))


def hamming_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    lut = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
    out = np.zeros((len(A), len(B)), np.int16)
    for i in range(len(A)):
        out[i] = lut[np.bitwise_xor(A[i][None, :], B)].sum(1)
    return out


def split_indices(rows: pd.DataFrame, fold: dict):
    tr = rows.index[rows.patient_id.isin(fold["train"])].to_numpy()
    te = rows.index[rows.patient_id.isin(fold["test"])].to_numpy()
    return tr, te


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(MANIFEST)
    sel = man[(man.label == 1) | (man.rule == "lung")].reset_index(drop=True)
    print(f"auditing condition {ARM}: {len(sel)} patches, {sel.patient_id.nunique()} patients",
          flush=True)

    tiles, rows = load_cohort(sel)
    print("rendering patches ...", flush=True)
    imgs = np.stack([build_patch(tiles[i], rows.iloc[i], ARM, out_px=RENDER_PX)
                     for i in range(len(rows))])
    del tiles

    print("computing fingerprints ...", flush=True)
    exact = [exact_hash(im) for im in imgs]
    fingerprints = {
        "aHash(16x16)": np.stack([pack(ahash(im)) for im in imgs]),
        "dHash(17x16)": np.stack([pack(dhash(im)) for im in imgs]),
        "pHash(8x8 DCT)": np.stack([pack(phash(im)) for im in imgs]),
    }

    s = pd.Series(exact)
    dup_groups = s[s.duplicated(keep=False)].groupby(s).groups
    n_exact_pairs = sum(len(v) * (len(v) - 1) // 2 for v in dup_groups.values())
    print(f"\nexact content-hash duplicates: {len(dup_groups)} groups, "
          f"{n_exact_pairs} pairs, {int(s.duplicated().sum())} redundant patches")

    folds = make_folds(man, k=K, seed=SEED)
    sweep_rows, dist_rows = [], []
    for name, packed in fingerprints.items():
        for f in folds:
            tr, te = split_indices(rows, f)
            nearest = hamming_matrix(packed[te], packed[tr]).min(axis=1)
            for d in range(0, HAMMING_MAX + 1):
                sweep_rows.append({"fingerprint": name, "fold": f["fold"], "hamming": d,
                                   "test_patches_flagged": int((nearest <= d).sum()),
                                   "n_test": len(te)})
            for q in (0, 1, 5, 10, 25, 50):
                dist_rows.append({"fingerprint": name, "fold": f["fold"], "percentile": q,
                                  "nearest_hamming": float(np.percentile(nearest, q))})

    sweep = pd.DataFrame(sweep_rows)
    agg = (sweep.groupby(["fingerprint", "hamming"])
           .agg(flagged=("test_patches_flagged", "sum"), n_test=("n_test", "sum"))
           .reset_index())
    agg["pct_of_test"] = 100 * agg.flagged / agg.n_test
    agg["normalised_hamming"] = agg.hamming / agg.fingerprint.map(BITS)
    agg.to_csv(RESULTS / "dedup_sweep.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(RESULTS / "dedup_distance_distribution.csv", index=False)

    print("\nCROSS-SPLIT NEAR DUPLICATES, pooled over 5 folds")
    print(f"  {'fingerprint':16s} " + "".join(f"{d:>7d}" for d in (0, 2, 4, 6, 8, 10, 12)))
    print(f"  {'(Hamming <=)':16s} " + "-" * 49)
    for name in fingerprints:
        g = agg[agg.fingerprint == name].set_index("hamming")
        print(f"  {name:16s} " + "".join(f"{g.loc[d, 'pct_of_test']:6.2f}%"
                                         for d in (0, 2, 4, 6, 8, 10, 12)))
    print(f"\n  (percent of the {int(agg.n_test.iloc[0])} pooled test patches whose nearest "
          f"training patch is within that Hamming distance)")

    pooled = pd.read_csv(RESULTS / "headline_pooled_predictions.csv")
    key = rows.set_index(KEY_COLS).index

    print("\nEFFECT OF REMOVAL ON THE HEADLINE METRIC")
    base = roc_auc_score(pooled.label, pooled.score)
    print(f"  baseline pooled AUC                                {base:.4f}")
    effect = [{"condition": "baseline", "hamming": None, "n_removed": 0, "auc": base}]
    ph = fingerprints["pHash(8x8 DCT)"]
    for d in REMOVAL_THRESHOLDS:
        drop = set()
        for f in folds:
            tr, te = split_indices(rows, f)
            flagged = te[hamming_matrix(ph[te], ph[tr]).min(axis=1) <= d]
            drop.update(flagged.tolist())
        keys = {key[i] for i in drop}
        keep = ~pooled.set_index(KEY_COLS).index.isin(keys)
        kept = pooled[keep]
        a = roc_auc_score(kept.label, kept.score) if kept.label.nunique() > 1 else np.nan
        print(f"  drop test-side pHash matches at Hamming<={d:<2d}  n={len(drop):5d}  "
              f"AUC {a:.4f}  ({a - base:+.4f})")
        effect.append({"condition": "drop_test_side", "hamming": d,
                       "n_removed": len(drop), "auc": float(a)})

    pd.DataFrame(effect).to_csv(RESULTS / "dedup_effect.csv", index=False)
    with open(RESULTS / "dedup_audit.json", "w") as fh:
        json.dump({"exact_duplicate_groups": len(dup_groups),
                   "exact_duplicate_pairs": int(n_exact_pairs),
                   "redundant_patches": int(s.duplicated().sum()),
                   "effect": effect}, fh, indent=2)
    print(f"\nwrote {RESULTS / 'dedup_sweep.csv'}, {RESULTS / 'dedup_effect.csv'}, "
          f"{RESULTS / 'dedup_audit.json'}")


if __name__ == "__main__":
    main()
