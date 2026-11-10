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

from arms import build_patch, reference_patch
from paths import MANIFEST, RESULTS
from probe import PROBE_PX, features_one, load_cohort

DIAM_BINS = [0, 6, 8, 10, 15, 1000]
DIAM_LABELS = ["<6mm", "6-8mm", "8-10mm", "10-15mm", ">=15mm"]


def diameter_band(series: pd.Series) -> pd.Series:
    return pd.cut(series, DIAM_BINS, labels=DIAM_LABELS, right=False)


def probe_within_strata():
    man = pd.read_csv(MANIFEST)
    tiles, rows = load_cohort(man)

    diam = rows.diameter_mm.copy()
    by_scan = rows[rows.label == 1].groupby(["patient_id", "scan_id"]).diameter_mm.median()
    for i in rows.index[rows.label == 0]:
        key = (rows.at[i, "patient_id"], rows.at[i, "scan_id"])
        if key in by_scan.index:
            diam.at[i] = by_scan.loc[key]
    rows = rows.assign(diam_stratum=diameter_band(diam))

    out = []
    for stratum in DIAM_LABELS:
        sel = rows.index[(rows.diam_stratum == stratum) &
                         ((rows.label == 1) | (rows.rule == "any"))].to_numpy()
        y = rows.label.to_numpy()[sel].astype(int)
        g = rows.patient_id.to_numpy()[sel]
        if len(np.unique(y)) < 2 or len(np.unique(g)) < 6 or len(sel) < 100:
            continue
        Fe = np.stack([features_one(build_patch(tiles[i], rows.iloc[i], "G0", out_px=PROBE_PX))
                       for i in sel])
        Fs = np.stack([features_one(reference_patch(tiles[i], rows.iloc[i])) for i in sel])

        def oof(F):
            s = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=5).split(F, y, groups=g):
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=3000)).fit(F[tr], y[tr])
                s[te] = m.predict_proba(F[te])[:, 1]
            return roc_auc_score(y, s)

        extracted, source = oof(Fe), oof(Fs)
        out.append({"stratum": stratum, "n": int(len(sel)),
                    "patients": int(len(np.unique(g))),
                    "probe_extracted": extracted, "probe_source": source,
                    "fabrication_gap": extracted - source})
    return pd.DataFrame(out)


def resample_factor_check():
    man = pd.read_csv(MANIFEST)
    sel = man[(man.label == 1) | (man.rule == "any")]
    pos, neg = sel[sel.label == 1], sel[sel.label == 0]
    rows = []
    for name, d in (("positives", pos), ("negatives", neg)):
        rf = d.legacy_resample_factor
        rows.append({"class": name, "n": len(d), "rf_median": rf.median(),
                     "rf_min": rf.min(), "rf_max": rf.max(), "rf_unique": rf.nunique()})
    per_stratum = []
    for s, g in pos.groupby(diameter_band(pos.diameter_mm)):
        if len(g) < 30:
            continue
        per_stratum.append({"stratum": str(s), "n_pos": len(g),
                            "pos_rf_median": g.legacy_resample_factor.median(),
                            "pos_rf_min": g.legacy_resample_factor.min(),
                            "neg_rf": 1.0,
                            "overlap_with_negatives":
                                float((g.legacy_resample_factor <= 1.0).mean())})
    return pd.DataFrame(rows), pd.DataFrame(per_stratum)


def strata_counts(gap: pd.DataFrame) -> pd.DataFrame:
    man = pd.read_csv(MANIFEST)
    pos = man[man.label == 1]
    nodules = (pos.groupby(["patient_id", "scan_id", "nodule_idx"])
               .diameter_mm.first().reset_index())
    n_nodules = nodules.groupby(diameter_band(nodules.diameter_mm)).size()
    n_positive = pos.groupby(diameter_band(pos.diameter_mm)).size()
    out = pd.DataFrame({"stratum": DIAM_LABELS,
                        "nodules": [int(n_nodules.get(s, 0)) for s in DIAM_LABELS],
                        "positive_patches": [int(n_positive.get(s, 0)) for s in DIAM_LABELS]})
    return out.merge(gap[["stratum", "n", "patients", "fabrication_gap"]]
                     .rename(columns={"n": "probe_patches", "patients": "probe_patients"}),
                     on="stratum", how="left")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("Resample factor by class, legacy construction\n")
    by_class, by_stratum = resample_factor_check()
    print(by_class.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  positives by diameter stratum (negatives are always exactly 1.0):")
    print(by_stratum.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    by_class.to_csv(RESULTS / "resample_factor_by_class.csv", index=False)
    by_stratum.to_csv(RESULTS / "resample_factor_by_stratum.csv", index=False)

    print("\nFabrication gap within nodule-diameter strata\n")
    gap = probe_within_strata()
    gap.to_csv(RESULTS / "gap_within_diameter_strata.csv", index=False)
    print(gap.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    counts = strata_counts(gap)
    counts.to_csv(RESULTS / "strata_counts.csv", index=False)
    print("\nNodules and patches per stratum\n")
    print(counts.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    with open(RESULTS / "size_conditioned.json", "w") as fh:
        json.dump({"within_strata": gap.to_dict("records"),
                   "resample_by_class": by_class.to_dict("records"),
                   "strata_counts": counts.to_dict("records")}, fh, indent=2)
    print(f"\nwrote {RESULTS / 'gap_within_diameter_strata.csv'}, "
          f"{RESULTS / 'strata_counts.csv'}, {RESULTS / 'size_conditioned.json'}")


if __name__ == "__main__":
    main()
