import json

import numpy as np
import pandas as pd

from extract import LUNG_FRACTION, MIN_ANNOTATIONS, SLICES_PER_NODULE
from paths import CENSUS, MANIFEST, RESULTS

DIAM_BINS = [0, 6, 8, 10, 15, np.inf]
DIAM_LABELS = ["<6mm", "6-8mm", "8-10mm", "10-15mm", ">=15mm"]


def collection_counts(census: list[dict]) -> dict:
    per_patient = pd.Series([s["patient_id"] for s in census]).value_counts()
    eligible = [s for s in census if any(c["n_ann"] >= MIN_ANNOTATIONS for c in s["clusters"])]
    eligible_patients = pd.Series([s["patient_id"] for s in eligible]).value_counts()
    return {
        "scans": len(census),
        "patients": int(len(per_patient)),
        "patients_with_two_scans": int((per_patient > 1).sum()),
        "annotation_clusters": sum(len(s["clusters"]) for s in census),
        "qualifying_nodules": sum(1 for s in census for c in s["clusters"]
                                  if c["n_ann"] >= MIN_ANNOTATIONS),
        "eligible_scans": len(eligible),
        "eligible_patients": int(len(eligible_patients)),
        "eligible_patients_with_two_scans": int((eligible_patients > 1).sum()),
    }


def nodule_table(man: pd.DataFrame) -> pd.DataFrame:
    pos = man[man.label == 1]
    return (pos.groupby(["patient_id", "scan_id", "nodule_idx"])
            .agg(diameter_mm=("diameter_mm", "first"), slices=("z", "size"))
            .reset_index())


def main():
    with open(CENSUS) as fh:
        census = json.load(fh)["scans"]
    man = pd.read_csv(MANIFEST)
    pos, neg = man[man.label == 1], man[man.label == 0]
    nodules = nodule_table(man)
    scans = man.groupby(["patient_id", "scan_id"]).agg(
        slice_thickness=("slice_thickness", "first"),
        pixel_spacing=("pixel_spacing", "first")).reset_index()

    slices_per_nodule = nodules.slices.value_counts().sort_index()
    fail_mask = pos.lung_fraction_window < LUNG_FRACTION
    band = pd.cut(nodules.diameter_mm, DIAM_BINS, labels=DIAM_LABELS, right=False)
    band_pos = pd.cut(pos.diameter_mm, DIAM_BINS, labels=DIAM_LABELS, right=False)

    stats = {
        "collection": collection_counts(census),
        "nodule_diameter_mm": {
            "median": float(nodules.diameter_mm.median()),
            "q1": float(nodules.diameter_mm.quantile(0.25)),
            "q3": float(nodules.diameter_mm.quantile(0.75)),
            "fraction_below_10mm": float((nodules.diameter_mm < 10).mean()),
        },
        "slice_thickness_mm": {
            "median": float(scans.slice_thickness.median()),
            "min": float(scans.slice_thickness.min()),
            "max": float(scans.slice_thickness.max()),
        },
        "pixel_spacing_mm": {
            "median": float(scans.pixel_spacing.median()),
            "min": float(scans.pixel_spacing.min()),
            "max": float(scans.pixel_spacing.max()),
        },
        "patches": {
            "nodules": int(len(nodules)),
            "theoretical_maximum_positives": int(len(nodules) * SLICES_PER_NODULE),
            "positives": int(len(pos)),
            "nodules_by_slices_obtained": {str(int(k)): int(v)
                                           for k, v in slices_per_nodule.items()},
            "nodules_with_fewer_than_3_slices": int((nodules.slices < SLICES_PER_NODULE).sum()),
            "negatives_per_rule": {r: int((neg.rule == r).sum()) for r in ("any", "lung", "hard")},
            "total": int(len(man)),
        },
        "positive_windows_failing_lung_rule": {
            "count": int(fail_mask.sum()),
            "fraction": float(fail_mask.mean()),
            "threshold": LUNG_FRACTION,
        },
        "diameter_bands": [
            {"band": str(b), "nodules": int((band == b).sum()),
             "positive_patches": int((band_pos == b).sum())}
            for b in DIAM_LABELS
        ],
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "cohort_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    c = stats["collection"]
    d = stats["nodule_diameter_mm"]
    t = stats["slice_thickness_mm"]
    s = stats["pixel_spacing_mm"]
    p = stats["patches"]
    f = stats["positive_windows_failing_lung_rule"]
    print(f"collection: {c['scans']} scans, {c['patients']} patients, "
          f"{c['patients_with_two_scans']} with two scans")
    print(f"qualifying nodules: {c['qualifying_nodules']} in {c['eligible_scans']} scans "
          f"from {c['eligible_patients']} patients")
    print(f"nodule diameter: median {d['median']:.2f} mm, IQR {d['q1']:.2f} to {d['q3']:.2f}, "
          f"{100 * d['fraction_below_10mm']:.1f}% below 10 mm")
    print(f"slice thickness: median {t['median']:.1f} mm, range {t['min']:.1f} to {t['max']:.1f}")
    print(f"pixel spacing: median {s['median']:.3f} mm, range {s['min']:.3f} to {s['max']:.3f}")
    print(f"positives: {p['positives']} of a possible {p['theoretical_maximum_positives']}; "
          f"{p['nodules_with_fewer_than_3_slices']} nodules on fewer than 3 slices "
          f"{p['nodules_by_slices_obtained']}")
    print(f"negatives per rule: {p['negatives_per_rule']}; total patches {p['total']}")
    print(f"positive windows below the {f['threshold']:.0%} lung rule: {f['count']} "
          f"({100 * f['fraction']:.1f}%)")
    for row in stats["diameter_bands"]:
        print(f"  band {row['band']:8s} nodules {row['nodules']:5d}  "
              f"positive patches {row['positive_patches']:5d}")
    print(f"wrote {RESULTS / 'cohort_stats.json'}")


if __name__ == "__main__":
    main()
