import json

import pandas as pd

from paths import CENSUS, MANIFEST, RESULTS, SCAN_LOG

MIN_ANNOTATIONS = 3
ARM_RULES = (("G0", "any"), ("G1", "any"), ("G2", "any"), ("G3", "lung"), ("G4", "hard"))


def build():
    with open(CENSUS) as fh:
        census = json.load(fh)["scans"]
    man = pd.read_csv(MANIFEST)
    log = pd.read_csv(SCAN_LOG)

    n_scans = len(census)
    patient_counts = pd.Series([s["patient_id"] for s in census]).value_counts()
    n_patients = len(patient_counts)
    dual = int((patient_counts > 1).sum())
    clusters = sum(len(s["clusters"]) for s in census)
    qualifying = sum(1 for s in census for c in s["clusters"] if c["n_ann"] >= MIN_ANNOTATIONS)
    elig_scans = [s for s in census if any(c["n_ann"] >= MIN_ANNOTATIONS for c in s["clusters"])]
    elig_patients = len({s["patient_id"] for s in elig_scans})

    ok = log[log.status == "ok"]
    err = log[log.status == "error"]

    pos, neg = man[man.label == 1], man[man.label == 0]
    flow = [
        ("TCIA collection: scans", n_scans, ""),
        ("TCIA collection: patients", n_patients, f"{dual} contributed 2 retained scans"),
        ("Annotation clusters (any reader count)", clusters, ""),
        (f"Clusters with >={MIN_ANNOTATIONS} annotations", qualifying, "inclusion criterion"),
        ("Eligible scans", len(elig_scans), ">=1 qualifying cluster"),
        ("Eligible patients", elig_patients, ""),
        ("Scans extracted successfully", len(ok), ""),
        ("Scans failed", len(err), "; ".join(err.note.astype(str).head(3)) if len(err) else "none"),
        ("Nodules contributing patches",
         int(pos.groupby(["patient_id", "scan_id", "nodule_idx"]).ngroups), ""),
        ("Positive patches", len(pos), "<= 3 slices per nodule"),
        ("Negative patches (all rules)", len(neg), ""),
        ("  rule=any (legacy)", int((neg.rule == "any").sum()), "uniform, no anatomical constraint"),
        ("  rule=lung", int((neg.rule == "lung").sum()), ">=80% of the 64 mm window in parenchyma"),
        ("  rule=hard", int((neg.rule == "hard").sum()), "in-parenchyma, structure-centred"),
        ("Total patches", len(man), ""),
        ("Patients represented", man.patient_id.nunique(), ""),
    ]

    per_arm = []
    for arm, rule in ARM_RULES:
        sel = man[(man.label == 1) | (man.rule == rule)]
        per_arm.append({
            "arm": arm, "negative_rule": rule,
            "positives": int((sel.label == 1).sum()),
            "negatives": int((sel.label == 0).sum()),
            "patients": int(sel.patient_id.nunique()),
            "total": len(sel),
        })

    quality = {
        "positives_with_clipped_bbox": int(pos.bbox_clipped.sum()),
        "patches_touching_slice_edge": int((man.tile_pad_fraction > 0).sum()),
        "nodules_yielding_fewer_than_3_slices": int(ok.slices_short.sum()),
        "legacy_resample_factor_positives_median": float(pos.legacy_resample_factor.median()),
        "legacy_resample_factor_positives_iqr": [
            float(pos.legacy_resample_factor.quantile(.25)),
            float(pos.legacy_resample_factor.quantile(.75))],
        "legacy_resample_factor_negatives": float(neg.legacy_resample_factor.median()),
        "pixel_spacing_range": [float(man.pixel_spacing.min()), float(man.pixel_spacing.max())],
    }
    return flow, per_arm, quality


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    flow, per_arm, quality = build()

    print("\nDATA FLOW")
    for label, n, note in flow:
        print(f"  {label:45s} {n:>7}   {note}")
    print("\nPER CONDITION")
    for r in per_arm:
        print(f"  {r['arm']}  neg={r['negative_rule']:5s} pos={r['positives']:6d} "
              f"neg={r['negatives']:6d} patients={r['patients']:4d} total={r['total']:6d}")
    print("\nQUALITY")
    for key, value in quality.items():
        print(f"  {key:48s} {value}")

    with open(RESULTS / "consort.json", "w") as fh:
        json.dump({"flow": [{"stage": a, "n": b, "note": c} for a, b, c in flow],
                   "per_arm": per_arm, "quality": quality}, fh, indent=2)
    pd.DataFrame(per_arm).to_csv(RESULTS / "consort_per_arm.csv", index=False)
    print(f"\nwrote {RESULTS / 'consort.json'}, {RESULTS / 'consort_per_arm.csv'}")


if __name__ == "__main__":
    main()
