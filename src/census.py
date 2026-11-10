import json
import time

import pylidc as pl

from paths import CENSUS


def scan_record(scan) -> dict:
    record = {
        "patient_id": str(scan.patient_id),
        "scan_id": int(scan.id),
        "slice_thickness": float(scan.slice_thickness) if scan.slice_thickness is not None else None,
        "pixel_spacing": float(scan.pixel_spacing) if scan.pixel_spacing is not None else None,
        "contrast_used": bool(getattr(scan, "contrast_used", False)),
    }
    try:
        clusters = scan.cluster_annotations(verbose=False)
    except Exception as exc:
        record["error"] = str(exc)[:200]
        clusters = []
    summary = []
    for cluster in clusters:
        try:
            diameters = [float(a.diameter) for a in cluster]
            malignancy = [int(a.malignancy) for a in cluster]
            subtlety = [int(a.subtlety) for a in cluster]
        except Exception:
            diameters, malignancy, subtlety = [], [], []
        summary.append({"n_ann": len(cluster), "diameters": diameters,
                        "malignancy": malignancy, "subtlety": subtlety})
    record["clusters"] = summary
    return record


def main():
    out = {"scans": [], "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    scans = pl.query(pl.Scan).all()
    print(f"total scans in database: {len(scans)}", flush=True)
    t0 = time.time()
    for i, scan in enumerate(scans):
        out["scans"].append(scan_record(scan))
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            remaining = elapsed / (i + 1) * (len(scans) - i - 1)
            print(f"{i + 1}/{len(scans)}  {elapsed:.0f}s  eta {remaining:.0f}s", flush=True)

    CENSUS.parent.mkdir(parents=True, exist_ok=True)
    with open(CENSUS, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {CENSUS}", flush=True)


if __name__ == "__main__":
    main()
