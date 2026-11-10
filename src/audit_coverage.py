import json
import sys

import pandas as pd

from paths import CENSUS, MANIFEST, RESULTS, SCAN_LOG

MIN_ANNOTATIONS = 3

CHECKS = (
    ("never_attempted", "eligible scans never attempted"),
    ("attempted_but_no_patches_unexplained", "scans that yielded nothing with no logged reason"),
    ("missing_patients", "eligible patients absent from the manifest"),
    ("unexpected_scans", "scans in the manifest that are not eligible"),
)


def audit(verbose: bool = True) -> dict:
    with open(CENSUS) as fh:
        census = json.load(fh)["scans"]
    log = pd.read_csv(SCAN_LOG)
    man = pd.read_csv(MANIFEST)

    eligible = [s for s in census
                if any(c["n_ann"] >= MIN_ANNOTATIONS for c in s["clusters"])]
    want_scans = {(s["patient_id"], int(s["scan_id"])) for s in eligible}
    want_patients = {s["patient_id"] for s in eligible}

    log_scans = set(zip(log.patient_id, log.scan_id.astype(int)))
    ok = log[log.status == "ok"]
    ok_scans = set(zip(ok.patient_id, ok.scan_id.astype(int)))
    failed = log[log.status != "ok"]

    have_scans = set(zip(man.patient_id, man.scan_id.astype(int)))
    have_patients = set(man.patient_id.unique())

    never_attempted = want_scans - log_scans
    attempted_no_patches = ok_scans - have_scans
    missing_patients = want_patients - have_patients
    unexpected = have_scans - want_scans

    failed_pairs = set(zip(failed.patient_id, failed.scan_id.astype(int)))
    unexplained = attempted_no_patches - failed_pairs

    result = {
        "eligible_scans": len(want_scans),
        "eligible_patients": len(want_patients),
        "scans_attempted": len(log_scans),
        "scans_ok": len(ok_scans),
        "scans_failed": len(failed),
        "scans_with_patches": len(have_scans),
        "patients_with_patches": len(have_patients),
        "patches_total": len(man),
        "patches_positive": int((man.label == 1).sum()),
        "patches_negative": int((man.label == 0).sum()),
        "never_attempted": sorted(never_attempted),
        "attempted_but_no_patches_unexplained": sorted(unexplained),
        "missing_patients": sorted(missing_patients),
        "unexpected_scans": sorted(unexpected),
        "failure_reasons": failed.note.astype(str).tolist(),
    }
    result["complete"] = not (never_attempted or unexplained or missing_patients or unexpected)

    if verbose:
        print("COHORT COVERAGE AUDIT")
        print(f"  eligible (census)      {result['eligible_scans']:>5} scans  "
              f"{result['eligible_patients']:>4} patients")
        print(f"  attempted (scan log)   {result['scans_attempted']:>5} scans  "
              f"({result['scans_ok']} ok, {result['scans_failed']} failed)")
        print(f"  present (manifest)     {result['scans_with_patches']:>5} scans  "
              f"{result['patients_with_patches']:>4} patients")
        print(f"  patches                {result['patches_total']:>5}  "
              f"({result['patches_positive']} positive, {result['patches_negative']} negative)")
        print()
        for key, label in CHECKS:
            items = result[key]
            state = "OK  " if not items else "HOLE"
            print(f"  [{state}] {label}: {len(items)}")
            for item in items[:10]:
                print(f"           {item}")
            if len(items) > 10:
                print(f"           ... and {len(items) - 10} more")
        if result["failure_reasons"]:
            print("\n  logged failures:")
            for reason in result["failure_reasons"][:10]:
                print(f"    {reason}")
        print()
        print("  RESULT:", "COMPLETE, every eligible patient is represented"
              if result["complete"] else "INCOMPLETE, see the holes above")

    return result


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    result = audit()
    with open(RESULTS / "coverage_audit.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"  wrote {RESULTS / 'coverage_audit.json'}")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
