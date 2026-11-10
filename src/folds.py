import json
from pathlib import Path

import numpy as np
import pandas as pd

from arms import ARM_NEG_RULE
from paths import MANIFEST, RESULTS

VAL_RATIO = 0.15
TEST_RATIO_SINGLE = 0.20
VAL_RATIO_SINGLE = 0.15


def make_folds(manifest: pd.DataFrame, k: int = 5, seed: int = 42) -> list[dict]:
    patients = np.array(sorted(manifest.patient_id.unique()))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(patients))
    chunks = np.array_split(order, k)

    folds = []
    for i in range(k):
        test_idx = chunks[i]
        rest = np.concatenate([chunks[j] for j in range(k) if j != i])
        rest = rng.permutation(rest)
        n_val = max(1, int(round(len(rest) * VAL_RATIO)))
        folds.append({
            "fold": i + 1,
            "test": sorted(patients[test_idx].tolist()),
            "val": sorted(patients[rest[:n_val]].tolist()),
            "train": sorted(patients[rest[n_val:]].tolist()),
        })
    return folds


def single_split(manifest: pd.DataFrame, seed: int = 42) -> dict:
    patients = np.array(sorted(manifest.patient_id.unique()))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(patients))
    n_test = max(1, int(round(len(patients) * TEST_RATIO_SINGLE)))
    n_val = max(1, int(round(len(patients) * VAL_RATIO_SINGLE)))
    return {
        "test": sorted(patients[order[:n_test]].tolist()),
        "val": sorted(patients[order[n_test:n_test + n_val]].tolist()),
        "train": sorted(patients[order[n_test + n_val:]].tolist()),
    }


def assemble_split(manifest: pd.DataFrame, patient_ids, arm: str = "G2") -> pd.DataFrame:
    known = set(manifest.patient_id.unique())
    missing = [p for p in patient_ids if p not in known]
    if missing:
        raise KeyError(
            f"{len(missing)} assigned patient(s) not in the manifest: {missing[:5]}"
            f"{' ...' if len(missing) > 5 else ''}. Refusing to assemble a split from a "
            f"subset of the requested patients."
        )
    if arm not in ARM_NEG_RULE:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARM_NEG_RULE)}")

    rule = ARM_NEG_RULE[arm]
    sel = manifest[manifest.patient_id.isin(list(patient_ids))]
    return sel[(sel.label == 1) | (sel.rule == rule)].reset_index(drop=True)


def fold_counts(manifest: pd.DataFrame, folds: list[dict]) -> pd.DataFrame:
    rows = []
    for f in folds:
        sel = manifest[manifest.patient_id.isin(f["test"])]
        pos = sel[sel.label == 1]
        per_rule = sel[sel.label == 0].groupby("rule").size()
        if per_rule.nunique() != 1:
            raise ValueError(f"fold {f['fold']}: negative counts differ by rule: {per_rule.to_dict()}")
        rows.append({
            "fold": str(f["fold"]),
            "patients": sel.patient_id.nunique(),
            "scans": len(sel[["patient_id", "scan_id"]].drop_duplicates()),
            "nodules": len(pos[["patient_id", "scan_id", "nodule_idx"]].drop_duplicates()),
            "positive_patches": len(pos),
            "negative_patches_per_rule": int(per_rule.iloc[0]),
        })
    counts = pd.DataFrame(rows)
    total = {"fold": "total", **counts.drop(columns="fold").sum().to_dict()}
    return pd.concat([counts, pd.DataFrame([total])], ignore_index=True)


def write_manifests(manifest: pd.DataFrame, out_dir: Path, k: int = 5, seed: int = 42):
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = make_folds(manifest, k=k, seed=seed)
    payload = {
        "seed": seed, "k": k,
        "n_patients": int(manifest.patient_id.nunique()),
        "folds": folds,
        "single_split": single_split(manifest, seed=seed),
    }
    (out_dir / "fold_manifest.json").write_text(json.dumps(payload, indent=2))
    fold_counts(manifest, folds).to_csv(out_dir / "fold_counts.csv", index=False)
    return payload


def main():
    man = pd.read_csv(MANIFEST)
    payload = write_manifests(man, RESULTS)
    print(f"{payload['n_patients']} patients, k={payload['k']}")
    for f in payload["folds"]:
        print(f"  fold {f['fold']}: train {len(f['train']):4d}  val {len(f['val']):3d}  "
              f"test {len(f['test']):3d}")
    s = payload["single_split"]
    print(f"  single split: train {len(s['train'])}  val {len(s['val'])}  test {len(s['test'])}")
    print(f"wrote {RESULTS / 'fold_manifest.json'} and {RESULTS / 'fold_counts.csv'}")


if __name__ == "__main__":
    main()
