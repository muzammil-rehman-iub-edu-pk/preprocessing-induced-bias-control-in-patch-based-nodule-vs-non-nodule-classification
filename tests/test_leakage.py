import json

import numpy as np
import pandas as pd
import pytest

import arms as A
from extract import LUNG_FRACTION, MIN_ANNOTATIONS
from folds import assemble_split, fold_counts, make_folds, single_split
from paths import CACHE, CENSUS, MANIFEST, RESULTS

SHIPPED_MANIFEST = RESULTS / "patch_manifest.csv"


def cache_files():
    return sorted(CACHE.glob("*.npz"))


def first_scan_with_both_classes():
    for f in cache_files()[:50]:
        d = np.load(f, allow_pickle=True)
        meta = pd.DataFrame({k[5:]: d[k] for k in d.files if k.startswith("meta_")})
        if len(meta) and meta.label.nunique() == 2:
            return d["tiles"], meta
    pytest.skip("no cached scan with both classes")


@pytest.fixture(scope="module")
def man():
    for path in (MANIFEST, SHIPPED_MANIFEST):
        if path.exists():
            return pd.read_csv(path)
    pytest.skip("no manifest; run src/extract.py first")


def test_folds_are_patient_disjoint(man):
    folds = make_folds(man, k=5, seed=42)
    for i, fi in enumerate(folds):
        for j, fj in enumerate(folds):
            if i < j:
                assert not set(fi["test"]) & set(fj["test"])


def test_train_val_test_are_patient_disjoint(man):
    for f in make_folds(man, k=5, seed=42):
        assert not set(f["train"]) & set(f["val"])
        assert not set(f["train"]) & set(f["test"])
        assert not set(f["val"]) & set(f["test"])


def test_every_patient_appears_in_exactly_one_test_fold(man):
    seen = [p for f in make_folds(man, k=5, seed=42) for p in f["test"]]
    assert len(seen) == len(set(seen))
    assert set(seen) == set(man.patient_id.unique())


def test_single_split_is_patient_disjoint_and_complete(man):
    s = single_split(man, seed=42)
    assert not set(s["train"]) & set(s["val"])
    assert not set(s["train"]) & set(s["test"])
    assert not set(s["val"]) & set(s["test"])
    assert set(s["train"]) | set(s["val"]) | set(s["test"]) == set(man.patient_id.unique())


def test_dual_scan_patients_stay_whole(man):
    multi = man.groupby("patient_id").scan_id.nunique()
    multi = multi[multi > 1].index.tolist()
    if not multi:
        pytest.skip("no patient with two scans in this manifest")
    for f in make_folds(man, k=5, seed=42):
        for pid in multi:
            where = {name for name in ("train", "val", "test") if pid in set(f[name])}
            assert len(where) == 1


def test_split_assembly_raises_on_missing_patient(man):
    with pytest.raises(KeyError, match="not in the manifest"):
        assemble_split(man, ["LIDC-IDRI-9999"], arm="G2")


def test_split_assembly_returns_rows_for_present_patients(man):
    pid = man.patient_id.iloc[0]
    rows = assemble_split(man, [pid], arm="G2")
    assert len(rows) > 0
    assert set(rows.patient_id.unique()) == {pid}


def test_cache_stores_raw_hu_only():
    files = cache_files()
    if not files:
        pytest.skip("no cache")
    for f in files[:20]:
        t = np.load(f)["tiles"]
        if not len(t):
            continue
        assert t.dtype == np.int16
        assert t.min() < -500
        assert t.max() > 100


def test_fixed_intensity_map_is_sample_independent():
    a = np.array([[-1000, 0], [200, 400]], dtype=np.float32)
    b = np.array([[-800, -600], [-400, -200]], dtype=np.float32)
    fa, fb = A._hu_affine(a), A._hu_affine(b)
    assert np.isclose(fa[0, 0], 0.0) and np.isclose(fa[1, 1], 255.0)
    assert np.isclose(A._hu_affine(np.array([[-400.0]]))[0, 0], fb[1, 0])


def test_minmax_is_sample_dependent():
    a = np.array([[-1000.0, 400.0]])
    b = np.array([[-1000.0, -900.0]])
    assert np.isclose(A._minmax(a)[0, 1], 1.0)
    assert np.isclose(A._minmax(b)[0, 1], 1.0)


def test_probe_feature_list_is_fixed():
    from probe import FEATURES, FREQUENCY, PHOTOMETRIC
    assert FEATURES == ("mean", "std", "p05", "p95", "lap_abs", "lap_std",
                        "grad_x", "grad_y", "frac0", "frac1")
    assert set(FREQUENCY) | set(PHOTOMETRIC) == set(FEATURES)
    assert not set(FREQUENCY) & set(PHOTOMETRIC)


def test_reference_is_not_resampled_and_is_class_independent():
    tiles, meta = first_scan_with_both_classes()
    sizes = {}
    for label in (0, 1):
        i = meta.index[meta.label == label][0]
        ref = A.reference_patch(tiles[i], meta.iloc[i])
        expect = int(round(A.MATCHED_FOV_MM / float(meta.iloc[i]["pixel_spacing"])))
        assert ref.shape == (expect, expect)
        assert ref.min() < -500 < ref.max()
        sizes[label] = ref.shape
    assert sizes[0] == sizes[1]


def test_matched_conditions_share_geometry_across_classes():
    tiles, meta = first_scan_with_both_classes()
    ip = meta.index[meta.label == 1][0]
    ineg = meta.index[meta.label == 0][0]
    for arm in ("G1", "G2", "G3", "G4"):
        a = A.build_patch(tiles[ip], meta.iloc[ip], arm, out_px=128)
        b = A.build_patch(tiles[ineg], meta.iloc[ineg], arm, out_px=128)
        assert a.shape == b.shape == (128, 128)
    assert float(meta.iloc[ip]["legacy_resample_factor"]) != 1.0
    assert float(meta.iloc[ineg]["legacy_resample_factor"]) == 1.0


def test_fixed_map_output_range():
    tiles, meta = first_scan_with_both_classes()
    img = A.build_patch(tiles[0], meta.iloc[0], "G2", out_px=96)
    assert img.min() >= 0.0 and img.max() <= 255.0 + 1e-3


def test_negative_rules_respect_their_constraints(man):
    lung = man[(man.label == 0) & (man.rule == "lung")]
    hard = man[(man.label == 0) & (man.rule == "hard")]
    if len(lung):
        assert lung.lung_fraction_window.min() >= LUNG_FRACTION - 1e-6
    if len(hard):
        assert hard.lung_fraction_window.min() >= LUNG_FRACTION - 1e-6


def test_no_negative_shares_a_slice_with_a_nodule(man):
    for (pid, sid), grp in man.groupby(["patient_id", "scan_id"]):
        pz = set(grp[grp.label == 1].z.tolist())
        nz = set(grp[grp.label == 0].z.tolist())
        assert not pz & nz, f"{pid} scan {sid}: negative on a nodule slice"


def test_classes_are_balanced_per_rule(man):
    positives = int((man.label == 1).sum())
    for rule in ("any", "lung", "hard"):
        assert int(((man.label == 0) & (man.rule == rule)).sum()) == positives


def test_fold_counts_tile_the_cohort(man):
    counts = fold_counts(man, make_folds(man))
    folds, total = counts[counts.fold != "total"], counts[counts.fold == "total"].iloc[0]
    assert len(folds) == 5
    assert int(total.patients) == man.patient_id.nunique()
    assert int(total.scans) == len(man[["patient_id", "scan_id"]].drop_duplicates())
    assert int(total.positive_patches) == int((man.label == 1).sum())
    assert (folds.positive_patches == folds.negative_patches_per_rule).all()


def test_cohort_coverage_is_complete(man):
    if not MANIFEST.exists():
        pytest.skip("the coverage audit reads the extraction cache")
    from audit_coverage import audit
    result = audit(verbose=False)
    assert not result["never_attempted"]
    assert not result["attempted_but_no_patches_unexplained"]
    assert not result["missing_patients"]
    assert not result["unexpected_scans"]
    assert result["complete"]


def test_manifest_covers_the_whole_eligible_cohort(man):
    if not CENSUS.exists():
        pytest.skip("no census; run src/census.py first")
    with open(CENSUS) as fh:
        census = json.load(fh)["scans"]
    eligible = {s["patient_id"] for s in census
                if any(c["n_ann"] >= MIN_ANNOTATIONS for c in s["clusters"])}
    present = set(man.patient_id.unique())
    assert present == eligible


def test_every_condition_reaches_the_network_in_0_255():
    tiles, meta = first_scan_with_both_classes()
    for arm in ("G0", "G1", "G2", "G3", "G4"):
        for i in (meta.index[meta.label == 1][0], meta.index[meta.label == 0][0]):
            img = A.build_patch(tiles[i], meta.iloc[i], arm, out_px=128)
            assert img.dtype == np.float32
            assert img.min() >= -1e-3 and img.max() <= 255.0 + 1e-3
            assert img.max() - img.min() > 50.0


def test_each_backbone_gets_its_own_imagenet_preprocessing():
    from tensorflow import keras

    import train
    x = np.full((1, 4, 4, 3), 255.0, dtype=np.float32)
    assert np.allclose(keras.applications.efficientnet.preprocess_input(x.copy()), x)
    for name, (app, prep) in train.BACKBONES.items():
        assert hasattr(keras.applications, app)
        if app.startswith("EfficientNet"):
            assert prep is None
        else:
            out = getattr(keras.applications, prep).preprocess_input(x.copy())
            assert not np.allclose(out, x)


def test_dose_conditions_change_positives_only():
    tiles, meta = first_scan_with_both_classes()
    i = meta.index[meta.label == 1][0]
    pos = meta.iloc[i].copy()
    neg = pos.copy()
    neg["label"] = 0
    same_pos = A.build_patch(tiles[i], pos, "D100", out_px=128)
    same_neg = A.build_patch(tiles[i], neg, "D100", out_px=128)
    assert np.array_equal(same_pos, same_neg)
    dosed_pos = A.build_patch(tiles[i], pos, "D366", out_px=128)
    dosed_neg = A.build_patch(tiles[i], neg, "D366", out_px=128)
    assert np.array_equal(dosed_neg, same_neg)
    assert not np.allclose(dosed_pos, same_pos)
    assert dosed_pos.min() >= -1e-3 and dosed_pos.max() <= 255.0 + 1e-3
