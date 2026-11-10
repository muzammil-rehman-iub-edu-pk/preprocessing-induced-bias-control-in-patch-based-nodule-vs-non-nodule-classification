import configparser
import importlib.metadata as md
import platform
import sys
from pathlib import Path

from paths import CACHE, CENSUS, MANIFEST, RESULTS

EXPECTED = {
    "numpy": "1.23.5",
    "scipy": "1.15.3",
    "pandas": "2.1.4",
    "scikit-learn": "1.3.2",
    "scikit-image": "0.21.0",
    "matplotlib": "3.8.2",
    "pydicom": "2.4.3",
    "pylidc": "0.2.3",
    "SQLAlchemy": "2.0.46",
    "keras": "2.14.0",
}
TENSORFLOW = ("tensorflow-macos", "tensorflow")
WEIGHTS = ("efficientnetb0_notop.h5", "efficientnetb3_notop.h5")


def version(dist: str):
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def report(label: str, ok: bool, detail: str):
    print(f"  [{'ok' if ok else 'check'}] {label:28s} {detail}")
    return ok


def main() -> int:
    problems = 0
    print("Python and platform")
    problems += not report("python", sys.version_info[:2] == (3, 11), platform.python_version())
    report("platform", True, platform.platform())

    print("\nPackages")
    for dist, want in EXPECTED.items():
        have = version(dist)
        problems += not report(dist, have == want, f"installed {have}, expected {want}")
    tf_dist = next((d for d in TENSORFLOW if version(d)), None)
    problems += not report("tensorflow", tf_dist is not None and version(tf_dist) == "2.14.0",
                           f"{tf_dist} {version(tf_dist) if tf_dist else None}")
    if platform.system() == "Darwin":
        report("tensorflow-metal", version("tensorflow-metal") is not None,
               f"installed {version('tensorflow-metal')}")
    st = version("setuptools")
    problems += not report("setuptools", st is not None and int(st.split(".")[0]) < 81,
                           f"installed {st}, needs < 81")

    print("\nImports")
    for module in ("scipy.ndimage", "skimage.transform", "sklearn.metrics", "pylidc"):
        try:
            __import__(module)
            report(module, True, "imports")
        except Exception as exc:
            problems += not report(module, False, str(exc)[:70])

    print("\npylidc configuration")
    rc = Path.home() / (".pylidcrc" if not sys.platform.startswith("win") else "pylidc.conf")
    dicom = None
    if rc.exists():
        parser = configparser.ConfigParser()
        parser.read(rc)
        dicom = parser.get("dicom", "path", fallback=None)
    report("config file", rc.exists(), str(rc))
    has_dicom = bool(dicom) and Path(dicom).is_dir()
    n_patients = len(list(Path(dicom).glob("LIDC-IDRI-*"))) if has_dicom else 0
    report("DICOM root", has_dicom, f"{dicom} ({n_patients} patient folders)")

    print("\nTensorFlow devices and weights")
    try:
        import tensorflow as tf
        devices = [d.device_type for d in tf.config.list_physical_devices()]
        report("devices", True, ", ".join(devices))
    except Exception as exc:
        problems += not report("tensorflow import", False, str(exc)[:70])
    cached = Path.home() / ".keras" / "models"
    for w in WEIGHTS:
        report(w, (cached / w).exists(), "cached" if (cached / w).exists()
               else "downloaded on first training run")

    print("\nRepository data")
    report("results directory", RESULTS.exists(), str(RESULTS))
    report("census", CENSUS.exists(), str(CENSUS))
    report("extraction cache", MANIFEST.exists(),
           f"{CACHE} ({len(list(CACHE.glob('*.npz')))} scan files)")

    print(f"\n{problems} blocking problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
