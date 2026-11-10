import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("NODULE_BIAS_CACHE", ROOT / "cache"))
RESULTS = Path(os.environ.get("NODULE_BIAS_RESULTS", ROOT / "results"))
FIGURES = Path(os.environ.get("NODULE_BIAS_FIGURES", ROOT / "figures"))

RUNS = RESULTS / "runs"
BACKBONE_RESULTS = RESULTS / "backbones"
CENSUS = RESULTS / "census_scans.json"
MANIFEST = CACHE / "manifest.csv"
SCAN_LOG = CACHE / "scan_log.csv"


def scan_file(patient_id, scan_id) -> Path:
    return CACHE / f"{patient_id}__scan_{int(scan_id)}.npz"
