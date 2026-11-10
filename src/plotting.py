import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from skimage.transform import resize

from arms import HU_WINDOW, MATCHED_FOV_MM
from paths import FIGURES, MANIFEST, scan_file

INK = "#1a202c"
MUTED = "#718096"
RED = "#c0392b"
BLUE = "#2471a3"
GREEN = "#2e8b57"
GREY = "#7f8c8d"
GRID = "#e2e8f0"


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", lw=.5, color=GRID)
    ax.set_axisbelow(True)


def save(fig, name: str, dpi: int = 300, **kwargs):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=dpi, facecolor="white", **kwargs)
    print(f"wrote {path}")
    return path


def illustration_row() -> pd.Series:
    m = pd.read_csv(MANIFEST)
    candidates = m[(m.label == 1)
                   & (m.diameter_mm.between(12, 22))
                   & (m.subtlety_median >= 4.5)
                   & (m.tile_pad_fraction == 0)
                   & (m.lung_fraction_window > 0.85)].sort_values(["diameter_mm", "patient_id"])
    if candidates.empty:
        raise SystemExit("no positive patch matched the illustration rule")
    return candidates.iloc[len(candidates) // 2]


def illustration_tile(row: pd.Series) -> np.ndarray:
    z = np.load(scan_file(row.patient_id, row.scan_id))
    sel = np.where((z["meta_z"] == int(row.z)) & (z["meta_y"] == int(row.y))
                   & (z["meta_x"] == int(row.x)) & (z["meta_label"] == 1))[0]
    if sel.size == 0:
        raise SystemExit("illustration tile not found in the cache")
    return z["tiles"][sel[0]].astype(np.float32)


def matched_stages(tile: np.ndarray, pixel_spacing: float, out_px: int = 224) -> dict:
    px = int(round(MATCHED_FOV_MM / pixel_spacing))
    c, h = tile.shape[0] // 2, px // 2
    window = tile[c - h:c - h + px, c - h:c - h + px]
    grid = resize(window, (int(MATCHED_FOV_MM), int(MATCHED_FOV_MM)),
                  anti_aliasing=True, preserve_range=True)
    lo, hi = HU_WINDOW
    mapped = (np.clip(grid, lo, hi) - lo) / (hi - lo) * 255.0
    network = resize(mapped, (out_px, out_px), anti_aliasing=True, preserve_range=True)
    return {"window_px": px, "window": window, "grid": grid, "mapped": mapped,
            "network": network}
