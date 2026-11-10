import argparse
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pylidc as pl
from matplotlib.patches import Rectangle

from extract import MIN_ANNOTATIONS
from plotting import RED, save

MARGIN_PX = 6


def scale_for_display(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    x = x.astype("float32")
    lo = float(np.percentile(x, p_lo))
    hi = float(np.percentile(x, p_hi))
    return np.clip((x - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def largest_nodule(scan):
    clusters = [c for c in scan.cluster_annotations() if len(c) >= MIN_ANNOTATIONS]
    if not clusters:
        return None
    best = max(clusters, key=lambda c: np.median([a.diameter for a in c]))
    sl = best[0].bbox()
    return (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop, sl[2].start, sl[2].stop,
            float(np.median([a.diameter for a in best])))


def slice_indices(n_slices: int, n_tiles: int, box):
    if box is None:
        return np.linspace(0, n_slices - 1, n_tiles).astype(int)
    z0, z1 = box[4], box[5] - 1
    pad = max(2, (z1 - z0 + 1) // 2)
    lo, hi = max(0, z0 - pad), min(n_slices - 1, z1 + pad)
    return np.unique(np.linspace(lo, hi, n_tiles).astype(int))


def overview(vol: np.ndarray, idxs, box=None, cols: int = 6):
    rows = int(np.ceil(len(idxs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2 * rows), dpi=200)
    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows * cols):
        ax = axes.flat[i]
        ax.axis("off")
        if i >= len(idxs):
            continue
        zi = int(idxs[i])
        ax.imshow(scale_for_display(vol[:, :, zi]), cmap="gray", vmin=0, vmax=1)
        if box is not None and box[4] <= zi < box[5]:
            y0, y1, x0, x1 = box[0] - MARGIN_PX, box[1] + MARGIN_PX, box[2] - MARGIN_PX, box[3] + MARGIN_PX
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=RED, lw=1.0))
        ax.set_title(f"z={zi}", fontsize=8, pad=6)
    plt.tight_layout(pad=0.6, h_pad=1.0, w_pad=0.4)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="LIDC-IDRI-0050")
    ap.add_argument("--tiles", type=int, default=12)
    ap.add_argument("--whole-scan", action="store_true")
    ap.add_argument("--out", default="fig01_overview.png")
    a = ap.parse_args()

    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == a.patient).first()
    if scan is None:
        raise SystemExit(f"{a.patient} is not in the pylidc database")
    vol = scan.to_volume(verbose=False)
    box = None if a.whole_scan else largest_nodule(scan)
    idxs = slice_indices(vol.shape[2], a.tiles, box)
    fig = overview(vol, idxs, box)
    save(fig, a.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    if box is not None:
        print(f"{a.patient}: nodule median diameter {box[6]:.1f} mm, slices {box[4]} to {box[5] - 1}")


if __name__ == "__main__":
    main()
