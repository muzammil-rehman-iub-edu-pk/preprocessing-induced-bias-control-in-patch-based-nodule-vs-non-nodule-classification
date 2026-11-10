import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate, shift

import train
from arms import HU_WINDOW
from paths import scan_file
from plotting import matched_stages, save

FILL = "reflect"


def load_positive(patient: str, scan: int, z: int, y: int, x: int):
    d = np.load(scan_file(patient, scan))
    sel = np.where((d["meta_z"] == z) & (d["meta_y"] == y) & (d["meta_x"] == x)
                   & (d["meta_label"] == 1))[0]
    if sel.size == 0:
        raise SystemExit(f"no positive patch at {patient} scan {scan} z{z} y{y} x{x}")
    i = sel[0]
    return d["tiles"][i].astype(np.float32), float(d["meta_pixel_spacing"][i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="LIDC-IDRI-0078")
    ap.add_argument("--scan", type=int, default=1)
    ap.add_argument("--z", type=int, default=67)
    ap.add_argument("--y", type=int, default=245)
    ap.add_argument("--x", type=int, default=343)
    a = ap.parse_args()

    tile, spacing = load_positive(a.patient, a.scan, a.z, a.y, a.x)
    img = matched_stages(tile, spacing)["network"]
    degrees = train.ROTATION * 360.0
    dy = dx = train.TRANSLATION * img.shape[0]

    def rot(v):
        return rotate(v, degrees, reshape=False, mode=FILL, order=1)

    def sft(v):
        return shift(v, (dy, dx), mode=FILL, order=1)

    panels = [
        (img, "Original"),
        (img[:, ::-1], "Horizontal flip"),
        (img[::-1, :], "Vertical flip"),
        (rot(img), f"Rotation, {degrees:+.0f} deg"),
        (sft(img), f"Translation, {train.TRANSLATION * 100:.0f} percent"),
        (sft(rot(img)[:, ::-1]), "Flip, rotate and shift"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(15, 3.1))
    for ax, (v, title) in zip(axes, panels):
        ax.imshow(v, cmap="gray", interpolation="nearest")
        ax.set_title(title, fontsize=10, pad=7)
        ax.set_xticks([])
        ax.set_yticks([])
    lo, hi = (int(v) for v in HU_WINDOW)
    fig.suptitle("Training augmentation operations, each shown at the configured maximum. "
                 f"Matched-construction patch, Hounsfield window {lo} to {hi}.",
                 fontsize=10.5, y=1.04)
    fig.tight_layout()
    save(fig, "fig06_augmentation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
