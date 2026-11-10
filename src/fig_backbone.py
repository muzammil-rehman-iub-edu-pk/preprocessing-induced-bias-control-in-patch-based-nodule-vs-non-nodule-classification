import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from paths import RUNS
from plotting import BLUE, RED, save, style_axes

GRID = {"B0": (96, 128, 224), "B3": (128, 224, 300)}
NAMES = {"B0": "EfficientNetB0", "B3": "EfficientNetB3"}
COLOURS = {"B0": BLUE, "B3": RED}
MARKERS = {"B0": "o", "B3": "s"}
EXTRA_SEEDS = (1234, 2026)


def test_auc(tag: str):
    path = RUNS / f"{tag}_run.json"
    if not path.exists():
        return None
    with open(path) as fh:
        return float(json.load(fh)["test_auc"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", action="store_true")
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for bb, resolutions in GRID.items():
        points = [(r, test_auc(f"G2_{bb}_{r}_single")) for r in resolutions]
        points = [(r, v) for r, v in points if v is not None]
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, "-", marker=MARKERS[bb], color=COLOURS[bb], lw=2.4, ms=10,
                label=NAMES[bb], zorder=3)
        for r, v in points:
            ax.annotate(f"{v:.4f}", (r, v), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, color=COLOURS[bb])
            if a.seeds:
                extra = [test_auc(f"G2_{bb}_{r}_single_seed{s}") for s in EXTRA_SEEDS]
                extra = [v for v in extra if v is not None]
                if extra:
                    vals = extra + [v]
                    ax.vlines(r, min(vals), max(vals), color=COLOURS[bb], lw=6, alpha=.25)
                    ax.plot([r] * len(extra), extra, MARKERS[bb], color=COLOURS[bb],
                            mfc="white", ms=5, zorder=4)

    ax.set_xticks([96, 128, 224, 300])
    ax.set_xlim(80, 316)
    ax.set_ylim(0.965, 0.999)
    ax.set_xlabel("Input resolution (pixels)", fontsize=11)
    ax.set_ylabel("Area under the curve", fontsize=11)
    ax.tick_params(labelsize=10)
    style_axes(ax)
    ax.legend(fontsize=10, frameon=True, loc="lower right")
    fig.tight_layout()
    save(fig, "fig08_backbone.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
