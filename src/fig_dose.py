import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import RESULTS
from plotting import BLUE, GRID, INK, MUTED, RED, save, style_axes

LEGACY_FACTOR = 3.66
MODELS = (("statistics-only baseline", "Statistics-only baseline", BLUE, "o", True),
          ("EfficientNetB0 128", "EfficientNetB0", RED, "s", False))


def legacy_network_accuracy():
    path = RESULTS / "arm_metrics.csv"
    if not path.exists():
        return None
    m = pd.read_csv(path).set_index("tag")
    tag = "G0_B0_128_single"
    return float(m.loc[tag, "adaptive_midpoint__accuracy"]) if tag in m.index else None


def main():
    d = pd.read_csv(RESULTS / "dose_response.csv")
    with open(RESULTS / "dose_fit.json") as fh:
        fits = json.load(fh)
    d["error_pct"] = 100 * d.errors / d.n_patches

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))

    ax = axes[0]
    for key, label, colour, marker, _ in MODELS:
        s = d[d.model == key].sort_values("factor")
        if len(s):
            ax.plot(s.factor, s.accuracy, "-" + marker, color=colour, lw=2, ms=7, label=label, zorder=3)
    legacy = legacy_network_accuracy()
    if legacy is not None:
        ax.plot([LEGACY_FACTOR], [legacy], "D", mfc="white", mec=RED, mew=1.6, ms=13, zorder=2,
                label="EfficientNetB0, full legacy recipe")
    for x, txt in ((1.0, "strict: both\nclasses alike"), (LEGACY_FACTOR, "legacy recipe,\nmedian factor")):
        ax.axvline(x, color=MUTED, lw=1, ls=":", zorder=1)
        ax.text(x + 0.08, 0.703, txt, fontsize=8.5, color=MUTED, va="bottom")
    ax.set_ylim(0.70, 1.01)
    ax.set_xlabel("Resampling factor applied to positive patches only", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title("a  Accuracy rises with the artifact", loc="left", fontsize=11, color=INK)
    ax.legend(fontsize=8.5, frameon=False, loc="center right")
    style_axes(ax)

    ax = axes[1]
    xs = np.linspace(1, 5, 100)
    for key, label, colour, marker, draw_fit in MODELS:
        s = d[d.model == key].sort_values("factor")
        if not len(s):
            continue
        fit = fits.get(key)
        if draw_fit and fit:
            n = int(s.n_patches.iloc[0])
            p = fit["power_law"]
            ax.plot(xs, 100 * p["scale"] * xs ** p["exponent"] / n, "-", color=colour, lw=1.2, alpha=.6,
                    zorder=2)
            label = rf"{label}, errors $\propto f^{{{p['exponent']:.2f}}}$"
        if not draw_fit:
            ax.plot(s.factor, s.error_pct, "-", color=colour, lw=1.2, alpha=.6, zorder=2)
            label = f"{label}, measured"
        ax.plot(s.factor, s.error_pct, marker, color=colour, ms=7, zorder=3, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 1.5, 2, 3, 4, 5])
    ax.set_xticklabels(["1", "1.5", "2", "3", "4", "5"])
    ax.set_yticks([0.3, 1, 3, 10, 30])
    ax.set_yticklabels(["0.3", "1", "3", "10", "30"])
    ax.minorticks_off()
    ax.set_xlabel("Resampling factor (log scale)", fontsize=10)
    ax.set_ylabel("Misclassified patches, percent (log scale)", fontsize=10)
    ax.set_title("b  Errors fall steeply as the factor grows", loc="left", fontsize=11, color=INK)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    style_axes(ax)
    ax.grid(axis="x", lw=.5, color=GRID)

    fig.tight_layout(w_pad=2.5)
    save(fig, "fig12_dose.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
