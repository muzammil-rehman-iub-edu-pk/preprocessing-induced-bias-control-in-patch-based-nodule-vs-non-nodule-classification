import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import RESULTS
from plotting import BLUE, GREY, RED, save, style_axes

ARMS = ["G0", "G1", "G2", "G3", "G4"]
LABEL = {"G0": "Legacy", "G1": "Geometry\nmatched", "G2": "Fixed\nintensity",
         "G3": "Parenchyma\nnegatives", "G4": "Hard\nnegatives"}
CNN_TAG = "{arm}_B0_128_single"


def load():
    probe = pd.read_csv(RESULTS / "probe_gap.csv").set_index("arm")
    metrics = pd.read_csv(RESULTS / "arm_metrics.csv").set_index("tag")
    rows = []
    for arm in ARMS:
        tag = CNN_TAG.format(arm=arm)
        if tag not in metrics.index:
            raise KeyError(f"{tag} missing from arm_metrics.csv")
        rows.append({"arm": arm,
                     "probe_extracted": probe.loc[arm, "probe_extracted"],
                     "probe_source": probe.loc[arm, "probe_source"],
                     "gap": probe.loc[arm, "fabrication_gap"],
                     "cnn": metrics.loc[tag, "auc"]})
    d = pd.DataFrame(rows)
    if not np.allclose(d.gap, d.probe_extracted - d.probe_source, atol=1e-9):
        raise ValueError("fabrication gap does not equal extracted minus source")
    return d


def main():
    d = load()
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))

    g0 = d.iloc[0]
    ax.vlines(0, g0.probe_source, g0.probe_extracted, color=RED, lw=13, alpha=.18, zorder=0)
    ax.plot(x, d.probe_extracted, "s-", color=RED, lw=2.2, ms=9, zorder=3,
            label="Statistics only baseline, extracted patches")
    ax.plot(x, d.probe_source, "^--", color=GREY, lw=2.0, ms=8, zorder=2,
            label="Statistics only baseline, source grid")
    ax.plot(x, d.cnn, "o-", color=BLUE, lw=2.2, ms=9, zorder=3, label="EfficientNetB0")

    ax.annotate(f"{g0.gap:.3f} introduced\nby extraction",
                xy=(0.07, (g0.probe_source + g0.probe_extracted) / 2 + 0.002),
                xytext=(0.75, 0.945), fontsize=10, color=RED, va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2,
                                connectionstyle="arc3,rad=-0.2"))

    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[a] for a in d.arm], fontsize=10)
    ax.set_xlim(-.55, len(d) - .65)
    lowest = float(min(d.probe_extracted.min(), d.probe_source.min(), d.cnn.min()))
    floor = 0.78 if lowest - 0.02 >= 0.78 else np.floor((lowest - 0.02) / 0.05) * 0.05
    ax.set_ylim(floor, 1.02)
    ax.set_yticks(np.round(np.arange(np.ceil(floor / 0.05) * 0.05, 1.0001, 0.05), 2))
    ax.set_xlabel("Patch construction", fontsize=11)
    ax.set_ylabel("Area under the curve", fontsize=11)
    ax.tick_params(labelsize=10)
    style_axes(ax)
    ax.legend(fontsize=9.5, frameon=True, loc="lower right")
    fig.tight_layout()
    save(fig, "fig07_construction.png", dpi=300)
    plt.close(fig)
    print(d.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
