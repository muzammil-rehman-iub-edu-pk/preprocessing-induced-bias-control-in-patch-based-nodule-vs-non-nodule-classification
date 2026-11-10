import json

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

from paths import RESULTS
from plotting import save

FRAME_FC, FRAME_EC = "#f9dedb", "#c0532f"
STEP_FC, STEP_EC = "#aed6f1", "#5d6d7e"
DATA_FC, DATA_EC = "#f5b7a5", "#5d6d7e"
ARROW = "#5d6d7e"
INK = "#000000"
SUB = "#1c2833"

W = 8.0
MARGIN = 0.25
BOX_X, BOX_W = 0.55, 6.9
BOX_H = 0.66
GAP = 0.24
CYL_H = 0.62


def counts():
    with open(RESULTS / "consort.json") as fh:
        flow = {r["stage"]: r["n"] for r in json.load(fh)["flow"]}
    return flow


def cylinder(ax, cx, top, width, height, label, fontsize=10, sublabel=None):
    ry = 0.11
    body = Rectangle((cx - width / 2, top - height), width, height - ry,
                     fc=DATA_FC, ec="none", zorder=2)
    ax.add_patch(body)
    ax.plot([cx - width / 2] * 2, [top - height, top - ry], color=DATA_EC, lw=1.6, zorder=3)
    ax.plot([cx + width / 2] * 2, [top - height, top - ry], color=DATA_EC, lw=1.6, zorder=3)
    ax.add_patch(Ellipse((cx, top - height), width, 2 * ry, fc=DATA_FC, ec=DATA_EC,
                         lw=1.6, zorder=3))
    ax.add_patch(Ellipse((cx, top - ry), width, 2 * ry, fc=DATA_FC, ec=DATA_EC,
                         lw=1.6, zorder=4))
    middle = top - height / 2 - 0.03
    if sublabel:
        ax.text(cx, middle + 0.09, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=INK, zorder=5)
        ax.text(cx, middle - 0.17, sublabel, ha="center", va="center",
                fontsize=8.6, color=SUB, zorder=5)
    else:
        ax.text(cx, middle, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=INK, zorder=5)


def step(ax, top, title, body):
    ax.add_patch(FancyBboxPatch((BOX_X, top - BOX_H), BOX_W, BOX_H,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                fc=STEP_FC, ec=STEP_EC, lw=1.6, zorder=2))
    ax.text(BOX_X + BOX_W / 2, top - 0.22, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color=INK, zorder=3)
    ax.text(BOX_X + BOX_W / 2, top - 0.48, body, ha="center", va="center",
            fontsize=8.6, color=SUB, zorder=3)
    return top - BOX_H


def arrow(ax, x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=13,
                                 color=ARROW, lw=1.8, shrinkA=0, shrinkB=0, zorder=1))


def main():
    c = counts()
    cx = BOX_X + BOX_W / 2

    before_split = [
        ("Step 1: Cohort selection",
         f"Nodules marked by at least 3 of 4 readers: {c['Clusters with >=3 annotations']} "
         f"nodules, {c['Eligible scans']} scans, {c['Eligible patients']} patients"),
        ("Step 2: Native-grid extraction",
         "One 224 x 224 Hounsfield tile per patch; 3 slices per nodule; "
         "negatives from nodule-free slices"),
        ("Step 3: Patient-level splitting",
         "Folds formed on patient identity; patients with two scans kept whole"),
    ]
    after_split = [
        ("Step 4: Matched patch construction",
         "64 mm field of view resampled to a 1 mm in-plane grid, for both classes"),
        ("Step 5: Fixed intensity mapping",
         "Clip to [-1000, 400] HU; fixed affine to [0, 255], not per crop"),
        ("Step 6: Duplicate audit",
         "Exact and perceptual hashes across splits; reported, nothing removed"),
        ("Step 7: Pipeline initialization",
         "Build input pipelines; initialize EfficientNetB0 with ImageNet weights"),
        ("Step 8: Phase 1 training",
         "Frozen backbone for 12 epochs; checkpoint on validation AUC"),
        ("Step 9: Phase 2 fine-tuning",
         "Upper 30% of layers unfrozen; cosine decay; early stopping on validation AUC"),
        ("Step 10: Threshold and inference",
         "Threshold selected on validation; evaluated once on the held-out test set"),
        ("Step 11: Evaluation",
         "AUC, confusion counts, calibration, prevalence-adjusted PPV"),
    ]

    n_steps = len(before_split) + len(after_split)
    H = (2 * MARGIN + 0.25 + 0.8 + GAP + n_steps * (BOX_H + GAP)
         + 0.25 + CYL_H + 0.45)
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.add_patch(Rectangle((MARGIN, MARGIN), W - 2 * MARGIN, H - 2 * MARGIN,
                           fc=FRAME_FC, ec=FRAME_EC, lw=3.0, zorder=0))

    y = H - MARGIN - 0.25
    cylinder(ax, cx, y, 4.0, 0.8, "LIDC-IDRI CT Scans Dataset", fontsize=11,
             sublabel=f"{c['TCIA collection: scans']} scans, "
                      f"{c['TCIA collection: patients']} patients")
    y -= 0.8 + 0.11
    arrow(ax, cx, y, y - GAP)
    y -= GAP

    for i, (title, body) in enumerate(before_split):
        y = step(ax, y, title, body)
        if i < len(before_split) - 1:
            arrow(ax, cx, y, y - GAP)
            y -= GAP

    xs = [BOX_X + BOX_W * f for f in (1 / 6, 1 / 2, 5 / 6)]
    branch = y - 0.16
    ax.plot([cx, cx], [y, branch], color=ARROW, lw=1.8, zorder=1)
    ax.plot([xs[0], xs[-1]], [branch, branch], color=ARROW, lw=1.8, zorder=1)
    top = branch - 0.20
    for x, label in zip(xs, ("Held-out test", "Validation", "Training")):
        arrow(ax, x, branch, top + 0.01)
        cylinder(ax, x, top, 2.0, CYL_H, label, fontsize=9.5)
    merge = top - CYL_H - 0.20
    for x in xs:
        ax.plot([x, x], [top - CYL_H - 0.11, merge], color=ARROW, lw=1.8, zorder=1)
    ax.plot([xs[0], xs[-1]], [merge, merge], color=ARROW, lw=1.8, zorder=1)
    arrow(ax, cx, merge, merge - 0.25)
    y = merge - 0.25

    for i, (title, body) in enumerate(after_split):
        y = step(ax, y, title, body)
        if i < len(after_split) - 1:
            arrow(ax, cx, y, y - GAP)
            y -= GAP

    save(fig, "fig02_pipeline.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
