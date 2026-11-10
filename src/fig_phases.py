import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import train
from plotting import illustration_row, illustration_tile, matched_stages, save

W, H = 13.0, 4.35
MID = 2.32

INK = "#12263a"
MUTED = "#5c6f80"
FAINT = "#8a9bab"
RULE = "#d6dee5"
PAPER = "#fbfcfd"
FROZ_FC, FROZ_EC = "#e4ebf1", "#9fb0bf"
LIVE_FC, LIVE_EC = "#bcd6ea", "#2f6390"
HEAD_FC, HEAD_EC = "#e9f1ee", "#43776c"
AUG_FC, AUG_EC = "#f6f0e6", "#9c7b44"

SERIF = {"family": "DejaVu Serif"}
SANS = {"family": "DejaVu Sans"}


def rbox(ax, x, y, w, h, fc, ec, lw=1.1, r=0.05, z=2, shadow=True):
    if shadow:
        ax.add_patch(FancyBboxPatch((x + .035, y - .035), w, h,
                                    boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc="#000000", ec="none", alpha=.055, zorder=z - 1))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def flow(ax, x0, x1, y=MID):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=11, color=FAINT, lw=1.15,
                                 shrinkA=0, shrinkB=0, zorder=4))


def footer_items(phase: int, clipnorm: float):
    if phase == 1:
        items = [("epochs", f"{train.PHASE1_EPOCHS}"),
                 ("optimizer", f"Adam, {train.PHASE1_LR:.0e}".replace("e-0", "e-")),
                 ("batch", f"{train.BATCH}"),
                 ("loss", f"focal BCE, gamma {train.FOCAL_GAMMA}")]
        last = ("stopping", f"val AUC, patience {train.PATIENCE}")
    else:
        items = [("epochs", f"up to {train.PHASE2_EPOCHS}"),
                 ("optimizer", f"Adam, {train.PHASE2_LR:.0e}".replace("e-0", "e-")),
                 ("schedule", f"cosine decay to {train.COSINE_ALPHA:.0%}"),
                 ("stopping", f"val AUC, patience {train.PATIENCE}")]
        last = ("batch", f"{train.BATCH}")
    items.append(("clip", f"grad norm {clipnorm:g}") if clipnorm else last)
    return items


def draw(phase: int, name: str, img, clipnorm: float):
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, fc=PAPER, ec="none", zorder=0))

    label = "Phase one" if phase == 1 else "Phase two"
    sub = ("classifier head trained against a frozen backbone"
           if phase == 1 else
           "upper backbone unfrozen and fine-tuned jointly with the head")
    ax.plot([0.45, W - 0.45], [3.93, 3.93], color=INK, lw=1.5, zorder=3)
    ax.text(0.45, 4.03, label, fontsize=13, color=INK, weight="bold", va="bottom", **SERIF)
    ax.text(2.30, 4.06, sub, fontsize=9.4, color=MUTED, va="bottom", style="italic", **SERIF)

    lx = W - 3.55
    for k, (fc, ec, txt) in enumerate([(FROZ_FC, FROZ_EC, "frozen"),
                                       (LIVE_FC, LIVE_EC, "trainable")]):
        rbox(ax, lx + k * 1.72, 4.02, .26, .17, fc, ec, lw=.9, r=.03, shadow=False)
        ax.text(lx + .34 + k * 1.72, 4.105, txt, fontsize=8.6, color=MUTED, va="center", **SANS)

    ix, iw = 0.46, 1.26
    ax.add_patch(Rectangle((ix + .04, MID - iw / 2 - .04), iw, iw,
                           fc="#000", ec="none", alpha=.07, zorder=1))
    ia = fig.add_axes([ix / W, (MID - iw / 2) / H, iw / W, iw / H], zorder=3)
    ia.imshow(img, cmap="gray", interpolation="bilinear")
    ia.set_xticks([])
    ia.set_yticks([])
    for s in ia.spines.values():
        s.set_color(INK)
        s.set_linewidth(1.2)
    ax.text(ix + iw / 2, MID - iw / 2 - .30, "224 x 224 x 3", ha="center",
            fontsize=9.2, color=INK, **SANS)
    ax.text(ix + iw / 2, MID - iw / 2 - .55, "one axial slice, replicated", ha="center",
            fontsize=8.2, color=MUTED, **SANS)
    ax.text(ix + iw / 2, MID + iw / 2 + .18, "INPUT", ha="center", fontsize=8.6,
            color=FAINT, weight="bold", **SANS)

    x = ix + iw + .16
    flow(ax, x, x + .32)
    x += .44

    aw = .74
    rbox(ax, x, MID - 1.02, aw, 2.04, AUG_FC, AUG_EC, lw=1.1, r=.07)
    ax.text(x + aw / 2, MID + .74, "Augment", ha="center", fontsize=9.4,
            color=INK, weight="bold", **SANS)
    for k, s in enumerate(["flip h+v", f"rotate {train.ROTATION}",
                           f"shift {train.TRANSLATION}"]):
        ax.text(x + aw / 2, MID + .30 - k * .38, s, ha="center",
                fontsize=7.9, color=MUTED, **SANS)
    ax.text(x + aw / 2, MID - .92, "train only", ha="center", fontsize=7.4,
            color=AUG_EC, style="italic", **SANS)
    x += aw + .16
    flow(ax, x, x + .32)
    x += .44

    bw = 3.24
    rbox(ax, x, MID - 1.26, bw, 2.52, "#ffffff", RULE, lw=1.0, r=.07)
    ax.text(x + bw / 2, MID + .94, "EfficientNetB0", ha="center", fontsize=11,
            color=INK, weight="bold", **SERIF)
    ax.text(x + bw / 2, MID + .70,
            "ImageNet initialization" if phase == 1 else "phase one weights restored",
            ha="center", fontsize=8.3, color=MUTED, **SANS)

    stages = [(112, .175, 1.34), (56, .225, 1.14), (28, .295, .95),
              (14, .385, .78), (7, .50, .63)]
    gap = .155
    total = sum(s[1] for s in stages) + gap * 4
    bx = x + (bw - total) / 2
    for k, (dim, sw, sh) in enumerate(stages):
        live = phase == 2 and k >= 3
        fc, ec = (LIVE_FC, LIVE_EC) if live else (FROZ_FC, FROZ_EC)
        ax.add_patch(Rectangle((bx + .03, MID - sh / 2 - .03), sw, sh,
                               fc="#000", ec="none", alpha=.05, zorder=2))
        ax.add_patch(Rectangle((bx, MID - sh / 2), sw, sh, fc=fc, ec=ec, lw=1.0, zorder=3))
        ax.text(bx + sw / 2, MID - .86, f"{dim}", ha="center",
                fontsize=7.3, color=FAINT, **SANS)
        bx += sw + gap

    if phase == 1:
        ax.text(x + bw / 2, MID - 1.11, "all layers frozen, no gradient reaches the backbone",
                ha="center", fontsize=8.4, color=MUTED, **SANS)
    else:
        frozen = int(round(train.FROZEN_FRACTION * 100))
        ax.plot([x + .28, x + 1.72], [MID - 1.02] * 2, color=FROZ_EC, lw=1.5)
        ax.plot([x + 1.84, x + bw - .28], [MID - 1.02] * 2, color=LIVE_EC, lw=1.8)
        ax.text(x + 1.00, MID - 1.22, f"lower {frozen}%, frozen", ha="center", fontsize=8.2,
                color=MUTED, **SANS)
        ax.text(x + 2.46, MID - 1.22, f"upper {100 - frozen}%, fine-tuned", ha="center",
                fontsize=8.2, color=LIVE_EC, weight="bold", **SANS)
    x += bw + .16
    flow(ax, x, x + .32)
    x += .44

    gw, gh = .30, 1.46
    ax.add_patch(Rectangle((x + .03, MID - gh / 2 - .03), gw, gh,
                           fc="#000", ec="none", alpha=.05, zorder=2))
    ax.add_patch(Rectangle((x, MID - gh / 2), gw, gh, fc="#f2f6f9", ec=FROZ_EC,
                           lw=1.0, zorder=3))
    for k in range(9):
        ax.plot([x + .05, x + gw - .05], [MID - gh / 2 + .145 * (k + 1)] * 2,
                color=FROZ_EC, lw=.55, zorder=4)
    ax.text(x + gw / 2, MID + gh / 2 + .20, "pool", ha="center", fontsize=8.6,
            color=INK, **SANS)
    ax.text(x + gw / 2, MID - gh / 2 - .26, "1280", ha="center", fontsize=8.2,
            color=MUTED, **SANS)
    x += gw + .16
    flow(ax, x, x + .32)
    x += .44

    hw = 2.86
    rbox(ax, x, MID - 1.26, hw, 2.52, HEAD_FC, HEAD_EC, lw=1.2, r=.07)
    ax.text(x + hw / 2, MID + .94, "Classifier head", ha="center", fontsize=11,
            color=INK, weight="bold", **SERIF)
    ax.text(x + hw / 2, MID + .70, "trainable in both phases", ha="center",
            fontsize=8.3, color=MUTED, **SANS)
    rows = ["batch normalization", f"dropout {train.DROPOUT}",
            f"dense 256, swish, L2 {train.L2:.0e}".replace("e-0", "e-"),
            f"dropout {train.DROPOUT / 2:g}", "dense 1, sigmoid"]
    rh, rg, top = .265, .075, MID + .48
    for k, s in enumerate(rows):
        yy = top - k * (rh + rg)
        strong = k in (2, 4)
        ax.add_patch(FancyBboxPatch((x + .17, yy - rh), hw - .34, rh,
                                    boxstyle="round,pad=0,rounding_size=.035",
                                    fc="white", ec=HEAD_EC if strong else "#b9cdc7",
                                    lw=1.05 if strong else .85, zorder=3))
        ax.text(x + hw / 2, yy - rh / 2, s, ha="center", va="center",
                fontsize=8.5, color=INK if strong else MUTED,
                weight="bold" if strong else "normal", **SANS)
    x += hw + .16
    flow(ax, x, x + .32)
    x += .44

    ax.add_patch(plt.Circle((x + .17, MID), .155, fc="white", ec=INK, lw=1.35, zorder=4))
    ax.add_patch(plt.Circle((x + .17, MID), .062, fc=LIVE_EC, ec="none", zorder=5))
    ax.text(x + .17, MID - .44, "p(nodule)", ha="center", fontsize=9.0, color=INK, **SANS)
    ax.text(x + .17, MID - .68, "0 to 1", ha="center", fontsize=8.2, color=MUTED, **SANS)
    ax.text(x + .17, MID + .40, "OUTPUT", ha="center", fontsize=8.6, color=FAINT,
            weight="bold", **SANS)

    ax.plot([0.45, W - 0.45], [0.60, 0.60], color=RULE, lw=1.0, zorder=3)
    items = footer_items(phase, clipnorm)
    span = (W - 1.5) / len(items)
    for k, (a, b) in enumerate(items):
        cx = 0.75 + span * (k + .5)
        ax.text(cx, 0.36, a, ha="center", fontsize=7.6, color=FAINT, **SANS)
        ax.text(cx, 0.13, b, ha="center", fontsize=8.8, color=INK, **SANS)
        if k:
            ax.plot([0.75 + span * k] * 2, [0.10, 0.46], color=RULE, lw=.9)

    save(fig, name, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clipnorm", type=float, default=0.0)
    a = ap.parse_args()
    row = illustration_row()
    img = matched_stages(illustration_tile(row), float(row.pixel_spacing))["mapped"]
    draw(1, "fig03_phase1.png", img, a.clipnorm)
    draw(2, "fig04_phase2.png", img, a.clipnorm)


if __name__ == "__main__":
    main()
