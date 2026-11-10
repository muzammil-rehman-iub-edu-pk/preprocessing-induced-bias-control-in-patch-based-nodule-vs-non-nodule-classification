import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from paths import RESULTS
from plotting import GREY, save, style_axes

ACC_COLOUR = "#2e75b6"
AUC_COLOUR = "#2e8b57"
ROC_COLOUR = "#2471a3"


def load():
    with open(RESULTS / "headline.json") as fh:
        head = json.load(fh)
    per_fold = pd.read_csv(RESULTS / "headline_per_fold.csv")
    pooled = pd.read_csv(RESULTS / "headline_pooled_predictions.csv")
    if len(pooled) != head["n_patches"] or pooled.patient_id.nunique() != head["n_patients"]:
        raise ValueError("pooled predictions disagree with headline.json")
    return head, per_fold, pooled


def fig_folds(head, per_fold):
    fig, axes = plt.subplots(1, 2, figsize=(9.9, 3.7))
    panels = [("accuracy", "Test accuracy", "Accuracy", ACC_COLOUR),
              ("auc", "Test AUC", "AUC", AUC_COLOUR)]
    x = per_fold.fold.to_numpy()
    for ax, (col, ylabel, title, colour) in zip(axes, panels):
        v = per_fold[col].to_numpy()
        mean, sd = v.mean(), v.std(ddof=1)
        ax.axhspan(mean - sd, mean + sd, color=colour, alpha=.15, zorder=0)
        ax.bar(x, v, width=0.62, color=colour, alpha=.9, zorder=2)
        ax.axhline(mean, color=colour, lw=2.2, zorder=3)
        for xi, vi in zip(x, v):
            ax.annotate(f"{vi:.3f}", (xi, vi), textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=8.5)
        ax.set_ylim(v.min() - 0.0117, v.max() + 0.006)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {i}" for i in x], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.set_title(f"{title}  (mean = {mean:.3f}, SD = {sd:.3f})", fontsize=10.5)
        style_axes(ax)
    fig.suptitle(f"Five-fold patient-level cross-validation, {head['n_patients']} patients",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "fig09_folds.png", dpi=300)
    plt.close(fig)


def fig_roc(head, pooled):
    fig, ax = plt.subplots(figsize=(5.1, 4.5))
    for _, g in pooled.groupby("fold"):
        fpr, tpr, _ = roc_curve(g.label, g.score)
        ax.plot(fpr, tpr, lw=1.0, color=ROC_COLOUR, alpha=.45, zorder=1)
    fpr, tpr, _ = roc_curve(pooled.label, pooled.score)
    lo, hi = head["pooled_auc_ci"]
    ax.plot(fpr, tpr, lw=2.6, color=ROC_COLOUR, zorder=2,
            label=f"Pooled out-of-fold\nAUC = {head['pooled_auc']:.3f} ({lo:.3f} to {hi:.3f})")
    ax.plot([0, 1], [0, 1], ":", lw=1.0, color=GREY, zorder=0)
    ax.set_xlim(-.01, 1.0)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("1 - specificity", fontsize=10)
    ax.set_ylabel("Sensitivity", fontsize=10)
    ax.set_title("Receiver operating characteristic\nthin lines are individual folds",
                 fontsize=10.5, loc="left")
    style_axes(ax)
    ax.grid(axis="x", lw=.5, color="#e2e8f0")
    ax.legend(fontsize=9, frameon=True, loc="lower right")
    fig.tight_layout()
    save(fig, "fig10_roc.png", dpi=300)
    plt.close(fig)


def fig_confusion(head):
    m = head["pooled_metrics"]
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    if cm.sum() != head["n_patches"]:
        raise ValueError("confusion counts do not sum to the number of patches")
    fig, ax = plt.subplots(figsize=(4.53, 3.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=15,
                    fontweight="bold", color="white" if cm[i, j] > cm.max() / 2 else "#1a202c")
    ax.set_xticks([0, 1], ["Predicted\nnon-nodule", "Predicted\nnodule"], fontsize=8.5)
    ax.set_yticks([0, 1], ["True\nnon-nodule", "True\nnodule"], fontsize=8.5)
    ax.set_title(f"Pooled out-of-fold, {head['n_patches']:,} patches from "
                 f"{head['n_patients']} patients", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, "fig11_confusion.png", dpi=300)
    plt.close(fig)


def main():
    head, per_fold, pooled = load()
    fig_folds(head, per_fold)
    fig_roc(head, pooled)
    fig_confusion(head)


if __name__ == "__main__":
    main()
