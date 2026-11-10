import matplotlib.pyplot as plt

from arms import HU_WINDOW, MATCHED_FOV_MM
from plotting import illustration_row, illustration_tile, matched_stages, save

OUT_PX = 224


def main():
    row = illustration_row()
    tile = illustration_tile(row)
    spacing = float(row.pixel_spacing)
    s = matched_stages(tile, spacing, OUT_PX)
    px = s["window_px"]
    fov = int(MATCHED_FOV_MM)
    lo, hi = (int(v) for v in HU_WINDOW)

    panels = [
        (tile, f"Cached tile, raw HU\n{tile.shape[0]} px at {spacing:.3f} mm"),
        (s["window"], f"{fov} mm window, native grid\n{px} x {px} px"),
        (s["grid"], f"Resampled to a 1 mm grid\n{fov} x {fov} samples"),
        (s["mapped"], f"HU clipped to [{lo}, {hi}]\nfixed map to [0, 255]"),
        (s["network"], f"Network input\n{OUT_PX} x {OUT_PX} by interpolation"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.set_title(title, fontsize=9.5, color="#1a202c", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#cbd5e0")
    fig.suptitle(f"Patch construction, identical for both classes  ({row.patient_id}, "
                 f"nodule diameter {row.diameter_mm:.1f} mm)", fontsize=10.5,
                 color="#4a5568", y=1.03)
    fig.tight_layout()
    save(fig, "fig05_construction_chain.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
