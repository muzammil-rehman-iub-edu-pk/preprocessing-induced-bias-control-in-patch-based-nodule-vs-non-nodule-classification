import numpy as np
from skimage.transform import resize

HU_WINDOW = (-1000.0, 400.0)
MATCHED_FOV_MM = 64.0
LEGACY_NEG_PX = 128
TILE_PX = 224
INPUT_MAX = 255.0

ARMS = {
    "G0": "legacy recipe: annotation box resized against a fixed native window; per-crop min-max to [0, 255]",
    "G1": "geometry matched: 64 mm field of view on a 1 mm grid for both classes; per-crop min-max to [0, 255]",
    "G2": "G1 with a fixed Hounsfield window and a fixed affine map",
    "G3": "G2 with negatives restricted to lung parenchyma",
    "G4": "G2 with negatives restricted to structure-centred windows",
}

ARM_NEG_RULE = {"G0": "any", "G1": "any", "G2": "any", "G3": "lung", "G4": "hard"}

DOSE_WINDOW_PX = 128
DOSE_FACTORS = {"D100": 1.0, "D125": 1.25, "D150": 1.5, "D200": 2.0, "D250": 2.5,
                "D300": 3.0, "D366": 3.66, "D400": 4.0, "D500": 5.0}
ARMS.update({arm: "dose: 128 px native window and fixed Hounsfield map for both classes; "
                   f"positives alone resampled by a factor of {f:g}"
              for arm, f in DOSE_FACTORS.items()})
ARM_NEG_RULE.update({arm: "any" for arm in DOSE_FACTORS})


def _centre_crop(tile: np.ndarray, px: int) -> np.ndarray:
    px = max(1, min(int(px), tile.shape[0]))
    o = (tile.shape[0] - px) // 2
    return tile[o:o + px, o:o + px]


def _minmax(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo + 1e-8)


def _hu_affine(a: np.ndarray) -> np.ndarray:
    lo, hi = HU_WINDOW
    return (np.clip(a.astype(np.float32), lo, hi) - lo) * (INPUT_MAX / (hi - lo))


def reference_patch(tile: np.ndarray, row) -> np.ndarray:
    px = int(round(MATCHED_FOV_MM / float(row["pixel_spacing"])))
    return _centre_crop(tile, px).astype(np.float32)


def resample_by(img: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return img
    n = max(4, int(round(img.shape[0] / factor)))
    small = resize(img, (n, n), anti_aliasing=True, preserve_range=True)
    return resize(small, img.shape, anti_aliasing=False, preserve_range=True)


def build_patch(tile: np.ndarray, row, arm: str, out_px: int = 224) -> np.ndarray:
    label = int(row["label"])
    spacing = float(row["pixel_spacing"])

    if arm in DOSE_FACTORS:
        img = _hu_affine(_centre_crop(tile, DOSE_WINDOW_PX))
        if label == 1:
            img = resample_by(img, DOSE_FACTORS[arm])
        if img.shape != (out_px, out_px):
            img = resize(img, (out_px, out_px), anti_aliasing=True, preserve_range=True)
        return img.astype(np.float32)

    if arm == "G0":
        if label == 1:
            y0, y1 = int(row["bbox_y0"]), int(row["bbox_y1"])
            x0, x1 = int(row["bbox_x0"]), int(row["bbox_x1"])
            y0, x0 = max(0, y0), max(0, x0)
            y1, x1 = min(tile.shape[0], y1), min(tile.shape[1], x1)
            crop = tile[y0:y1, x0:x1]
            if crop.size == 0:
                crop = _centre_crop(tile, 32)
        else:
            crop = _centre_crop(tile, LEGACY_NEG_PX)
        img = _minmax(crop)
        if img.shape != (out_px, out_px):
            img = resize(img, (out_px, out_px), anti_aliasing=True).astype(np.float32)
        return (img * INPUT_MAX).astype(np.float32)

    px = int(round(MATCHED_FOV_MM / spacing))
    crop = _centre_crop(tile, px).astype(np.float32)
    crop = resize(crop, (int(MATCHED_FOV_MM), int(MATCHED_FOV_MM)),
                  anti_aliasing=True, preserve_range=True)

    if arm == "G1":
        img = _minmax(crop) * INPUT_MAX
    elif arm in ("G2", "G3", "G4"):
        img = _hu_affine(crop)
    else:
        raise ValueError(f"unknown arm {arm!r}")

    return resize(img, (out_px, out_px), anti_aliasing=True,
                  preserve_range=True).astype(np.float32)


def select_rows(manifest, arm: str):
    rule = ARM_NEG_RULE[arm]
    return manifest[(manifest.label == 1) | (manifest.rule == rule)]
