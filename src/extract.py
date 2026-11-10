import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pydicom
import pylidc as pl
from scipy import ndimage
from skimage import measure, morphology

from paths import CACHE, CENSUS, MANIFEST, SCAN_LOG, scan_file

TILE_PX = 224
MATCHED_FOV_MM = 64.0
LEGACY_NEG_PX = 128
LEGACY_PAD_PX = 10
SLICES_PER_NODULE = 3
MIN_ANNOTATIONS = 3
AIR_HU = -1024
LUNG_HU = -400
STRUCTURE_HU = -300
LUNG_FRACTION = 0.80
SEED = 42

META_COLS = [
    "patient_id", "scan_id", "label", "rule", "nodule_idx", "y", "x", "z",
    "n_annotations", "diameter_mm", "malignancy_median", "subtlety_median",
    "pixel_spacing", "slice_thickness", "slice_spacing",
    "bbox_y0", "bbox_y1", "bbox_x0", "bbox_x1", "bbox_clipped",
    "legacy_resample_factor", "matched_resample_factor",
    "lung_fraction_window", "lung_fraction_tile", "tile_pad_fraction",
]

SCAN_LOG_COLS = ["patient_id", "scan_id", "status", "seconds", "n_patches",
                 "clusters_total", "clusters_qualifying", "positives",
                 "slices_short", "bbox_clipped",
                 "neg_any", "neg_lung", "neg_hard", "neg_wanted", "note"]


def lung_mask(sl: np.ndarray) -> np.ndarray:
    air = sl < LUNG_HU
    labels = measure.label(air, connectivity=1)
    if labels.max() == 0:
        return np.zeros_like(air, dtype=bool)

    border = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    border.discard(0)
    interior = np.isin(labels, list(border), invert=True) & air
    if not interior.any():
        return np.zeros_like(air, dtype=bool)

    labels = measure.label(interior, connectivity=1)
    min_area = 0.005 * sl.size
    keep = [p.label for p in measure.regionprops(labels) if p.area >= min_area]
    if not keep:
        return np.zeros_like(air, dtype=bool)
    mask = np.isin(labels, keep)

    mask = morphology.binary_closing(mask, morphology.disk(6))
    return ndimage.binary_fill_holes(mask)


def window_lung_fraction(mask: np.ndarray, cy: int, cx: int, px: int) -> float:
    h = max(1, px // 2)
    win = mask[max(0, cy - h):cy + h, max(0, cx - h):cx + h]
    return float(win.mean()) if win.size else 0.0


def cut_tile(sl: np.ndarray, cy: int, cx: int, size: int = TILE_PX):
    half = size // 2
    y0, y1 = cy - half, cy - half + size
    x0, x1 = cx - half, cx - half + size
    tile = np.full((size, size), AIR_HU, dtype=np.int16)

    sy0, sy1 = max(0, y0), min(sl.shape[0], y1)
    sx0, sx1 = max(0, x0), min(sl.shape[1], x1)
    if sy1 > sy0 and sx1 > sx0:
        tile[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = sl[sy0:sy1, sx0:sx1]

    filled = max(0, sy1 - sy0) * max(0, sx1 - sx0)
    return tile, 1.0 - filled / float(size * size)


def slice_indices(z_lo: int, z_hi: int, n_slices: int, thickness: float, spacing: float):
    step = max(1, int(np.ceil(float(thickness) / max(float(spacing), 1e-6))))
    centre = (z_lo + z_hi - 1) // 2
    offsets = [0]
    while len(offsets) < n_slices:
        k = (len(offsets) + 1) // 2
        offsets.append(k * step if len(offsets) % 2 else -k * step)
    picked = [centre + o for o in offsets[:n_slices]]
    return sorted({z for z in picked if z_lo <= z < z_hi})


def scan_acquisition_metadata(scan) -> dict:
    out = {"manufacturer": "", "model": "", "kernel": "", "kvp": None}
    try:
        path = Path(scan.get_path_to_dicom_files())
        first = sorted(path.glob("*.dcm"))[0]
        ds = pydicom.dcmread(str(first), stop_before_pixels=True)
        out["manufacturer"] = str(getattr(ds, "Manufacturer", "") or "").strip()
        out["model"] = str(getattr(ds, "ManufacturerModelName", "") or "").strip()
        out["kernel"] = str(getattr(ds, "ConvolutionKernel", "") or "").strip()
        kvp = getattr(ds, "KVP", None)
        out["kvp"] = float(kvp) if kvp is not None else None
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"[:120]
    return out


def extract_scan(scan, rng: np.random.Generator):
    vol = scan.to_volume(verbose=False)
    ny, nx, nz = vol.shape
    spacing = float(scan.pixel_spacing)
    thickness = float(scan.slice_thickness or spacing)
    z_spacing = float(scan.slice_spacing or thickness)
    acq = scan_acquisition_metadata(scan)

    matched_px = int(round(MATCHED_FOV_MM / spacing))
    stats = {
        "clusters_total": 0, "clusters_qualifying": 0, "nodules_used": 0,
        "positives": 0, "slices_short": 0, "bbox_clipped": 0,
        "neg_any": 0, "neg_lung": 0, "neg_hard": 0, "neg_wanted": 0,
    }

    clusters = scan.cluster_annotations(verbose=False)
    stats["clusters_total"] = len(clusters)
    qualifying = [c for c in clusters if len(c) >= MIN_ANNOTATIONS]
    stats["clusters_qualifying"] = len(qualifying)
    if not qualifying:
        return np.zeros((0, TILE_PX, TILE_PX), np.int16), [], stats

    tiles, rows = [], []
    nodule_z_ranges = []
    masks: dict[int, np.ndarray] = {}

    def mask_for(z):
        if z not in masks:
            masks[z] = lung_mask(vol[:, :, z])
        return masks[z]

    for idx, cluster in enumerate(qualifying):
        ann = cluster[0]
        by, bx, bz = ann.bbox()
        nodule_z_ranges.append((bz.start, bz.stop))
        zs = slice_indices(bz.start, bz.stop, SLICES_PER_NODULE, thickness, z_spacing)
        if len(zs) < SLICES_PER_NODULE:
            stats["slices_short"] += 1

        cy, cx = (by.start + by.stop) // 2, (bx.start + bx.stop) // 2
        py0, py1 = max(0, by.start - LEGACY_PAD_PX), min(ny, by.stop + LEGACY_PAD_PX)
        px0, px1 = max(0, bx.start - LEGACY_PAD_PX), min(nx, bx.stop + LEGACY_PAD_PX)
        bh, bw = py1 - py0, px1 - px0
        clipped = bool(bh > TILE_PX or bw > TILE_PX)
        stats["bbox_clipped"] += int(clipped)

        diam = float(np.median([a.diameter for a in cluster]))
        mal = float(np.median([a.malignancy for a in cluster]))
        sub = float(np.median([a.subtlety for a in cluster]))

        for z in zs:
            tile, pad_frac = cut_tile(vol[:, :, z], cy, cx)
            m = mask_for(z)
            half = TILE_PX // 2
            ys = slice(max(0, cy - half), min(ny, cy + half))
            xs = slice(max(0, cx - half), min(nx, cx + half))
            lf_win = window_lung_fraction(m, cy, cx, matched_px)
            lf_tile = float(m[ys, xs].mean()) if m.any() else 0.0

            tiles.append(tile)
            rows.append({
                "patient_id": scan.patient_id, "scan_id": int(scan.id), "label": 1,
                "rule": "nodule", "nodule_idx": idx, "y": cy, "x": cx, "z": z,
                "n_annotations": len(cluster), "diameter_mm": diam,
                "malignancy_median": mal, "subtlety_median": sub,
                "pixel_spacing": spacing, "slice_thickness": thickness,
                "slice_spacing": z_spacing,
                "bbox_y0": py0 - (cy - half), "bbox_y1": py1 - (cy - half),
                "bbox_x0": px0 - (cx - half), "bbox_x1": px1 - (cx - half),
                "bbox_clipped": clipped,
                "legacy_resample_factor": 128.0 / max(bh, bw, 1),
                "matched_resample_factor": 224.0 / max(matched_px, 1),
                "lung_fraction_window": lf_win, "lung_fraction_tile": lf_tile,
                "tile_pad_fraction": pad_frac,
                **acq,
            })
            stats["positives"] += 1

    n_per_rule = max(1, stats["positives"])
    stats["neg_wanted"] = n_per_rule
    nodule_z = {z for lo, hi in nodule_z_ranges for z in range(lo, hi)}
    free_z = [z for z in range(nz) if z not in nodule_z]
    if free_z:
        half = TILE_PX // 2
        margin = max(half, LEGACY_NEG_PX // 2, matched_px // 2) + 1

        def emit(rule, cy, cx, z):
            tile, pad_frac = cut_tile(vol[:, :, z], cy, cx)
            m = mask_for(z)
            ys = slice(max(0, cy - half), min(ny, cy + half))
            xs = slice(max(0, cx - half), min(nx, cx + half))
            tiles.append(tile)
            rows.append({
                "patient_id": scan.patient_id, "scan_id": int(scan.id), "label": 0,
                "rule": rule, "nodule_idx": -1, "y": cy, "x": cx, "z": z,
                "n_annotations": 0, "diameter_mm": np.nan,
                "malignancy_median": np.nan, "subtlety_median": np.nan,
                "pixel_spacing": spacing, "slice_thickness": thickness,
                "slice_spacing": z_spacing,
                "bbox_y0": -1, "bbox_y1": -1, "bbox_x0": -1, "bbox_x1": -1,
                "bbox_clipped": False,
                "legacy_resample_factor": 1.0,
                "matched_resample_factor": 224.0 / max(matched_px, 1),
                "lung_fraction_window": window_lung_fraction(m, cy, cx, matched_px),
                "lung_fraction_tile": float(m[ys, xs].mean()) if m.any() else 0.0,
                "tile_pad_fraction": pad_frac, **acq,
            })
            stats[f"neg_{rule}"] += 1

        for _ in range(n_per_rule):
            z = int(rng.choice(free_z))
            emit("any", int(rng.integers(margin, ny - margin)),
                 int(rng.integers(margin, nx - margin)), z)

        rng.shuffle(free_z)
        got_lung = got_hard = 0
        for z in free_z:
            if got_lung >= n_per_rule and got_hard >= n_per_rule:
                break
            m = mask_for(z)
            if not m.any():
                continue
            inner = np.zeros_like(m)
            inner[margin:ny - margin, margin:nx - margin] = True
            sl = vol[:, :, z]

            if got_lung < n_per_rule:
                cand = np.argwhere(m & inner)
                rng.shuffle(cand)
                for cy, cx in cand[:40]:
                    win = m[cy - matched_px // 2:cy + matched_px // 2,
                            cx - matched_px // 2:cx + matched_px // 2]
                    if win.size and win.mean() >= LUNG_FRACTION:
                        emit("lung", int(cy), int(cx), z)
                        got_lung += 1
                        break

            if got_hard < n_per_rule:
                cand = np.argwhere(m & inner & (sl > STRUCTURE_HU))
                rng.shuffle(cand)
                for cy, cx in cand[:40]:
                    if window_lung_fraction(m, int(cy), int(cx), matched_px) >= LUNG_FRACTION:
                        emit("hard", int(cy), int(cx), z)
                        got_hard += 1
                        break

    if not tiles:
        return np.zeros((0, TILE_PX, TILE_PX), np.int16), [], stats
    return np.stack(tiles).astype(np.int16), rows, stats


def eligible_scans(census: dict) -> list[dict]:
    return [s for s in census["scans"]
            if any(c["n_ann"] >= MIN_ANNOTATIONS for c in s["clusters"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)

    with open(CENSUS) as fh:
        census = json.load(fh)
    eligible = eligible_scans(census)
    if args.limit:
        eligible = eligible[: args.limit]
    wanted = {(s["patient_id"], s["scan_id"]) for s in eligible}
    print(f"eligible scans: {len(eligible)}  patients: {len({p for p, _ in wanted})}",
          flush=True)

    by_id = {int(s.id): s for s in pl.query(pl.Scan).all()}
    t0 = time.time()

    scan_log = open(SCAN_LOG, "a", newline="")
    scan_writer = csv.writer(scan_log)
    if scan_log.tell() == 0:
        scan_writer.writerow(SCAN_LOG_COLS)

    for i, rec in enumerate(eligible, 1):
        pid, sid = rec["patient_id"], rec["scan_id"]
        out = scan_file(pid, sid)
        if out.exists() and not args.force:
            continue

        t = time.time()
        try:
            scan = by_id[sid]
            rng = np.random.default_rng(SEED * 1_000_003 + sid)
            tiles, rows, stats = extract_scan(scan, rng)
            meta = {c: np.array([r.get(c) for r in rows]) for c in META_COLS} if rows else {}
            np.savez_compressed(out, tiles=tiles,
                                **{f"meta_{k}": v for k, v in meta.items()})
            if rows:
                new = not MANIFEST.exists()
                with open(MANIFEST, "a", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                    if new:
                        writer.writeheader()
                    writer.writerows(rows)
            scan_writer.writerow([pid, sid, "ok", f"{time.time() - t:.1f}", len(tiles),
                                  stats["clusters_total"], stats["clusters_qualifying"],
                                  stats["positives"], stats["slices_short"],
                                  stats["bbox_clipped"], stats["neg_any"],
                                  stats["neg_lung"], stats["neg_hard"],
                                  stats["neg_wanted"], ""])
        except Exception as exc:
            scan_writer.writerow([pid, sid, "error", f"{time.time() - t:.1f}", 0,
                                  "", "", "", "", "", "", "", "", "",
                                  f"{type(exc).__name__}: {exc}"[:200]])
        scan_log.flush()

        if i % 10 == 0 or i == len(eligible):
            elapsed = time.time() - t0
            print(f"{i}/{len(eligible)}  {elapsed / 60:.1f}min  "
                  f"eta {elapsed / i * (len(eligible) - i) / 60:.1f}min", flush=True)

    scan_log.close()
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
