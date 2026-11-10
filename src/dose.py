import json

import numpy as np
import pandas as pd

from arms import DOSE_FACTORS, build_patch
from paths import MANIFEST, RESULTS
from probe import PROBE_PX, SEED, auc, boot_auc_ci, features_one, grouped_scores, load_cohort

CNN_TAG = "{arm}_B0_128_single"
LEGACY_FACTOR = 3.66


def fit_log_errors(x, errors):
    y = np.log(errors)
    coef = np.polyfit(x, y, 1)
    pred = np.polyval(coef, x)
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(coef[0]), float(np.exp(coef[1])), float(r2)


def baseline_curve():
    man = pd.read_csv(MANIFEST)
    man = man[(man.label == 1) | (man.rule == "any")].reset_index(drop=True)
    tiles, rows = load_cohort(man)
    y = rows.label.to_numpy().astype(int)
    g = rows.patient_id.to_numpy()
    rng = np.random.default_rng(SEED)
    out = []
    for arm, f in DOSE_FACTORS.items():
        F = np.stack([features_one(build_patch(tiles[i], rows.iloc[i], arm, out_px=PROBE_PX))
                      for i in range(len(rows))])
        s = grouped_scores(F, y, g)
        acc = float(((s > 0.5).astype(int) == y).mean())
        lo, hi = boot_auc_ci(y, s, g, rng)
        out.append({"arm": arm, "factor": f, "model": "statistics-only baseline",
                    "n_patches": int(len(y)), "n_patients": int(len(np.unique(g))),
                    "auc": auc(y, s), "auc_lo": lo, "auc_hi": hi, "accuracy": acc,
                    "errors": int(round((1 - acc) * len(y)))})
        print(f"baseline {arm} f {f:4.2f}  AUC {out[-1]['auc']:.4f} [{lo:.4f}, {hi:.4f}]  "
              f"accuracy {acc:.4f}  errors {out[-1]['errors']}", flush=True)
    return out


def cnn_points():
    path = RESULTS / "arm_metrics.csv"
    if not path.exists():
        return []
    m = pd.read_csv(path).set_index("tag")
    k = "adaptive_midpoint__"
    out = []
    for arm, f in DOSE_FACTORS.items():
        tag = CNN_TAG.format(arm=arm)
        if tag not in m.index:
            continue
        r = m.loc[tag]
        out.append({"arm": arm, "factor": f, "model": "EfficientNetB0 128",
                    "n_patches": int(r.n_test), "n_patients": int(r.patients_test),
                    "auc": float(r.auc), "auc_lo": float(r.auc_lo), "auc_hi": float(r.auc_hi),
                    "accuracy": float(r[k + "accuracy"]), "errors": int(r[k + "fp"] + r[k + "fn"])})
        print(f"network  {arm} f {f:4.2f}  AUC {r.auc:.4f}  accuracy {r[k + 'accuracy']:.4f}  "
              f"errors {out[-1]['errors']}", flush=True)
    return out


def summary(rows):
    if len(rows) < 3:
        return None
    d = pd.DataFrame(rows)
    f, e = d.factor.to_numpy(), d.errors.to_numpy(float)
    if (e <= 0).any():
        return None
    p_slope, p_scale, p_r2 = fit_log_errors(np.log(f), e)
    x_slope, x_scale, x_r2 = fit_log_errors(f, e)
    strict = d[d.factor == 1.0]
    legacy = d[d.factor == LEGACY_FACTOR]
    ratio = (float(strict.errors.iloc[0] / legacy.errors.iloc[0])
             if len(strict) and len(legacy) and legacy.errors.iloc[0] > 0 else None)
    return {"points": int(len(d)),
            "power_law": {"scale": p_scale, "exponent": p_slope, "r2_log": p_r2,
                          "error_cut_per_doubling": float(2 ** -p_slope)},
            "exponential": {"scale": x_scale, "rate": x_slope, "r2_log": x_r2},
            "errors_strict_over_legacy": ratio}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    base = baseline_curve()
    net = cnn_points()
    pd.DataFrame(base + net).to_csv(RESULTS / "dose_response.csv", index=False)
    fits = {"statistics-only baseline": summary(base), "EfficientNetB0 128": summary(net)}
    with open(RESULTS / "dose_fit.json", "w") as fh:
        json.dump(fits, fh, indent=2)
    for model, s in fits.items():
        if s:
            print(f"{model}: errors = {s['power_law']['scale']:.0f} f^{s['power_law']['exponent']:.2f} "
                  f"(R2 {s['power_law']['r2_log']:.3f}); exponential R2 {s['exponential']['r2_log']:.3f}; "
                  f"strict over legacy {s['errors_strict_over_legacy']}")
    print(f"wrote {RESULTS / 'dose_response.csv'}, {RESULTS / 'dose_fit.json'}")


if __name__ == "__main__":
    main()
