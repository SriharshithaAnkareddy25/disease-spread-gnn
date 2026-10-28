"""Does the turning-point-weighted loss improve wave-onset accuracy?
Person 3 deliverable 4 (analysis half). Averages over seeds, because with
~70 training samples a single run swings enough to reverse the ranking.
"""

import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from evaluation.sir import build_states, sir_forward
from evaluation.turning_points import detect_onsets
from training.data import prepare_data

ROOT = Path(__file__).resolve().parents[2]
STGNN_DIR = ROOT / "outputs" / "stgnn"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"
PATTERN = re.compile(r"predictions_(\w+?)_w([\d.]+)_s(\d+)\.npz$")


def mae(pred, target, mask):
    valid = mask > 0.5
    return None if not valid.any() else float(np.mean(np.abs(pred[valid] - target[valid])))


def main():
    data = prepare_data(lookback=8)
    M, codes = data["M"].numpy(), data["codes"]
    sample_times = np.asarray(data["sample_times"], dtype=int)
    test_start, test_end = data["split"]["test"]

    S, I, new_conf, pop, weeks = build_states(codes)
    onset_all = detect_onsets(new_conf)
    test_k = np.arange(test_start, test_end)
    prev_i, curr_i = sample_times[test_k] - 1, sample_times[test_k]
    S_prev, I_prev, pop_b = S[prev_i], I[prev_i], pop[None, :]
    actual = np.nan_to_num(new_conf[curr_i])
    prev_cases = np.nan_to_num(new_conf[curr_i - 1])

    base_mask = M[test_k] * np.isfinite(new_conf[curr_i]) * np.isfinite(new_conf[curr_i - 1])
    onset_mask = base_mask * onset_all[curr_i]
    flat_mask = base_mask * (~onset_all[curr_i])
    base_onset = mae(prev_cases, actual, onset_mask)
    base_flat = mae(prev_cases, actual, flat_mask)
    base_all = mae(prev_cases, actual, base_mask)

    runs = defaultdict(list)
    for path in sorted(STGNN_DIR.glob("predictions_*_w*_s*.npz")):
        m = PATTERN.search(path.name)
        if not m:
            continue
        graph, weight, seed = m.group(1), float(m.group(2)), int(m.group(3))
        beta = np.load(path)["predictions"].astype(np.float64)
        cases = np.minimum(np.maximum(sir_forward(beta, S_prev, I_prev, pop_b), 0.0), S_prev)
        runs[(graph, weight)].append({"seed": seed,
                                      "onset": mae(cases, actual, onset_mask),
                                      "flat": mae(cases, actual, flat_mask),
                                      "all": mae(cases, actual, base_mask)})

    if not runs:
        print("No weighted runs found. Run train_weighted.py first."); return

    print(f"persistence       MAE all={base_all:.1f}  onset={base_onset:.1f}  flat={base_flat:.1f}\n")
    print(f"{'graph':<12}{'weight':>8}{'seeds':>7}{'MAE onset':>20}{'MAE flat':>12}{'onset vs base':>15}")
    print("-" * 76)
    summary = {}
    for (graph, weight) in sorted(runs):
        rs = runs[(graph, weight)]
        on = np.array([r["onset"] for r in rs]); fl = np.array([r["flat"] for r in rs])
        al = np.array([r["all"] for r in rs])
        delta = (on.mean() - base_onset) / base_onset * 100
        print(f"{graph:<12}{weight:>8g}{len(rs):>7}{on.mean():>13.1f} ±{on.std():<6.1f}"
              f"{fl.mean():>12.1f}{delta:>14.1f}%")
        summary[f"{graph}_w{weight:g}"] = {"n_seeds": len(rs),
            "onset_MAE_mean": float(on.mean()), "onset_MAE_std": float(on.std()),
            "flat_MAE_mean": float(fl.mean()), "all_MAE_mean": float(al.mean()),
            "onset_vs_persistence_pct": float(delta)}
    print("-" * 76)

    print("\nEffect of the turning-point weighting (same graph, same seeds):")
    for graph in sorted({g for g, _ in runs}):
        weights = sorted({w for g, w in runs if g == graph})
        if 1.0 not in weights:
            continue
        b = np.mean([r["onset"] for r in runs[(graph, 1.0)]])
        for w in weights:
            if w == 1.0:
                continue
            a = np.mean([r["onset"] for r in runs[(graph, w)]])
            print(f"  {graph:<12} w=1 -> w={w:g}: onset MAE {b:.1f} -> {a:.1f}  "
                  f"({(a-b)/b*100:+.1f}%, {'better' if a < b else 'worse'})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "weighted_ablation.json"
    out.write_text(json.dumps({"units": "weekly new confirmed cases per district",
        "persistence": {"all": base_all, "onset": base_onset, "flat": base_flat},
        "runs": summary}, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
