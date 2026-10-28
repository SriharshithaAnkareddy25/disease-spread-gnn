"""Wave-onset detection, and error split by wave-onset vs flat weeks.
Person 3 deliverable 3 - the core evaluation contribution.

A week t is a wave onset for a district when all of:
    cases rising        : c[t] > c[t-1]
    rise accelerating   : (c[t]-c[t-1]) > (c[t-1]-c[t-2])
    growth is material  : c[t] >= GROWTH_RATIO * c[t-1]
    counts are not noise: c[t] >= MIN_CASES
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evaluation.sir import build_states, sir_forward
from training.data import prepare_data

ROOT = Path(__file__).resolve().parents[2]
STGNN_DIR = ROOT / "outputs" / "stgnn"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"
GRAPHS = ["geographic", "gravity", "learned", "hybrid"]
MIN_CASES = 20
GROWTH_RATIO = 1.20


def detect_onsets(new_confirmed, min_cases=MIN_CASES, ratio=GROWTH_RATIO):
    """Boolean [T, N] array: True where a district's curve turns upward."""
    c = np.nan_to_num(new_confirmed, nan=0.0)
    T, N = c.shape
    onset = np.zeros((T, N), dtype=bool)
    d1 = np.zeros_like(c)
    d1[1:] = c[1:] - c[:-1]
    rising = d1 > 0
    accelerating = np.zeros_like(rising)
    accelerating[2:] = d1[2:] > d1[1:-1]
    material = np.zeros_like(rising)
    material[1:] = c[1:] >= ratio * np.maximum(c[:-1], 1.0)
    big_enough = c >= min_cases
    onset[2:] = rising[2:] & accelerating[2:] & material[2:] & big_enough[2:]
    return onset


def metrics(pred, target, mask):
    valid = mask > 0.5
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "MAE": None, "RMSE": None}
    err = pred[valid] - target[valid]
    return {"n": n, "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2)))}


def plot_examples(new_conf, onset, weeks, codes, path, n_examples=4):
    totals = np.nan_to_num(new_conf).sum(axis=0)
    picks = np.argsort(totals)[-n_examples:][::-1]
    fig, axes = plt.subplots(n_examples, 1, figsize=(11, 2.3 * n_examples), sharex=True)
    for ax, j in zip(np.atleast_1d(axes), picks):
        series = np.nan_to_num(new_conf[:, j])
        ax.plot(weeks, series, color="#1D6A62", lw=1.4)
        marks = np.flatnonzero(onset[:, j])
        ax.scatter(weeks[marks], series[marks], color="#C4553B", s=26, zorder=3,
                   label="wave onset")
        ax.set_ylabel(f"code {codes[j]}", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
    plt.suptitle("Detected wave onsets (4 highest-caseload districts)", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main():
    data = prepare_data(lookback=8)
    M = data["M"].numpy()
    codes = data["codes"]
    sample_times = np.asarray(data["sample_times"], dtype=int)
    test_start, test_end = data["split"]["test"]

    S, I, new_conf, pop, weeks = build_states(codes)
    onset_all = detect_onsets(new_conf)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / "wave_onsets.png"
    plot_examples(new_conf, onset_all, weeks, codes, plot_path)
    print(f"Sanity plot: {plot_path}")
    print(f"Wave-onset weeks across the whole dataset: "
          f"{onset_all.sum() / onset_all.size * 100:.1f}%")

    test_k = np.arange(test_start, test_end)
    prev_i, curr_i = sample_times[test_k] - 1, sample_times[test_k]
    S_prev, I_prev = S[prev_i], I[prev_i]
    actual = np.nan_to_num(new_conf[curr_i])
    prev_cases = np.nan_to_num(new_conf[curr_i - 1])
    pop_b = pop[None, :]

    base_mask = M[test_k] * np.isfinite(new_conf[curr_i]) * np.isfinite(new_conf[curr_i - 1])
    onset_mask = base_mask * onset_all[curr_i]
    flat_mask = base_mask * (~onset_all[curr_i])
    print(f"\nTest entries: {int(base_mask.sum())} total | "
          f"{int(onset_mask.sum())} wave-onset "
          f"({onset_mask.sum()/base_mask.sum()*100:.1f}%) | {int(flat_mask.sum())} flat")

    preds = {"persistence": prev_cases}
    for graph in GRAPHS:
        path = STGNN_DIR / f"predictions_{graph}.npz"
        if not path.exists():
            continue
        beta = np.load(path)["predictions"].astype(np.float64)
        cases = sir_forward(beta, S_prev, I_prev, pop_b)
        preds[graph] = np.minimum(np.maximum(cases, 0.0), S_prev)

    rows = {name: {"all": metrics(p, actual, base_mask),
                   "wave_onset": metrics(p, actual, onset_mask),
                   "flat": metrics(p, actual, flat_mask)}
            for name, p in preds.items()}

    b_on = rows["persistence"]["wave_onset"]["MAE"]
    b_flat = rows["persistence"]["flat"]["MAE"]
    print()
    print(f"{'model':<14}{'MAE all':>11}{'MAE onset':>12}{'MAE flat':>11}{'onset vs base':>16}")
    print("-" * 64)
    for name, r in rows.items():
        on = r["wave_onset"]["MAE"]
        delta = "" if name == "persistence" else f"{(on-b_on)/b_on*100:+.1f}%"
        print(f"{name:<14}{r['all']['MAE']:>11.1f}{on:>12.1f}{r['flat']['MAE']:>11.1f}{delta:>16}")
    print("-" * 64)
    print(f"Persistence MAE is {b_on/b_flat:.1f}x worse on wave-onset weeks than on flat weeks.")
    print("That gap is the whole point: it is invisible in a single average.")

    out = OUTPUT_DIR / "wave_onset_comparison.json"
    out.write_text(json.dumps({"definition": {"min_cases": MIN_CASES,
                                              "growth_ratio": GROWTH_RATIO},
                               "test_weeks": [str(weeks[i].date()) for i in curr_i],
                               "onset_share_pct": float(onset_mask.sum()/base_mask.sum()*100),
                               "results": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
