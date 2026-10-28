"""Build the result figures for Person 3's contribution. Deliverable 7.

    fig1_onset_vs_flat.png       headline: error at wave onset vs flat weeks,
                                 under rolling-origin evaluation
    fig2_weighting_ablation.png  turning-point weighting across all four
                                 graphs, mean +/- sd over seeds
    fig3_rt_distribution.png     implied Rt, weighted vs unweighted, against
                                 the Rt = 1 epidemic threshold
    fig4_forecast_examples.png   predicted vs actual weekly cases

Any figure whose inputs are missing is skipped with a note, not a crash.
"""

import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evaluation.sir import build_states, sir_forward
from evaluation.turning_points import detect_onsets
from training.data import prepare_data

ROOT = Path(__file__).resolve().parents[2]
STGNN_DIR = ROOT / "outputs" / "stgnn"
EVAL_DIR = ROOT / "outputs" / "evaluation"
TEAL, TERRA, MUST, SLATE = "#1D6A62", "#C4553B", "#D9A441", "#6E7F7C"


def load_json(name):
    path = EVAL_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fig1_onset_vs_flat():
    files = sorted(EVAL_DIR.glob("rolling_origin_*.json"))
    if not files:
        print("  fig1 skipped: no rolling_origin_*.json"); return
    panels = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.6), squeeze=False)
    for ax, rep in zip(axes[0], panels):
        rows = rep["results"]
        names = list(rows.keys())
        labels = [n.replace("_w1", " (unweighted)").replace("_w5", " (weighted)") for n in names]
        onset = [rows[n]["onset"] for n in names]
        flat = [rows[n]["flat"] for n in names]
        x = np.arange(len(names))
        ax.bar(x - 0.19, onset, 0.38, label="wave-onset weeks", color=TERRA)
        ax.bar(x + 0.19, flat, 0.38, label="flat weeks", color=TEAL)
        ax.axhline(rows["persistence"]["onset"], color=TERRA, ls="--", lw=1, alpha=0.6)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
        ax.set_ylabel("MAE (weekly cases per district)", fontsize=9)
        ax.set_title(f"{rep['graph']} graph — {rep['n_origins']} rolling origins", fontsize=10)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    plt.suptitle("Forecast error at wave onset vs flat weeks "
                 "(dashed line = persistence at onset)", fontsize=11.5)
    plt.tight_layout(); plt.savefig(EVAL_DIR / "fig1_onset_vs_flat.png", dpi=140); plt.close()
    print("  wrote fig1_onset_vs_flat.png")


def fig2_weighting_ablation():
    rep = load_json("weighted_ablation.json")
    if rep is None:
        print("  fig2 skipped: no weighted_ablation.json"); return
    runs = rep["runs"]
    graphs = sorted({k.rsplit("_w", 1)[0] for k in runs})
    base = rep["persistence"]["onset"]
    un_mean = [runs.get(f"{g}_w1", {}).get("onset_MAE_mean") for g in graphs]
    un_std = [runs.get(f"{g}_w1", {}).get("onset_MAE_std", 0) for g in graphs]
    w_mean = [runs.get(f"{g}_w5", {}).get("onset_MAE_mean") for g in graphs]
    w_std = [runs.get(f"{g}_w5", {}).get("onset_MAE_std", 0) for g in graphs]
    if any(v is None for v in un_mean + w_mean):
        print("  fig2 skipped: need both w1 and w5 for every graph"); return
    x = np.arange(len(graphs))
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(x - 0.19, un_mean, 0.38, yerr=un_std, capsize=4, label="uniform loss (w=1)", color=SLATE)
    ax.bar(x + 0.19, w_mean, 0.38, yerr=w_std, capsize=4,
           label="turning-point weighted (w=5)", color=TERRA)
    ax.axhline(base, color="black", ls="--", lw=1.2, label="persistence baseline")
    for xi, (a, b) in enumerate(zip(un_mean, w_mean)):
        ax.annotate(f"{(b-a)/a*100:+.1f}%", (xi + 0.19, b), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8.5, color=TERRA)
    ax.set_xticks(x); ax.set_xticklabels(graphs)
    ax.set_ylabel("MAE on wave-onset weeks", fontsize=9.5)
    ax.set_title("Turning-point weighting improves wave-onset accuracy "
                 "for every graph variant", fontsize=11)
    ax.legend(fontsize=8.5); ax.grid(axis="y", alpha=0.25)
    plt.tight_layout(); plt.savefig(EVAL_DIR / "fig2_weighting_ablation.png", dpi=140); plt.close()
    print("  wrote fig2_weighting_ablation.png")


def _collect_betas():
    found = defaultdict(list)
    for path in sorted(STGNN_DIR.glob("predictions_*_w*_s*.npz")):
        parts = path.stem.replace("predictions_", "").rsplit("_", 2)
        found[f"{parts[0]}_{parts[1]}"].append(path)
    return {name: np.mean([np.load(p)["predictions"].astype(np.float64) for p in paths], axis=0)
            for name, paths in found.items()}


def fig3_rt(S_prev, pop_b, valid):
    betas = _collect_betas()
    if not betas:
        print("  fig3 skipped: no weighted prediction files"); return
    graphs = sorted({k.rsplit("_w", 1)[0] for k in betas})
    series = []
    for g in graphs:
        for suffix, color, label in (("w1", SLATE, "uniform"), ("w5", TERRA, "weighted")):
            key = f"{g}_{suffix}"
            if key in betas:
                series.append((f"{g}\n{label}", (betas[key] * S_prev / pop_b)[valid], color))
    if not series:
        print("  fig3 skipped: nothing to plot"); return
    fig, ax = plt.subplots(figsize=(1.05 * len(series) + 4, 4.4))
    positions = np.arange(len(series))
    bp = ax.boxplot([s[1] for s in series], positions=positions, widths=0.6,
                    showfliers=False, patch_artist=True)
    for patch, (_, _, color) in zip(bp["boxes"], series):
        patch.set_facecolor(color); patch.set_alpha(0.65)
    for median in bp["medians"]:
        median.set_color("black")
    ax.axhline(1.0, color=TERRA, ls="--", lw=1.4)
    ax.annotate("Rt = 1  (epidemic threshold)", (len(series) - 0.4, 1.0),
                textcoords="offset points", xytext=(0, 6), ha="right",
                fontsize=8.5, color=TERRA)
    ax.set_xticks(positions); ax.set_xticklabels([s[0] for s in series], fontsize=8)
    ax.set_ylabel("implied Rt", fontsize=9.5)
    ax.set_title("Implied reproduction number during the Omicron test window\n"
                 "uniform-loss models sit below 1, which a growing wave cannot do",
                 fontsize=10.5)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout(); plt.savefig(EVAL_DIR / "fig3_rt_distribution.png", dpi=140); plt.close()
    print("  wrote fig3_rt_distribution.png")


def fig4_examples(codes, weeks, curr_i, actual, prev_cases, S_prev, I_prev,
                  pop_b, valid, onset_now, n=4):
    betas = _collect_betas()
    pick_key = next((k for k in betas if k.endswith("_w5")), None)
    if pick_key is None:
        print("  fig4 skipped: no weighted run"); return
    cases = np.minimum(np.maximum(sir_forward(betas[pick_key], S_prev, I_prev, pop_b), 0.0), S_prev)
    picks = np.argsort(np.where(valid, actual, 0).sum(axis=0))[-n:][::-1]
    xs = [weeks[i].date() for i in curr_i]
    fig, axes = plt.subplots(n, 1, figsize=(9.6, 2.3 * n), sharex=True)
    for ax, j in zip(np.atleast_1d(axes), picks):
        ax.plot(xs, actual[:, j], color=TEAL, lw=1.8, marker="o", ms=3.5, label="actual")
        ax.plot(xs, cases[:, j], color=TERRA, lw=1.6, ls="--", marker="s", ms=3.5,
                label=f"predicted ({pick_key})")
        ax.plot(xs, prev_cases[:, j], color=SLATE, lw=1.1, ls=":", label="persistence")
        marks = np.flatnonzero(onset_now[:, j])
        if marks.size:
            ax.scatter([xs[m] for m in marks], actual[marks, j], color=MUST, s=58,
                       zorder=4, marker="*", label="wave onset")
        ax.set_ylabel(f"code {codes[j]}", fontsize=8)
        ax.grid(alpha=0.22); ax.legend(fontsize=7, ncol=4, loc="upper left")
    plt.suptitle("Predicted vs actual weekly cases, highest-caseload districts", fontsize=11)
    plt.tight_layout(); plt.savefig(EVAL_DIR / "fig4_forecast_examples.png", dpi=140); plt.close()
    print("  wrote fig4_forecast_examples.png")


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    print("Building figures:")
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
    valid = (M[test_k] * np.isfinite(new_conf[curr_i]) * np.isfinite(new_conf[curr_i - 1])) > 0.5

    fig1_onset_vs_flat()
    fig2_weighting_ablation()
    fig3_rt(S_prev, pop_b, valid)
    fig4_examples(codes, weeks, curr_i, actual, prev_cases, S_prev, I_prev,
                  pop_b, valid, onset_all[curr_i])
    print(f"\nAll figures in {EVAL_DIR}")


if __name__ == "__main__":
    main()
