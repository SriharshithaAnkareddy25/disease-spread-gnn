"""Turn forecasts into a ranked high-risk district list, and derive Rt.
Person 3 deliverable 5.

A health department does not act on an error figure - it acts on "which
districts do we send resources to this week". That is a ranking problem, and
ranking quality is essentially absent from this literature.

    burden = predicted new cases                  (where the load will be)
    surge  = predicted cases / last week's cases  (where it is accelerating)

Scored against binary ground truth: did that district actually hit a wave
onset that week?  Baselines: persistence ranking, and ranking by population
(because any useful model must beat "just send resources to the biggest").

Rt = beta * S / N, assuming a one-week infectious period (gamma = 1/week).
"""

import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from evaluation.sir import build_states, sir_forward
from evaluation.turning_points import detect_onsets
from training.data import prepare_data

ROOT = Path(__file__).resolve().parents[2]
STGNN_DIR = ROOT / "outputs" / "stgnn"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"
KS = [10, 20, 50]


def precision_at_k(scores, relevant, valid, k):
    idx = np.flatnonzero(valid)
    if idx.size < k:
        return None
    top = idx[np.argsort(scores[idx])[::-1][:k]]
    return float(relevant[top].mean())


def ndcg_at_k(scores, gains, valid, k):
    idx = np.flatnonzero(valid)
    if idx.size < k:
        return None
    g = np.maximum(gains[idx], 0.0)
    order = np.argsort(scores[idx])[::-1][:k]
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((g[order] * discount).sum())
    ideal = float((np.sort(g)[::-1][:k] * discount).sum())
    return dcg / ideal if ideal > 0 else None


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
    valid = (M[test_k] * np.isfinite(new_conf[curr_i])
             * np.isfinite(new_conf[curr_i - 1])) > 0.5
    relevant = onset_all[curr_i].astype(float)
    actual_growth = actual / np.maximum(prev_cases, 1.0)

    print(f"Test weeks: {len(test_k)} | mean valid districts/week: {int(valid.sum(axis=1).mean())}")
    print(f"Mean actual surge rate: {relevant[valid].mean()*100:.1f}% "
          f"(the score a random ranking would get)")

    rankers = {
        "persistence": {"burden": prev_cases,
                        "surge": prev_cases / np.maximum(np.nan_to_num(new_conf[curr_i - 2]), 1.0)},
        "population": {"burden": np.broadcast_to(pop_b, actual.shape).copy(),
                       "surge": np.broadcast_to(pop_b, actual.shape).copy()},
    }

    found = defaultdict(list)
    for path in sorted(STGNN_DIR.glob("predictions_*_w*_s*.npz")):
        stem = path.stem.replace("predictions_", "")
        parts = stem.rsplit("_", 2)
        found[f"{parts[0]}_{parts[1]}"].append(path)

    rt_store = {}
    for name, paths in sorted(found.items()):
        betas = np.mean([np.load(p)["predictions"].astype(np.float64) for p in paths], axis=0)
        cases = np.minimum(np.maximum(sir_forward(betas, S_prev, I_prev, pop_b), 0.0), S_prev)
        rankers[name] = {"burden": cases, "surge": cases / np.maximum(prev_cases, 1.0)}
        rt_store[name] = betas * S_prev / pop_b

    results = {}
    for name, scores in rankers.items():
        entry = {}
        for mode in ("burden", "surge"):
            per_k = {}
            for k in KS:
                precs = [p for t in range(len(test_k))
                         if (p := precision_at_k(scores[mode][t], relevant[t], valid[t], k)) is not None]
                ndcgs = [n for t in range(len(test_k))
                         if (n := ndcg_at_k(scores[mode][t], actual_growth[t], valid[t], k)) is not None]
                per_k[f"P@{k}"] = float(np.mean(precs)) if precs else None
                per_k[f"NDCG@{k}"] = float(np.mean(ndcgs)) if ndcgs else None
            entry[mode] = per_k
        results[name] = entry

    for mode in ("surge", "burden"):
        print(f"\nranking by predicted {'GROWTH' if mode=='surge' else 'CASE BURDEN'}")
        header = f"{'ranker':<20}" + "".join(f"{'P@'+str(k):>9}" for k in KS) \
                 + "".join(f"{'NDCG@'+str(k):>11}" for k in KS)
        print(header); print("-" * len(header))
        for name, entry in results.items():
            row = f"{name:<20}"
            for k in KS:
                v = entry[mode][f"P@{k}"]
                row += f"{v:>9.3f}" if v is not None else f"{'-':>9}"
            for k in KS:
                v = entry[mode][f"NDCG@{k}"]
                row += f"{v:>11.3f}" if v is not None else f"{'-':>11}"
            print(row)
        print("-" * len(header))

    print("\nImplied Rt (effective reproduction number, gamma = 1/week):")
    for name, rt in sorted(rt_store.items()):
        v = rt[valid]
        print(f"  {name:<20} mean={v.mean():.2f}  median={np.median(v):.2f}  "
              f"share Rt>1: {float((v > 1).mean())*100:.1f}%")
    print("  (Omicron test window - a high share above 1 is expected)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "risk_ranking.json"
    out.write_text(json.dumps({"ks": KS,
        "test_weeks": [str(weeks[i].date()) for i in curr_i],
        "base_surge_rate": float(relevant[valid].mean()),
        "rt_note": "Rt = beta * S / N, assuming gamma = 1/week",
        "rt_summary": {n: {"mean": float(rt[valid].mean()),
                           "share_above_1": float((rt[valid] > 1).mean())}
                       for n, rt in rt_store.items()},
        "results": results}, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
