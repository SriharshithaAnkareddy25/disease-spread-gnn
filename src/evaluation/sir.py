"""Differentiable SIR forward step: turn predicted beta into predicted cases.
Person 3 deliverable 2 - the missing link in the pipeline.

data.py builds the beta target by INVERTING the SIR incidence equation:
    beta[t] = new_cases[t] * N / (S[t-1] * I[t-1])
This module runs it FORWARDS, which is what makes the model useful:
    new_cases[t] = beta[t] * S[t-1] * I[t-1] / N
with N = population, I = Confirmed-Recovered-Deceased, S = Population-Confirmed.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from training.data import prepare_data

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
STGNN_DIR = ROOT / "outputs" / "stgnn"
OUTPUT_DIR = ROOT / "outputs" / "evaluation"
GRAPHS = ["geographic", "gravity", "learned", "hybrid"]


def sir_forward(beta, susceptible, infected, population):
    """new_cases = beta * S * I / N.  Works with numpy or torch."""
    return beta * susceptible * infected / population


class SIRLayer(nn.Module):
    """Differentiable SIR incidence step, to sit on the end of the GNN."""
    def forward(self, beta, susceptible, infected, population):
        cases = sir_forward(beta, susceptible, infected, population)
        return torch.clamp(cases, min=0.0).minimum(susceptible)


def build_states(codes):
    """Rebuild the S / I states the beta target was derived from."""
    nodes = pd.read_csv(DATA / "nodes.csv").set_index("censuscode")
    cases = pd.read_csv(DATA / "weekly_case_features.csv", parse_dates=["week"])
    cases = cases[cases["censuscode"].isin(codes)]
    weeks = pd.DatetimeIndex(sorted(cases["week"].unique()))
    T, N = len(weeks), len(codes)

    cumulative = np.full((T, N, 3), np.nan, dtype=np.float64)
    new_confirmed = np.full((T, N), np.nan, dtype=np.float64)
    for j, code in enumerate(codes):
        sub = cases[cases["censuscode"] == code].set_index("week").reindex(weeks)
        for k, col in enumerate(["Confirmed", "Recovered", "Deceased"]):
            cumulative[:, j, k] = pd.to_numeric(sub[col], errors="coerce")
        new_confirmed[:, j] = pd.to_numeric(sub["new_confirmed"], errors="coerce")
    for j in range(N):
        cumulative[:, j, :] = pd.DataFrame(cumulative[:, j, :]).ffill().fillna(0).to_numpy()

    confirmed, recovered, deceased = (cumulative[:, :, i] for i in range(3))
    population = nodes.reindex(codes)["Population"].to_numpy(dtype=np.float64)
    infected = np.maximum(confirmed - recovered - deceased, 0.0)
    susceptible = np.maximum(population[None, :] - confirmed, 1.0)
    return susceptible, infected, new_confirmed, population, weeks


def metrics(pred, target, mask):
    valid = mask > 0.5
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "MAE": None, "RMSE": None}
    err = pred[valid] - target[valid]
    return {"n": n, "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2)))}


def main():
    data = prepare_data(lookback=8)
    Y = data["Y"].numpy().astype(np.float64)
    M = data["M"].numpy()
    codes = data["codes"]
    sample_times = np.asarray(data["sample_times"], dtype=int)
    test_start, test_end = data["split"]["test"]

    S, I, new_conf, pop, weeks = build_states(codes)
    test_k = np.arange(test_start, test_end)
    prev_i, curr_i = sample_times[test_k] - 1, sample_times[test_k]
    S_prev, I_prev = S[prev_i], I[prev_i]
    actual_cases = new_conf[curr_i]
    pop_b = pop[None, :]

    mask = M[test_k] * np.isfinite(actual_cases)
    actual_cases = np.nan_to_num(actual_cases)
    print(f"Test weeks: {[str(weeks[i].date()) for i in curr_i]}")
    print(f"Districts per week: {int(mask.sum(axis=1).mean())} (mean observed)")

    rt = sir_forward(Y[test_k], S_prev, I_prev, pop_b)
    exact = float(np.mean(np.abs(rt - actual_cases)[mask > 0.5] < 1.0) * 100)
    print(f"\nRound-trip check (true beta -> cases): {exact:.1f}% within 1 case")
    print("  (shortfall = beta clipped at beta_max=2.0 when the target was built)")

    rows = {}
    prev_cases = new_conf[curr_i - 1]
    pers_mask = mask * np.isfinite(prev_cases)
    prev_cases = np.nan_to_num(prev_cases)
    rows["persistence (cases)"] = metrics(prev_cases, actual_cases, pers_mask)

    for graph in GRAPHS:
        path = STGNN_DIR / f"predictions_{graph}.npz"
        if not path.exists():
            print(f"  (skipping {graph}: not found)")
            continue
        beta_pred = np.load(path)["predictions"].astype(np.float64)
        cases_pred = sir_forward(beta_pred, S_prev, I_prev, pop_b)
        cases_pred = np.minimum(np.maximum(cases_pred, 0.0), S_prev)
        rows[graph] = metrics(cases_pred, actual_cases, pers_mask)

    base = rows["persistence (cases)"]["MAE"]
    print()
    print(f"{'model':<22}{'n':>7}{'MAE':>12}{'RMSE':>12}{'vs base':>11}{'beats?':>9}")
    print("-" * 73)
    for name, m in rows.items():
        if m["MAE"] is None:
            continue
        is_base = name.startswith("persistence")
        shown = "" if is_base else f"{(m['MAE'] - base) / base * 100:+.1f}%"
        beats = "" if is_base else ("YES" if m["MAE"] < base else "no")
        print(f"{name:<22}{m['n']:>7}{m['MAE']:>12.1f}{m['RMSE']:>12.1f}{shown:>11}{beats:>9}")
    print("-" * 73)
    print("Units are WEEKLY CASES per district. Negative = better than persistence.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "case_space_comparison.json"
    out.write_text(json.dumps({"units": "weekly new confirmed cases per district",
                               "test_weeks": [str(weeks[i].date()) for i in curr_i],
                               "round_trip_within_1_case_pct": exact,
                               "results": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
