"""Bundle model outputs into the small files the Streamlit demo reads.

Run after training (needs outputs/stgnn/predictions_<graph>.npz):

    PYTHONPATH=src python src/demo/export_demo_data.py

Writes outputs/demo/ (tracked in git, so the demo runs without retraining):
  districts.geojson   simplified district polygons (the full file is 27MB)
  forecasts.csv       per graph x test week x district: actual, predicted and
                      persistence cases, predicted beta, implied Rt, onset flag
  learned_edges.csv   each district's 3 strongest learned-graph neighbours
  history.csv         weekly new cases for every district, for context plots
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from evaluation.sir import build_states, sir_forward
from evaluation.turning_points import detect_onsets
from training.data import prepare_data

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
STGNN_DIR = ROOT / "outputs" / "stgnn"
OUT = ROOT / "outputs" / "demo"
GRAPHS = ["geographic", "gravity", "learned", "hybrid"]


def export_geometry():
    gdf = gpd.read_file(DATA / "nodes.geojson")[["censuscode", "DISTRICT", "ST_NM", "Population", "geometry"]]
    # ~1km tolerance: invisible at country zoom, cuts the file ~40x.
    gdf["geometry"] = gdf.geometry.simplify(0.01, preserve_topology=True)
    gdf.to_file(OUT / "districts.geojson", driver="GeoJSON", COORDINATE_PRECISION=4)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    export_geometry()

    data = prepare_data(lookback=8)
    codes = list(data["codes"])
    sample_times = np.asarray(data["sample_times"], dtype=int)
    test_start, test_end = data["split"]["test"]
    M = data["M"].numpy()

    S, I, new_conf, pop, weeks = build_states(codes)
    onset = detect_onsets(new_conf)
    test_k = np.arange(test_start, test_end)
    prev_i, curr_i = sample_times[test_k] - 1, sample_times[test_k]
    S_prev, I_prev, pop_b = S[prev_i], I[prev_i], pop[None, :]
    valid = (M[test_k] * np.isfinite(new_conf[curr_i]) * np.isfinite(new_conf[curr_i - 1])) > 0.5

    base = pd.DataFrame({
        "week": np.repeat(weeks[curr_i], len(codes)),
        "censuscode": np.tile(codes, len(test_k)),
        "actual": new_conf[curr_i].ravel(),
        "persistence": new_conf[curr_i - 1].ravel(),
        "onset": onset[curr_i].ravel(),
        "valid": valid.ravel(),
    })

    frames = []
    for graph in GRAPHS:
        path = STGNN_DIR / f"predictions_{graph}.npz"
        if not path.exists():
            print(f"  (skipping {graph}: {path.name} not found - train it first)")
            continue
        beta = np.load(path)["predictions"].astype(np.float64)
        cases = np.minimum(np.maximum(sir_forward(beta, S_prev, I_prev, pop_b), 0.0), S_prev)
        frame = base.copy()
        frame["graph"] = graph
        frame["predicted"] = cases.ravel()
        frame["beta"] = beta.ravel()
        frame["rt"] = (beta * S_prev / pop_b).ravel()
        frames.append(frame)
    if not frames:
        raise SystemExit("No predictions found in outputs/stgnn/ - run src/training/train.py first.")
    forecasts = pd.concat(frames)
    forecasts.to_csv(OUT / "forecasts.csv", index=False, float_format="%.4f")

    adj_path = STGNN_DIR / "adjacency_learned.npy"
    if adj_path.exists():
        adj = np.load(adj_path)
        np.fill_diagonal(adj, 0.0)
        top = np.argsort(adj, axis=1)[:, ::-1][:, :3]
        rows = [{"src": codes[i], "dst": codes[j], "weight": float(adj[i, j])}
                for i in range(len(codes)) for j in top[i]]
        pd.DataFrame(rows).to_csv(OUT / "learned_edges.csv", index=False)

    cases = pd.read_csv(DATA / "weekly_case_features.csv", usecols=["censuscode", "week", "new_confirmed"])
    cases.dropna(subset=["new_confirmed"]).to_csv(OUT / "history.csv", index=False)

    print(f"Saved demo bundle to {OUT} ({forecasts['graph'].nunique()} graphs, "
          f"{len(test_k)} test weeks, {len(codes)} districts)")


if __name__ == "__main__":
    main()
