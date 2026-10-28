"""Persistence baseline for the beta-proxy target. Person 3 deliverable 1."""

import json
from pathlib import Path
import numpy as np
from training.data import prepare_data

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "evaluation"
STGNN_DIR = ROOT / "outputs" / "stgnn"
GRAPHS = ["geographic", "gravity", "learned", "hybrid"]


def metrics(pred, target, mask):
    valid = mask > 0.5
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "MAE": None, "RMSE": None}
    err = pred[valid] - target[valid]
    return {"n": n,
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2)))}


def build_persistence(Y, M, test_start, test_end):
    """Predict beta[week t] = beta[week t-1] over the test weeks."""
    preds, targs, masks = [], [], []
    for k in range(test_start, test_end):
        preds.append(Y[k - 1])
        targs.append(Y[k])
        masks.append(M[k] * M[k - 1])
    return np.stack(preds), np.stack(targs), np.stack(masks)


def main():
    data = prepare_data(lookback=8)
    Y, M = data["Y"].numpy(), data["M"].numpy()
    test_start, test_end = data["split"]["test"]
    print(f"Test samples: {test_start}..{test_end-1} ({test_end-test_start} weeks)")

    p_pred, p_targ, shared_mask = build_persistence(Y, M, test_start, test_end)
    rows = {"persistence": metrics(p_pred, p_targ, shared_mask)}

    for graph in GRAPHS:
        path = STGNN_DIR / f"predictions_{graph}.npz"
        if not path.exists():
            print(f"  (skipping {graph}: not found)")
            continue
        saved = np.load(path)
        pred = saved["predictions"]
        if pred.shape != p_targ.shape:
            raise ValueError(f"{graph}: shape {pred.shape} != {p_targ.shape}")
        if not np.allclose(saved["targets"], p_targ, atol=1e-5):
            raise ValueError(f"{graph}: saved targets differ from recomputed.")
        rows[graph] = metrics(pred, p_targ, shared_mask)

    base = rows["persistence"]["MAE"]
    print()
    print(f"{'model':<14}{'n':>7}{'MAE':>10}{'RMSE':>10}{'MAE vs base':>14}{'beats?':>9}")
    print("-" * 64)
    for name, m in rows.items():
        if m["MAE"] is None:
            continue
        shown = "" if name == "persistence" else f"{(m['MAE']-base)/base*100:+.1f}%"
        beats = "" if name == "persistence" else ("YES" if m["MAE"] < base else "no")
        print(f"{name:<14}{m['n']:>7}{m['MAE']:>10.4f}{m['RMSE']:>10.4f}{shown:>14}{beats:>9}")
    print("-" * 64)
    print("Negative = better than persistence. Units are beta, not cases.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "baseline_comparison.json"
    out.write_text(json.dumps({"target": "beta proxy",
                               "test_window": data["target_note"],
                               "n_test_weeks": test_end - test_start,
                               "results": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
