"""Rolling-origin (real-time) evaluation. Person 3 deliverable 6.

train.py uses ONE fixed chronological split - the protocol every paper in our
literature review uses, and the one SpatialEpiBench (2026) argues makes models
look better than they are. Here, for each origin week o:
    train on every sample before o (minus a short validation tail)
    predict week o, then move forward and retrain from scratch

Known limitation, stated rather than hidden: prepare_data() normalises features
using its own fixed train split, so early origins carry a little information
from later weeks. Minor, and identical across every configuration compared.
"""

import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from evaluation.sir import build_states, sir_forward
from evaluation.turning_points import detect_onsets
from models.stgnn import SpatioTemporalGNN
from training.data import prepare_data

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "evaluation"
VAL_TAIL = 6


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_loss(pred, target, mask, weight):
    valid = mask > 0.5
    if not valid.any():
        return None
    elementwise = F.smooth_l1_loss(pred, target, reduction="none")
    w = weight * mask
    return (elementwise * w).sum() / w.sum().clamp(min=1e-8)


def train_one_origin(data, W, origin, graph, device, seed, epochs, patience):
    X, Y, M = data["X"], data["Y"], data["M"]
    set_seed(seed)
    train_end = origin - VAL_TAIL
    if train_end < 10:
        raise ValueError(f"Origin {origin} leaves too little training data.")

    def loader(a, b, shuffle=False):
        return DataLoader(TensorDataset(X[a:b], Y[a:b], M[a:b], W[a:b]),
                          batch_size=2, shuffle=shuffle, num_workers=0)

    train_loader, val_loader = loader(0, train_end, True), loader(train_end, origin)

    if graph == "gravity":
        adjacency = torch.as_tensor(data["gravity"], dtype=torch.float32)
    elif graph in ("geographic", "hybrid"):
        adjacency = torch.as_tensor(data["geographic"], dtype=torch.float32)
    else:
        adjacency = None
    if adjacency is not None:
        adjacency = adjacency.to(device)

    _, _, num_nodes, input_dim = X.shape
    model = SpatioTemporalGNN(num_nodes=num_nodes, input_dim=input_dim,
                              hidden_dim=32, graph_mode=graph).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    best_val, best_state, stale = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for x, y, m, w in train_loader:
            x, y, m, w = (t.to(device) for t in (x, y, m, w))
            opt.zero_grad(set_to_none=True)
            pred, _ = model(x, adjacency)
            loss = weighted_loss(pred, y, m, w)
            if loss is None:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval(); tot, nb = 0.0, 0
        with torch.no_grad():
            for x, y, m, w in val_loader:
                x, y, m, w = (t.to(device) for t in (x, y, m, w))
                pred, _ = model(x, adjacency)
                l = weighted_loss(pred, y, m, w)
                if l is not None:
                    tot += l.item(); nb += 1
        val = tot / max(nb, 1)
        if val < best_val:
            best_val, stale = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred, _ = model(X[origin:origin + 1].to(device), adjacency)
    return pred.cpu().numpy()[0]


def mae(pred, target, mask):
    valid = mask > 0.5
    return None if not valid.any() else float(np.mean(np.abs(pred[valid] - target[valid])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default="hybrid",
                   choices=["geographic", "gravity", "learned", "hybrid"])
    p.add_argument("--n-origins", type=int, default=12)
    p.add_argument("--weights", type=float, nargs="+", default=[1.0, 5.0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = prepare_data(lookback=8)
    codes = data["codes"]
    sample_times = np.asarray(data["sample_times"], dtype=int)
    n_samples = data["X"].shape[0]

    S, I, new_conf, pop, weeks = build_states(codes)
    onset_all = detect_onsets(new_conf)
    origins = list(range(n_samples - args.n_origins, n_samples))
    origin_weeks = [str(weeks[sample_times[o]].date()) for o in origins]
    print(f"Graph: {args.graph} | device: {device}")
    print(f"{len(origins)} rolling origins: {origin_weeks[0]} .. {origin_weeks[-1]}")
    print("Each origin retrains from scratch on data strictly before it.\n")

    M = data["M"].numpy()
    prev_i, curr_i = sample_times[origins] - 1, sample_times[origins]
    S_prev, I_prev, pop_b = S[prev_i], I[prev_i], pop[None, :]
    actual = np.nan_to_num(new_conf[curr_i])
    prev_cases = np.nan_to_num(new_conf[curr_i - 1])
    base_mask = M[origins] * np.isfinite(new_conf[curr_i]) * np.isfinite(new_conf[curr_i - 1])
    onset_mask = base_mask * onset_all[curr_i]
    flat_mask = base_mask * (~onset_all[curr_i])

    rows = {"persistence": {"all": mae(prev_cases, actual, base_mask),
                            "onset": mae(prev_cases, actual, onset_mask),
                            "flat": mae(prev_cases, actual, flat_mask)}}

    for weight in args.weights:
        W = torch.from_numpy(np.where(onset_all[sample_times], weight, 1.0).astype(np.float32))
        betas = []
        for n, o in enumerate(origins, 1):
            betas.append(train_one_origin(data, W, o, args.graph, device,
                                          args.seed, args.epochs, args.patience))
            print(f"  w={weight:g}  origin {n}/{len(origins)} ({origin_weeks[n-1]}) done", flush=True)
        betas = np.stack(betas).astype(np.float64)
        cases = np.minimum(np.maximum(sir_forward(betas, S_prev, I_prev, pop_b), 0.0), S_prev)
        rows[f"{args.graph}_w{weight:g}"] = {"all": mae(cases, actual, base_mask),
                                             "onset": mae(cases, actual, onset_mask),
                                             "flat": mae(cases, actual, flat_mask)}

    b_on = rows["persistence"]["onset"]
    print()
    print(f"{'model':<22}{'MAE all':>11}{'MAE onset':>12}{'MAE flat':>11}{'onset vs base':>16}")
    print("-" * 72)
    for name, r in rows.items():
        delta = "" if name == "persistence" else f"{(r['onset']-b_on)/b_on*100:+.1f}%"
        print(f"{name:<22}{r['all']:>11.1f}{r['onset']:>12.1f}{r['flat']:>11.1f}{delta:>16}")
    print("-" * 72)
    print(f"Averaged over {len(origins)} independent retrain-and-predict deployments.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"rolling_origin_{args.graph}.json"
    out.write_text(json.dumps({"graph": args.graph, "n_origins": len(origins),
        "origin_weeks": origin_weeks, "val_tail_samples": VAL_TAIL, "seed": args.seed,
        "units": "weekly new confirmed cases per district", "results": rows},
        indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
