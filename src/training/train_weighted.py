"""Train the STGNN with a turning-point-weighted loss. Person 3 deliverable 4.

A controlled A/B of exactly one change: wave-onset district-weeks get
`--weight` times more weight in the loss. Everything else is identical to
train.py, and `--weight 1.0` reproduces it.
"""

import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from evaluation.sir import build_states
from evaluation.turning_points import detect_onsets
from models.stgnn import SpatioTemporalGNN
from training.data import prepare_data

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "stgnn"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_loss(prediction, target, mask, weight):
    valid = mask > 0.5
    if not valid.any():
        return None
    elementwise = F.smooth_l1_loss(prediction, target, reduction="none")
    w = weight * mask
    return (elementwise * w).sum() / w.sum().clamp(min=1e-8)


def build_onset_weights(data, onset_weight):
    codes = data["codes"]
    sample_times = np.asarray(data["sample_times"], dtype=int)
    _, _, new_conf, _, _ = build_states(codes)
    is_onset = detect_onsets(new_conf)[sample_times]
    w = np.where(is_onset, onset_weight, 1.0).astype(np.float32)
    return torch.from_numpy(w), float(is_onset.mean() * 100)


def evaluate(model, loader, adjacency, device, weight_value):
    model.eval()
    total, batches = 0.0, 0
    preds, targs, masks = [], [], []
    with torch.no_grad():
        for x, y, mask, w in loader:
            x, y, mask, w = (t.to(device) for t in (x, y, mask, w))
            pred, _ = model(x, adjacency)
            loss = weighted_loss(pred, y, mask, w)
            if loss is not None:
                total += loss.item(); batches += 1
            preds.append(pred.cpu().numpy()); targs.append(y.cpu().numpy())
            masks.append(mask.cpu().numpy())
    return {"loss": total / max(batches, 1),
            "predictions": np.concatenate(preds),
            "targets": np.concatenate(targs),
            "masks": np.concatenate(masks)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", choices=["geographic","gravity","learned","hybrid"], required=True)
    p.add_argument("--weight", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lookback", type=int, default=8)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--patience", type=int, default=12)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = prepare_data(args.lookback)
    X, Y, M, splits = data["X"], data["Y"], data["M"], data["split"]
    W, onset_pct = build_onset_weights(data, args.weight)

    n_samples, seq_len, num_nodes, input_dim = X.shape
    tag = f"{args.graph}_w{args.weight:g}_s{args.seed}"
    print(f"Device: {device} | graph={args.graph} | weight={args.weight} | seed={args.seed}")
    print(f"Wave-onset share of district-weeks: {onset_pct:.1f}%")

    def make_loader(name, shuffle=False):
        a, b = splits[name]
        return DataLoader(TensorDataset(X[a:b], Y[a:b], M[a:b], W[a:b]),
                          batch_size=args.batch_size, shuffle=shuffle, num_workers=0)

    train_loader, val_loader, test_loader = (
        make_loader("train", True), make_loader("val"), make_loader("test"))

    if args.graph == "gravity":
        adjacency = torch.as_tensor(data["gravity"], dtype=torch.float32)
    elif args.graph in ("geographic", "hybrid"):
        adjacency = torch.as_tensor(data["geographic"], dtype=torch.float32)
    else:
        adjacency = None
    if adjacency is not None:
        adjacency = adjacency.to(device)

    model = SpatioTemporalGNN(num_nodes=num_nodes, input_dim=input_dim,
                              hidden_dim=args.hidden, graph_mode=args.graph).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                           factor=0.5, patience=4)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CHECKPOINT_DIR / f"best_{tag}.pt"
    best_val, best_epoch, stale = float("inf"), 0, 0

    for epoch in range(1, args.epochs + 1):
        model.train(); running, nb = 0.0, 0
        for x, y, mask, w in train_loader:
            x, y, mask, w = (t.to(device) for t in (x, y, mask, w))
            optimizer.zero_grad(set_to_none=True)
            pred, _ = model(x, adjacency)
            loss = weighted_loss(pred, y, mask, w)
            if loss is None:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step(); running += loss.item(); nb += 1

        val = evaluate(model, val_loader, adjacency, device, args.weight)
        scheduler.step(val["loss"])
        if epoch % 20 == 0 or epoch == 1:
            print(f"  epoch {epoch:03d} | train={running/max(nb,1):.6f} | val={val['loss']:.6f}", flush=True)
        if val["loss"] < best_val:
            best_val, best_epoch, stale = val["loss"], epoch, 0
            torch.save({"model_state_dict": model.state_dict()}, ckpt)
        else:
            stale += 1
        if stale >= args.patience:
            break

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)["model_state_dict"])
    test = evaluate(model, test_loader, adjacency, device, args.weight)
    out = OUTPUT_DIR / f"predictions_{tag}.npz"
    np.savez_compressed(out, predictions=test["predictions"],
                        targets=test["targets"], mask=test["masks"])
    (OUTPUT_DIR / f"metrics_{tag}.json").write_text(json.dumps(
        {"graph": args.graph, "onset_weight": args.weight, "seed": args.seed,
         "best_epoch": best_epoch, "best_validation_loss": best_val}, indent=2), encoding="utf-8")
    print(f"  best epoch {best_epoch} | val {best_val:.6f} -> {out.name}")


if __name__ == "__main__":
    main()
