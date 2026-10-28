
"""Train and evaluate the spatio-temporal GNN beta-proxy models."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from models.stgnn import SpatioTemporalGNN
from training.data import prepare_data


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "stgnn"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_loss(prediction, target, mask):
    """Smooth L1 loss over observed targets only."""
    valid = mask > 0.5
    if not valid.any():
        return None
    return F.smooth_l1_loss(prediction[valid], target[valid])


def evaluate(model, loader, adjacency, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    predictions = []
    targets = []
    masks = []

    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            pred, _ = model(x, adjacency)
            loss = masked_loss(pred, y, mask)

            if loss is not None:
                total_loss += loss.item()
                total_batches += 1

            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
            masks.append(mask.cpu().numpy())

    if not predictions:
        raise RuntimeError("Evaluation loader produced no batches.")

    result = {
        "loss": total_loss / max(total_batches, 1),
        "predictions": np.concatenate(predictions),
        "targets": np.concatenate(targets),
        "masks": np.concatenate(masks),
    }
    return result


def calculate_metrics(pred, target, mask):
    valid = mask > 0.5
    if not np.any(valid):
        return {"n": 0, "MAE": None, "RMSE": None}

    errors = pred[valid] - target[valid]
    return {
        "n": int(valid.sum()),
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph",
        choices=["geographic", "gravity", "learned", "hybrid"],
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lookback", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}", flush=True)
    print(f"Graph mode: {args.graph}", flush=True)

    data = prepare_data(args.lookback)
    X, Y, M = data["X"], data["Y"], data["M"]
    splits = data["split"]

    n_samples, sequence_length, num_nodes, input_dim = X.shape
    print(
        f"Data: samples={n_samples}, lookback={sequence_length}, "
        f"nodes={num_nodes}, features={input_dim}",
        flush=True,
    )
    print(f"Splits: {splits}", flush=True)

    def make_loader(split_name, shuffle=False):
        start, end = splits[split_name]
        dataset = TensorDataset(X[start:end], Y[start:end], M[start:end])
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,
        )

    train_loader = make_loader("train", shuffle=True)
    val_loader = make_loader("val")
    test_loader = make_loader("test")

    if args.graph == "gravity":
        adjacency = torch.as_tensor(data["gravity"], dtype=torch.float32)
    elif args.graph in ("geographic", "hybrid"):
        adjacency = torch.as_tensor(data["geographic"], dtype=torch.float32)
    else:
        adjacency = None

    if adjacency is not None:
        adjacency = adjacency.to(device)

    model = SpatioTemporalGNN(
        num_nodes=num_nodes,
        input_dim=input_dim,
        hidden_dim=args.hidden,
        graph_mode=args.graph,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = CHECKPOINT_DIR / f"best_{args.graph}.pt"
    history = []
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches_with_targets = 0

        for x, y, mask in train_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(x, adjacency)
            loss = masked_loss(prediction, y, mask)

            if loss is None:
                continue

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch {epoch}."
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            batches_with_targets += 1

        if batches_with_targets == 0:
            raise RuntimeError("No valid masked targets in training data.")

        train_loss = running_loss / batches_with_targets
        val_result = evaluate(model, val_loader, adjacency, device)
        val_loss = val_result["loss"]

        if not np.isfinite(val_loss):
            raise RuntimeError("Validation loss is not finite.")

        scheduler.step(val_loss)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"lr={record['lr']:.2e}",
            flush=True,
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "graph": args.graph,
                    "input_dim": input_dim,
                    "num_nodes": num_nodes,
                    "lookback": args.lookback,
                    "hidden": args.hidden,
                    "seed": args.seed,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print("Early stopping.", flush=True)
            break

    # Restore the checkpoint selected by validation, then evaluate test once.
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_result = evaluate(model, test_loader, adjacency, device)
    metrics = calculate_metrics(
        test_result["predictions"],
        test_result["targets"],
        test_result["masks"],
    )

    predictions_path = OUTPUT_DIR / f"predictions_{args.graph}.npz"
    np.savez_compressed(
        predictions_path,
        predictions=test_result["predictions"],
        targets=test_result["targets"],
        mask=test_result["masks"],
    )

    with torch.no_grad():
        _, selected_adjacency = model(
            X[splits["test"][0]:splits["test"][0] + 1].to(device),
            adjacency,
        )
        np.save(
            OUTPUT_DIR / f"adjacency_{args.graph}.npy",
            selected_adjacency.cpu().numpy(),
        )

    report = {
        "graph": args.graph,
        "device": str(device),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "test_loss": test_result["loss"],
        "test_metrics_against_beta_proxy": metrics,
        "target_note": data["target_note"],
        "split_indices": {
            name: list(bounds) for name, bounds in splits.items()
        },
        "history": history,
    }

    report_path = OUTPUT_DIR / f"metrics_{args.graph}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nFinished training.", flush=True)
    print(f"Best epoch: {best_epoch}", flush=True)
    print(f"Best validation loss: {best_val:.6f}", flush=True)
    print(f"Test metrics: {metrics}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Report: {report_path}", flush=True)
    print(f"Predictions: {predictions_path}", flush=True)


if __name__ == "__main__":
    main()
