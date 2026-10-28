
"""Prepare weekly COVID features and an SIR-inspired beta proxy.

Uses new_confirmed directly; never differences cumulative counts to create
weekly incidence. Missing observations remain masked.

The beta target is an experimental proxy, not ground-truth transmission rate.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"

# Chronological evaluation boundaries.
# The January 2022 Omicron period is held out for testing.
VAL_START = pd.Timestamp("2021-10-18")
TEST_START = pd.Timestamp("2021-12-27")
TEST_END = pd.Timestamp("2022-02-07")


def _load_adj(name: str) -> np.ndarray:
    return sp.load_npz(DATA / name).tocsr().toarray().astype(np.float32)


def _as_observed(series: pd.Series) -> pd.Series:
    """Normalize Boolean, numeric, or string observation flags."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def prepare_data(lookback: int = 8, beta_max: float = 2.0):
    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    nodes = pd.read_csv(DATA / "nodes.csv")
    order = pd.read_csv(DATA / "node_order.csv").sort_values("row")
    cases = pd.read_csv(
        DATA / "weekly_case_features.csv",
        parse_dates=["week"],
    )

    required = {
        "censuscode",
        "week",
        "Confirmed",
        "Recovered",
        "Deceased",
        "observed",
        "new_confirmed",
        "new_recovered",
        "new_deceased",
        "new_tested",
    }
    missing = required - set(cases.columns)
    if missing:
        raise ValueError(
            f"Updated case CSV is missing columns: {sorted(missing)}"
        )

    codes = order["censuscode"].tolist()
    node_info = nodes.set_index("censuscode").reindex(codes)

    if node_info["Population"].isna().any():
        raise ValueError(
            "Some node_order censuscode values are missing from nodes.csv"
        )

    cases = cases[cases["censuscode"].isin(codes)].copy()
    cases["week"] = pd.to_datetime(cases["week"], errors="coerce")
    cases = cases.dropna(subset=["week"])
    cases["observed"] = _as_observed(cases["observed"])

    if cases.duplicated(["censuscode", "week"]).any():
        raise ValueError(
            "Duplicate censuscode/week rows found. Resolve them before training."
        )

    weeks = pd.DatetimeIndex(sorted(cases["week"].unique()))
    T, N = len(weeks), len(codes)

    if T <= lookback:
        raise ValueError("Not enough weekly data for the requested lookback")

    new_cols = [
        "new_confirmed",
        "new_recovered",
        "new_deceased",
        "new_tested",
    ]

    new_values = np.full((T, N, 4), np.nan, dtype=np.float32)
    observed = np.zeros((T, N), dtype=bool)
    cumulative = np.full((T, N, 3), np.nan, dtype=np.float32)

    for j, code in enumerate(codes):
        sub = (
            cases[cases["censuscode"] == code]
            .set_index("week")
            .reindex(weeks)
        )

        observed[:, j] = (
            sub["observed"].fillna(False).to_numpy(dtype=bool)
        )

        for k, col in enumerate(new_cols):
            new_values[:, j, k] = pd.to_numeric(
                sub[col], errors="coerce"
            ).to_numpy(dtype=np.float32)

        for k, col in enumerate(
            ["Confirmed", "Recovered", "Deceased"]
        ):
            cumulative[:, j, k] = pd.to_numeric(
                sub[col], errors="coerce"
            ).to_numpy(dtype=np.float32)

    # Use supplied weekly increments. Preserve availability separately so
    # a missing increment is not mistaken for a real zero.
    available = np.isfinite(new_values) & observed[:, :, None]
    safe_new = np.where(available, new_values, 0.0)

    if np.any(safe_new < 0):
        raise ValueError(
            "Negative weekly increments found; investigate source data"
        )

    dynamic = np.log1p(safe_new).astype(np.float32)

    pop = node_info["Population"].to_numpy(dtype=np.float32)
    density = node_info["density_per_km2"].to_numpy(dtype=np.float32)

    static = np.stack(
        [
            np.log1p(np.maximum(pop, 0)),
            np.log1p(np.maximum(density, 0)),
        ],
        axis=-1,
    )
    static = np.broadcast_to(
        static[None, :, :], (T, N, 2)
    ).copy()

    # 4 logged increments + 4 availability indicators + 2 static features
    # + observation indicator = 11 features per node per time step.
    features = np.concatenate(
        [
            dynamic,
            available.astype(np.float32),
            static,
            observed[:, :, None].astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)

    # Forward-fill cumulative states only for the SIR-inspired proxy.
    # Target validity still requires observed current and previous weeks.
    cumulative_filled = np.empty_like(cumulative)

    for j in range(N):
        cumulative_filled[:, j, :] = (
            pd.DataFrame(cumulative[:, j, :])
            .ffill()
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )

    confirmed = cumulative_filled[:, :, 0]
    recovered = cumulative_filled[:, :, 1]
    deceased = cumulative_filled[:, :, 2]

    active = np.maximum(confirmed - recovered - deceased, 0.0)
    susceptible = np.maximum(pop[None, :] - confirmed, 1.0)

    beta_proxy = np.zeros((T, N), dtype=np.float32)
    target_mask = np.zeros((T, N), dtype=np.float32)

    # Target at week i uses new_confirmed[i] and S/I states at week i-1.
    for i in range(1, T):
        weekly_cases = new_values[i, :, 0]

        valid = (
            observed[i]
            & observed[i - 1]
            & np.isfinite(weekly_cases)
            & np.isfinite(confirmed[i - 1])
            & np.isfinite(recovered[i - 1])
            & np.isfinite(deceased[i - 1])
            & (active[i - 1] >= 5)
        )

        # Require a true one-week interval, not merely the previous
        # available row for a district.
        previous_week = weeks[i - 1]
        current_week = weeks[i]
        if (current_week - previous_week).days != 7:
            valid[:] = False

        denom = susceptible[i - 1] * np.maximum(active[i - 1], 1.0)
        raw_beta = (
            weekly_cases * pop / np.maximum(denom, 1.0)
        )
        valid &= np.isfinite(raw_beta)

        beta_proxy[i, valid] = np.clip(
            raw_beta[valid], 0.0, beta_max
        ).astype(np.float32)
        target_mask[i, valid] = 1.0

    # Construct each sample with a target week at sample_times[k].
    # Its input window ends at the week before the target.
    sample_x = []
    sample_y = []
    sample_m = []
    sample_times = []

    for target_i in range(lookback, T):
        target_date = weeks[target_i]

        if target_date > TEST_END:
            continue

        sample_x.append(features[target_i - lookback:target_i])
        sample_y.append(beta_proxy[target_i])
        sample_m.append(target_mask[target_i])
        sample_times.append(target_i)

    if not sample_x:
        raise ValueError(
            "No samples available for the configured dates and lookback"
        )

    X = np.stack(sample_x).astype(np.float32)
    Y = np.stack(sample_y).astype(np.float32)
    M = np.stack(sample_m).astype(np.float32)

    sample_times = np.asarray(sample_times, dtype=int)
    target_dates = weeks[sample_times]

    # Date-based partitions; don't split by percentages of the full dataset.
    train_idx = np.flatnonzero(target_dates < VAL_START)
    val_idx = np.flatnonzero(
        (target_dates >= VAL_START) & (target_dates < TEST_START)
    )
    test_idx = np.flatnonzero(
        (target_dates >= TEST_START) & (target_dates <= TEST_END)
    )

    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError(
            "An empty split was produced. Check the date boundaries."
        )

    # These date ranges must form contiguous chronological partitions.
    n_train = len(train_idx)
    n_val = len(val_idx)
    n_total = len(sample_x)

    if not np.array_equal(train_idx, np.arange(n_train)):
        raise ValueError("Training samples are not contiguous")
    if not np.array_equal(
        val_idx, np.arange(n_train, n_train + n_val)
    ):
        raise ValueError("Validation samples are not contiguous")
    if not np.array_equal(
        test_idx, np.arange(n_train + n_val, n_total)
    ):
        raise ValueError("Test samples are not contiguous")

    # Normalize input features using training samples only.
    train_flat = X[:n_train].reshape(-1, X.shape[-1])
    mean = train_flat.mean(axis=0)
    std = train_flat.std(axis=0)
    std[std < 1e-6] = 1.0

    X = np.clip((X - mean) / std, -8, 8).astype(np.float32)

    split = {
        "train": (0, n_train),
        "val": (n_train, n_train + n_val),
        "test": (n_train + n_val, n_total),
    }

    return {
        "X": torch.from_numpy(X),
        "Y": torch.from_numpy(Y),
        "M": torch.from_numpy(M),
        "split": split,
        "codes": codes,
        "weeks": [
            str(pd.Timestamp(w).date()) for w in weeks
        ],
        "sample_times": sample_times.tolist(),
        "feature_mean": mean,
        "feature_std": std,
        "geographic": _load_adj("geographic_adjacency.npz"),
        "gravity": _load_adj("gravity_adjacency.npz"),
        "target_note": (
            "Experimental SIR-inspired beta proxy using supplied "
            "new_confirmed increments; not ground-truth beta. "
            f"Test target dates: {TEST_START.date()} to {TEST_END.date()}."
        ),
    }
