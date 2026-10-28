
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedAdjacency(nn.Module):
    """Learn a trainable district-to-district adjacency matrix."""

    def __init__(self, num_nodes: int, embedding_dim: int = 16):
        super().__init__()

        self.node_emb1 = nn.Parameter(
            torch.randn(num_nodes, embedding_dim) * 0.1
        )
        self.node_emb2 = nn.Parameter(
            torch.randn(num_nodes, embedding_dim) * 0.1
        )

    def forward(self) -> torch.Tensor:
        scores = (
            self.node_emb1 @ self.node_emb2.T
        ) / math.sqrt(self.node_emb1.shape[-1])

        # Symmetric learned graph.
        scores = (scores + scores.T) / 2.0

        # Numerically stable row-wise softmax.
        scores = scores - scores.max(dim=1, keepdim=True).values

        return torch.softmax(scores, dim=1)


class GraphConvolution(nn.Module):
    """Propagate district features through an adjacency matrix."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        # x: [batch, time, nodes, features]
        # adjacency: [nodes, nodes]
        x = torch.einsum("ij,btjf->btif", adjacency, x)
        x = self.linear(x)
        x = F.relu(x)
        return self.dropout(x)


class SpatioTemporalGNN(nn.Module):
    """
    Spatio-temporal GNN using a GRU temporal encoder, graph convolution,
    and a learned adjacency matrix.

    Supported graph modes:
        geographic: supplied geographic graph
        gravity:    supplied gravity graph
        learned:    trainable learned graph
        hybrid:     average of supplied and learned graphs

    Input:
        x: [batch, sequence_length, num_nodes, input_features]

    Output:
        beta:      [batch, num_nodes]
        adjacency: [num_nodes, num_nodes]
    """

    VALID_GRAPH_MODES = {
        "geographic",
        "gravity",
        "learned",
        "hybrid",
    }

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int = 32,
        embedding_dim: int = 16,
        dropout: float = 0.1,
        beta_max: float = 2.0,
        graph_mode: str = "hybrid",
        **kwargs,
    ):
        super().__init__()

        # Allow common alternative names without breaking initialization.
        if "graph_embedding_dim" in kwargs:
            embedding_dim = kwargs.pop("graph_embedding_dim")

        if "beta_max" in kwargs:
            beta_max = kwargs.pop("beta_max")

        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected model arguments: {unexpected}")

        graph_mode = str(graph_mode).lower()

        if graph_mode not in self.VALID_GRAPH_MODES:
            raise ValueError(
                f"Invalid graph_mode '{graph_mode}'. "
                f"Choose from {sorted(self.VALID_GRAPH_MODES)}."
            )

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.beta_max = beta_max
        self.graph_mode = graph_mode

        # Temporal encoder.
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Trainable graph.
        self.learned_graph = LearnedAdjacency(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
        )

        # Spatial encoder.
        self.graph_conv = GraphConvolution(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Predict beta for each district.
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
        """Normalize a nonnegative adjacency matrix by row."""
        adjacency = torch.nan_to_num(
            adjacency, nan=0.0, posinf=0.0, neginf=0.0
        )
        adjacency = adjacency.clamp(min=0.0)

        # Include self-loops so isolated nodes retain their own features.
        adjacency = adjacency + torch.eye(
            adjacency.shape[0],
            device=adjacency.device,
            dtype=adjacency.dtype,
        )

        row_sum = adjacency.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return adjacency / row_sum

    def _select_adjacency(
        self,
        supplied_adjacency: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        learned = self.learned_graph().to(device=device, dtype=dtype)

        if self.graph_mode == "learned":
            return self._normalize_adjacency(learned)

        if supplied_adjacency is None:
            raise ValueError(
                f"graph_mode='{self.graph_mode}' requires an adjacency "
                "matrix from the training/data pipeline. Pass the "
                "geographic or gravity matrix to model.forward()."
            )

        supplied_adjacency = torch.as_tensor(
            supplied_adjacency,
            device=device,
            dtype=dtype,
        )

        expected_shape = (self.num_nodes, self.num_nodes)
        if tuple(supplied_adjacency.shape) != expected_shape:
            raise ValueError(
                f"Expected adjacency shape {expected_shape}, got "
                f"{tuple(supplied_adjacency.shape)}."
            )

        supplied_adjacency = self._normalize_adjacency(
            supplied_adjacency
        )

        if self.graph_mode in {"geographic", "gravity"}:
            return supplied_adjacency

        # Hybrid: combine the supplied graph with the learned graph.
        hybrid = 0.5 * supplied_adjacency + 0.5 * learned
        return self._normalize_adjacency(hybrid)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None,
    ):
        if x.ndim != 4:
            raise ValueError(
                "Expected input shape [batch, time, nodes, features]; "
                f"received {tuple(x.shape)}."
            )

        batch_size, sequence_length, num_nodes, input_dim = x.shape

        if num_nodes != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes, received {num_nodes}."
            )

        if input_dim != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input features, received "
                f"{input_dim}."
            )

        # Encode every district's historical sequence with the GRU.
        # [B, T, N, F] -> [B*N, T, F]
        district_series = (
            x.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(
                batch_size * num_nodes,
                sequence_length,
                input_dim,
            )
        )

        _, hidden = self.gru(district_series)

        # [B*N, H] -> [B, N, H]
        temporal = hidden[-1].reshape(
            batch_size, num_nodes, self.hidden_dim
        )

        selected_adjacency = self._select_adjacency(
            adjacency,
            device=x.device,
            dtype=temporal.dtype,
        )

        # Graph convolution over the district representations.
        spatial = self.graph_conv(
            temporal.unsqueeze(1),
            selected_adjacency,
        ).squeeze(1)

        # Combine temporal and spatial information.
        combined = torch.cat([temporal, spatial], dim=-1)

        # Positive, bounded beta proxy.
        beta = self.beta_max * torch.sigmoid(
            self.head(combined).squeeze(-1)
        )

        return beta, selected_adjacency
