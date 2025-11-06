"""Graph encoder and registry utilities for Coconut graph conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


GRAPH_ENCODER_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_graph_encoder(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Decorator to register a graph encoder constructor by name."""

    def decorator(cls_or_fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in GRAPH_ENCODER_REGISTRY:
            raise ValueError(f"Graph encoder '{name}' already registered")
        GRAPH_ENCODER_REGISTRY[name] = cls_or_fn
        return cls_or_fn

    return decorator


def build_graph_encoder(name: str, **kwargs) -> nn.Module:
    """Instantiate a registered graph encoder."""

    if name not in GRAPH_ENCODER_REGISTRY:
        known = ", ".join(sorted(GRAPH_ENCODER_REGISTRY.keys())) or "<empty>"
        raise ValueError(f"Unknown graph encoder '{name}'. Known encoders: {known}")
    return GRAPH_ENCODER_REGISTRY[name](**kwargs)


@register_graph_encoder("gcn2")
class GraphEncoder(nn.Module):
    """Two-layer GCN encoder that returns pooled and per-node representations."""

    def __init__(self, d_in: int = 259, d_hidden: int = 256, d_out: int = 256, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GCNConv(d_in, d_hidden, cached=False, add_self_loops=True)
        self.conv2 = GCNConv(d_hidden, d_out, cached=False, add_self_loops=True)
        self.ln1 = nn.LayerNorm(d_hidden)
        self.ln2 = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv1(x, edge_index)
        h = self.ln1(F.relu(h))
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = self.ln2(h)
        z_g = global_mean_pool(h, batch)
        return z_g, h


class GraphProjector(nn.Module):
    """Projects graph embeddings into the language model hidden space."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GraphPrefixAdapter(nn.Module):
    """Transforms projected graph embeddings into virtual token prefixes."""

    def __init__(self, hidden_size: int, prefix_len: int, dropout: float = 0.1):
        super().__init__()
        self.prefix_len = prefix_len
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * prefix_len),
        )

    def forward(self, conditioned: torch.Tensor) -> torch.Tensor:
        batch, hidden = conditioned.shape
        projected = self.mlp(conditioned)
        projected = self.dropout(projected)
        return projected.view(batch, self.prefix_len, hidden)


@dataclass
class GraphConditioningBundle:
    """Convenience bundle for all graph conditioning modules."""

    encoder: nn.Module
    projector: GraphProjector
    prefix_adapter: GraphPrefixAdapter
    residual_norm: nn.LayerNorm

    def to(self, device: torch.device | str) -> "GraphConditioningBundle":
        self.encoder.to(device)
        self.projector.to(device)
        self.prefix_adapter.to(device)
        self.residual_norm.to(device)
        return self

