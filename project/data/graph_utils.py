"""Graph construction utilities for static and wind-driven dynamic adjacency."""

from __future__ import annotations

import torch


def haversine_distance_matrix(coords: torch.Tensor) -> torch.Tensor:
    """Compute pairwise great-circle distance matrix in kilometers."""

    lon = torch.deg2rad(coords[:, 0]).unsqueeze(1)
    lat = torch.deg2rad(coords[:, 1]).unsqueeze(1)
    dlon = lon.transpose(0, 1) - lon
    dlat = lat.transpose(0, 1) - lat
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat) * torch.cos(lat.transpose(0, 1)) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a.clamp_min(0.0)), torch.sqrt((1 - a).clamp_min(1e-12)))
    return 6371.0 * c


def build_bearing_matrix(coords: torch.Tensor) -> torch.Tensor:
    """Compute bearing angles phi_ij from node i to node j in radians."""

    lon = torch.deg2rad(coords[:, 0])
    lat = torch.deg2rad(coords[:, 1])
    lon_i = lon.unsqueeze(1)
    lon_j = lon.unsqueeze(0)
    lat_i = lat.unsqueeze(1)
    lat_j = lat.unsqueeze(0)
    dlon = lon_j - lon_i
    y = torch.sin(dlon) * torch.cos(lat_j)
    x = torch.cos(lat_i) * torch.sin(lat_j) - torch.sin(lat_i) * torch.cos(lat_j) * torch.cos(dlon)
    return torch.atan2(y, x)


def build_static_adj(coords: torch.Tensor, sigma: float, add_self_loop: bool = True, top_k: int | None = None) -> torch.Tensor:
    """Build static adjacency A_ij = exp(-d_ij^2 / sigma^2), shape [N, N]."""

    if coords.ndim != 2 or coords.size(-1) != 2:
        raise ValueError(f"coords must be [N, 2], got {tuple(coords.shape)}")
    dist = haversine_distance_matrix(coords)
    sigma_sq = max(sigma, 1e-6) ** 2
    adj = torch.exp(-(dist ** 2) / sigma_sq)
    if top_k is not None and top_k < adj.size(0):
        values, indices = torch.topk(adj, k=top_k, dim=-1)
        sparse = torch.zeros_like(adj)
        sparse.scatter_(dim=-1, index=indices, src=values)
        adj = sparse
    if add_self_loop:
        adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    return adj


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric adjacency normalization for [N, N] or [B, T, N, N]."""

    if adj.ndim == 2:
        degree = adj.sum(dim=-1)
        inv_sqrt = torch.pow(degree.clamp_min(1e-12), -0.5)
        d_mat = torch.diag(inv_sqrt)
        return d_mat @ adj @ d_mat
    if adj.ndim == 4:
        degree = adj.sum(dim=-1)
        inv_sqrt = torch.pow(degree.clamp_min(1e-12), -0.5)
        return adj * inv_sqrt.unsqueeze(-1) * inv_sqrt.unsqueeze(-2)
    raise ValueError(f"adj must be [N, N] or [B, T, N, N], got {tuple(adj.shape)}")


def build_dynamic_adj(
    static_adj: torch.Tensor,
    coords: torch.Tensor,
    wind_speed_seq: torch.Tensor,
    wind_dir_seq: torch.Tensor,
    add_self_loop: bool = True,
) -> torch.Tensor:
    """Construct wind-driven dynamic adjacency, output [B, T, N, N]."""

    assert static_adj.ndim == 2
    assert wind_speed_seq.shape == wind_dir_seq.shape
    phi = build_bearing_matrix(coords).to(wind_speed_seq.device)
    theta = torch.deg2rad(wind_dir_seq).unsqueeze(-1)
    dir_weight = torch.cos(theta - phi.view(1, 1, *phi.shape))
    dir_weight = torch.clamp(dir_weight, min=0.0)
    spd_weight = wind_speed_seq.unsqueeze(-1).clamp_min(0.0)
    dynamic_adj = static_adj.view(1, 1, *static_adj.shape) * dir_weight * spd_weight
    if add_self_loop:
        eye = torch.eye(static_adj.size(0), device=dynamic_adj.device, dtype=dynamic_adj.dtype)
        dynamic_adj = dynamic_adj + eye.view(1, 1, *eye.shape)
    return dynamic_adj
