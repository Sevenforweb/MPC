from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .decoder import SpatialRecouplingDecoder
from .serialization import DensityHilbertSerializer
from .ssm import SpatialModulatedSSMBlock


@dataclass
class MPCConfig:
    num_joints: int = 15
    grid_size: int = 16
    coordinate_range: Sequence[Sequence[float]] = ((-1.0, 1.0),) * 3
    d_model: int = 96
    d_state: int = 8
    depth: int = 4
    kde_sigma: float = 1.25
    dropout: float = 0.0


@dataclass
class MPCPoseOutput:
    coordinates: Tensor
    heatmap_logits: Tensor
    heatmap_probabilities: Tensor
    uncertainty: Tensor
    correction: Tensor
    physical_context: Tensor
    point_foreground_logits: Tensor
    point_mask: Tensor
    token_mask: Tensor


class ConditionalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, grid_size: int) -> None:
        super().__init__()
        sequence_length = 3 * grid_size ** 2
        self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length, d_model))
        self.projection_embedding = nn.Embedding(3, d_model)
        self.depthwise_convolution = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, inputs: Tensor, projection_ids: Tensor) -> Tensor:
        positional = self.position_embedding[:, : inputs.shape[1]]
        positional = positional + self.projection_embedding(projection_ids).unsqueeze(0)
        return inputs + positional + self.depthwise_convolution(inputs.transpose(1, 2)).transpose(1, 2)


class MPCPoseModel(nn.Module):
    def __init__(self, config: Optional[MPCConfig] = None) -> None:
        super().__init__()
        self.config = config or MPCConfig()
        config = self.config
        self.point_encoder = nn.Sequential(
            nn.Linear(3, config.d_model // 2),
            nn.GELU(),
            nn.Linear(config.d_model // 2, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.point_foreground_head = nn.Linear(config.d_model, 1)
        self.serializer = DensityHilbertSerializer(config.grid_size, config.coordinate_range, config.kde_sigma)
        self.token_projection = nn.Linear(config.d_model + 3, config.d_model)
        self.position_encoding = ConditionalPositionEncoding(config.d_model, config.grid_size)
        self.blocks = nn.ModuleList(
            SpatialModulatedSSMBlock(config.d_model, config.d_state, dropout=config.dropout)
            for _ in range(config.depth)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.decoder = SpatialRecouplingDecoder(config.d_model, config.num_joints)
        self.heatmap_head = nn.Linear(config.d_model, 3 * config.grid_size)
        self.correction_head = nn.Linear(config.d_model, 3)
        coordinate_range = torch.as_tensor(config.coordinate_range, dtype=torch.float32)
        bins = torch.stack(
            [torch.linspace(axis_range[0], axis_range[1], config.grid_size) for axis_range in coordinate_range],
            dim=0,
        )
        self.register_buffer("coordinate_bins", bins, persistent=True)
        self.register_buffer("coordinate_range", coordinate_range, persistent=True)

    def forward(self, points: Tensor, point_mask: Optional[Tensor] = None) -> MPCPoseOutput:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points must have shape (batch, points, 3)")
        if point_mask is None:
            point_mask = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        point_features = self.point_encoder(points)
        point_features = point_features * point_mask.unsqueeze(-1).to(point_features.dtype)
        point_foreground_logits = self.point_foreground_head(point_features).squeeze(-1)
        serialized = self.serializer(points, point_features, point_mask)
        tokens = self.token_projection(torch.cat((serialized.point_features, serialized.density_features), dim=-1))
        tokens = self.position_encoding(tokens, serialized.projection_ids)
        for block in self.blocks:
            tokens = block(tokens, serialized.token_mask)
        tokens = self.final_norm(tokens)
        recoupled = self.decoder(tokens, serialized.physical_anchors, serialized.token_mask)
        heatmap_logits = self.heatmap_head(recoupled.joint_features)
        heatmap_logits = heatmap_logits.view(points.shape[0], self.config.num_joints, 3, self.config.grid_size)
        probabilities = torch.softmax(heatmap_logits, dim=-1)
        bins = self.coordinate_bins.to(device=points.device, dtype=points.dtype)
        base_coordinates = (probabilities * bins.view(1, 1, 3, -1)).sum(dim=-1)
        uncertainty = (probabilities * (bins.view(1, 1, 3, -1) - base_coordinates.unsqueeze(-1)).square()).sum(dim=-1)
        coordinate_span = (self.coordinate_range[:, 1] - self.coordinate_range[:, 0]).to(points)
        maximum_variance = coordinate_span.square() / 4.0
        uncertainty_gate = (uncertainty / maximum_variance.view(1, 1, 3).clamp_min(1e-6)).clamp(0.0, 1.0)
        voxel_size = coordinate_span / max(1, self.config.grid_size - 1)
        correction = torch.tanh(self.correction_head(recoupled.joint_features)) * voxel_size.view(1, 1, 3) * 2.0
        correction = correction * (0.25 + uncertainty_gate)
        coordinates = base_coordinates + correction
        lower = self.coordinate_range[:, 0].to(points).view(1, 1, 3)
        upper = self.coordinate_range[:, 1].to(points).view(1, 1, 3)
        coordinates = torch.maximum(torch.minimum(coordinates, upper), lower)
        return MPCPoseOutput(
            coordinates=coordinates,
            heatmap_logits=heatmap_logits,
            heatmap_probabilities=probabilities,
            uncertainty=uncertainty,
            correction=correction,
            physical_context=recoupled.physical_context,
            point_foreground_logits=point_foreground_logits,
            point_mask=point_mask,
            token_mask=serialized.token_mask,
        )

    @torch.no_grad()
    def decode_argmax(self, heatmap_logits: Tensor) -> Tensor:
        indices = heatmap_logits.argmax(dim=-1)
        bins = self.coordinate_bins.to(device=heatmap_logits.device, dtype=heatmap_logits.dtype)
        expanded_bins = bins.view(1, 1, 3, -1).expand(*indices.shape, -1)
        return torch.gather(expanded_bins, dim=-1, index=indices.unsqueeze(-1)).squeeze(-1)
