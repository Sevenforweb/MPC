import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LinearSelfAttention(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        query = F.elu(self.query(inputs)) + 1.0
        key = F.elu(self.key(inputs)) + 1.0
        value = self.value(inputs)
        if mask is not None:
            valid = mask.unsqueeze(-1).to(inputs.dtype)
            key = key * valid
            value = value * valid
        key_value = torch.einsum("bld,ble->bde", key, value)
        normalizer = torch.einsum("bld,bd->bl", query, key.sum(dim=1)).clamp_min(1e-6)
        output = torch.einsum("bld,bde->ble", query, key_value)
        return self.output(output / normalizer.unsqueeze(-1))


@dataclass
class RecoupledFeatures:
    joint_features: Tensor
    physical_context: Tensor
    channel_attention: Tensor
    spatial_attention: Tensor


class SpatialRecouplingDecoder(nn.Module):
    def __init__(self, d_model: int, num_joints: int) -> None:
        super().__init__()
        self.coordinate_embedding = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.feature_fusion = nn.Linear(d_model * 2, d_model)
        self.self_attention = LinearSelfAttention(d_model)
        self.self_attention_norm = nn.LayerNorm(d_model)
        self.joint_queries = nn.Parameter(torch.empty(num_joints, d_model))
        self.channel_query = nn.Linear(d_model, d_model)
        self.channel_key = nn.Linear(d_model, d_model)
        self.channel_value = nn.Linear(d_model, d_model)
        self.spatial_query = nn.Linear(d_model, d_model)
        self.spatial_key = nn.Linear(d_model, d_model)
        self.spatial_value = nn.Linear(3, d_model)
        self.output = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        nn.init.normal_(self.joint_queries, std=0.02)

    def forward(self, features: Tensor, physical_anchors: Tensor, mask: Optional[Tensor] = None) -> RecoupledFeatures:
        coordinate_features = self.coordinate_embedding(physical_anchors)
        fused = self.feature_fusion(torch.cat((features, coordinate_features), dim=-1))
        fused = fused + self.self_attention(self.self_attention_norm(fused), mask)
        batch_size = features.shape[0]
        queries = self.joint_queries.unsqueeze(0).expand(batch_size, -1, -1)
        channel_query = F.normalize(self.channel_query(queries), dim=-1)
        channel_key = F.normalize(self.channel_key(fused), dim=-1)
        channel_logits = torch.einsum("bjd,bld->bjl", channel_query, channel_key) / math.sqrt(features.shape[-1])
        spatial_query = F.normalize(self.spatial_query(queries), dim=-1)
        spatial_key = F.normalize(self.spatial_key(coordinate_features), dim=-1)
        spatial_logits = torch.einsum("bjd,bld->bjl", spatial_query, spatial_key) / math.sqrt(features.shape[-1])
        if mask is not None:
            invalid = ~mask.unsqueeze(1)
            channel_logits = channel_logits.masked_fill(invalid, torch.finfo(channel_logits.dtype).min)
            spatial_logits = spatial_logits.masked_fill(invalid, torch.finfo(spatial_logits.dtype).min)
        channel_attention = torch.softmax(channel_logits, dim=-1)
        spatial_attention = torch.softmax(spatial_logits, dim=-1)
        if mask is not None:
            any_valid = mask.any(dim=1, keepdim=True).unsqueeze(1)
            channel_attention = torch.where(any_valid, channel_attention, torch.zeros_like(channel_attention))
            spatial_attention = torch.where(any_valid, spatial_attention, torch.zeros_like(spatial_attention))
        channel_context = torch.einsum("bjl,bld->bjd", channel_attention, self.channel_value(fused))
        spatial_context = torch.einsum("bjl,bld->bjd", spatial_attention, self.spatial_value(physical_anchors))
        physical_context = torch.einsum("bjl,blc->bjc", spatial_attention, physical_anchors)
        joint_features = self.output(torch.cat((channel_context, spatial_context), dim=-1))
        return RecoupledFeatures(joint_features, physical_context, channel_attention, spatial_attention)
