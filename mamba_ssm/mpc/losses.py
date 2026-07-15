from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .model import MPCPoseOutput


def gaussian_axis_heatmaps(joints: Tensor, coordinate_bins: Tensor, sigma: float = 0.04) -> Tensor:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    bins = coordinate_bins.to(device=joints.device, dtype=joints.dtype).view(1, 1, 3, -1)
    heatmaps = torch.exp(-0.5 * ((bins - joints.unsqueeze(-1)) / sigma).square())
    return heatmaps / heatmaps.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class MPCCalibrationLoss(nn.Module):
    def __init__(
        self,
        coordinate_bins: Tensor,
        skeleton_edges: Sequence[Tuple[int, int]] = (),
        heatmap_sigma: float = 0.04,
        heatmap_weight: float = 1.0,
        coordinate_weight: float = 1.0,
        connectivity_weight: float = 0.2,
        foreground_weight: float = 0.2,
        correction_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.register_buffer("coordinate_bins", coordinate_bins.detach().clone(), persistent=True)
        self.skeleton_edges = tuple((int(start), int(end)) for start, end in skeleton_edges)
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_weight = heatmap_weight
        self.coordinate_weight = coordinate_weight
        self.connectivity_weight = connectivity_weight
        self.foreground_weight = foreground_weight
        self.correction_weight = correction_weight

    def _connectivity_loss(self, predicted: Tensor, target: Tensor) -> Tensor:
        if not self.skeleton_edges:
            return predicted.new_zeros(())
        predicted_bones = torch.stack([predicted[:, end] - predicted[:, start] for start, end in self.skeleton_edges], dim=1)
        target_bones = torch.stack([target[:, end] - target[:, start] for start, end in self.skeleton_edges], dim=1)
        length_loss = F.smooth_l1_loss(predicted_bones.norm(dim=-1), target_bones.norm(dim=-1))
        direction_loss = 1.0 - F.cosine_similarity(predicted_bones, target_bones, dim=-1).mean()
        return length_loss + direction_loss

    def forward(
        self,
        output: MPCPoseOutput,
        target_joints: Tensor,
        point_foreground: Optional[Tensor] = None,
        stage: str = "finetune",
    ) -> Dict[str, Tensor]:
        if stage not in {"pretrain", "finetune", "joint"}:
            raise ValueError("stage must be 'pretrain', 'finetune', or 'joint'")
        target_heatmaps = gaussian_axis_heatmaps(target_joints, self.coordinate_bins, self.heatmap_sigma)
        heatmap_kl = F.kl_div(
            F.log_softmax(output.heatmap_logits, dim=-1),
            target_heatmaps,
            reduction="batchmean",
        )
        coordinate = F.smooth_l1_loss(output.coordinates, target_joints)
        connectivity = self._connectivity_loss(output.coordinates, target_joints)
        foreground = output.coordinates.new_zeros(())
        if point_foreground is not None:
            foreground = F.binary_cross_entropy_with_logits(
                output.point_foreground_logits,
                point_foreground.to(output.point_foreground_logits.dtype),
            )
        correction = output.correction.square().mean()
        pretrain_total = self.connectivity_weight * connectivity + self.foreground_weight * foreground
        finetune_total = self.heatmap_weight * heatmap_kl + self.coordinate_weight * coordinate + self.correction_weight * correction
        if stage == "pretrain":
            total = pretrain_total
        elif stage == "finetune":
            total = finetune_total
        else:
            total = pretrain_total + finetune_total
        return {
            "loss": total,
            "heatmap_kl": heatmap_kl,
            "coordinate": coordinate,
            "connectivity": connectivity,
            "foreground": foreground,
            "correction": correction,
        }
