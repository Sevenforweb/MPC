from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@lru_cache(maxsize=16)
def hilbert_indices(size: int) -> Tensor:
    if size < 2 or size & (size - 1):
        raise ValueError("Hilbert grid size must be a power of two and at least 2")

    def rotate(scale: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
        if ry == 0:
            if rx == 1:
                x = scale - 1 - x
                y = scale - 1 - y
            x, y = y, x
        return x, y

    coordinates = []
    for distance in range(size * size):
        x = 0
        y = 0
        value = distance
        scale = 1
        while scale < size:
            rx = 1 & (value // 2)
            ry = 1 & (value ^ rx)
            x, y = rotate(scale, x, y, rx, ry)
            x += scale * rx
            y += scale * ry
            value //= 4
            scale *= 2
        coordinates.append(y * size + x)
    return torch.tensor(coordinates, dtype=torch.long)


@dataclass
class SerializedPointCloud:
    point_features: Tensor
    density_features: Tensor
    physical_anchors: Tensor
    token_mask: Tensor
    projection_ids: Tensor


class DensityHilbertSerializer(nn.Module):
    def __init__(
        self,
        grid_size: int = 16,
        coordinate_range: Sequence[Sequence[float]] = ((-1.0, 1.0),) * 3,
        kde_sigma: float = 1.25,
    ) -> None:
        super().__init__()
        if grid_size < 2 or grid_size & (grid_size - 1):
            raise ValueError("grid_size must be a power of two and at least 2")
        coordinate_range_tensor = torch.as_tensor(coordinate_range, dtype=torch.float32)
        if coordinate_range_tensor.shape != (3, 2):
            raise ValueError("coordinate_range must have shape (3, 2)")
        if torch.any(coordinate_range_tensor[:, 1] <= coordinate_range_tensor[:, 0]):
            raise ValueError("Each coordinate range must have max > min")
        self.grid_size = grid_size
        self.kde_sigma = float(kde_sigma)
        self.register_buffer("coordinate_range", coordinate_range_tensor, persistent=True)
        self.register_buffer("hilbert_order", hilbert_indices(grid_size), persistent=False)

    def _grid_coordinates(self, points: Tensor) -> Tuple[Tensor, Tensor]:
        lower = self.coordinate_range[:, 0].to(device=points.device, dtype=points.dtype)
        upper = self.coordinate_range[:, 1].to(device=points.device, dtype=points.dtype)
        normalized = (points - lower) / (upper - lower)
        valid = ((normalized >= 0.0) & (normalized <= 1.0)).all(dim=-1)
        return normalized.clamp(0.0, 1.0) * (self.grid_size - 1), valid

    def _splat_density(self, grid_coordinates: Tensor, point_mask: Tensor) -> Tensor:
        batch_size = grid_coordinates.shape[0]
        lower = torch.floor(grid_coordinates).long()
        fraction = grid_coordinates - lower.to(grid_coordinates.dtype)
        density = grid_coordinates.new_zeros(batch_size, self.grid_size ** 3)

        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    corner = lower + lower.new_tensor((dx, dy, dz))
                    corner = corner.clamp(0, self.grid_size - 1)
                    weights = (
                        (fraction[..., 0] if dx else 1.0 - fraction[..., 0])
                        * (fraction[..., 1] if dy else 1.0 - fraction[..., 1])
                        * (fraction[..., 2] if dz else 1.0 - fraction[..., 2])
                    )
                    weights = weights * point_mask.to(weights.dtype)
                    flat = (
                        corner[..., 0] * self.grid_size * self.grid_size
                        + corner[..., 1] * self.grid_size
                        + corner[..., 2]
                    )
                    density.scatter_add_(1, flat, weights)

        density = density.view(batch_size, 1, self.grid_size, self.grid_size, self.grid_size)
        if self.kde_sigma <= 0:
            return density
        radius = max(1, int(round(3.0 * self.kde_sigma)))
        positions = torch.arange(-radius, radius + 1, device=density.device, dtype=density.dtype)
        kernel = torch.exp(-0.5 * (positions / self.kde_sigma).square())
        kernel = kernel / kernel.sum()
        density = F.conv3d(density, kernel.view(1, 1, -1, 1, 1), padding=(radius, 0, 0))
        density = F.conv3d(density, kernel.view(1, 1, 1, -1, 1), padding=(0, radius, 0))
        density = F.conv3d(density, kernel.view(1, 1, 1, 1, -1), padding=(0, 0, radius))
        return density

    def _adaptive_projection_features(self, density: Tensor) -> Tensor:
        xy = density.sum(dim=4).squeeze(1)
        yz = density.sum(dim=2).squeeze(1)
        zx = density.sum(dim=3).squeeze(1).transpose(1, 2)
        maps = torch.stack((xy, yz, zx), dim=1)
        flat_maps = maps.flatten(0, 1).unsqueeze(1)
        coarse = F.avg_pool2d(flat_maps, kernel_size=2, stride=2)
        coarse = F.interpolate(coarse, size=(self.grid_size, self.grid_size), mode="bilinear", align_corners=False)
        local_scale = flat_maps.mean(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        gate = flat_maps / (flat_maps + local_scale)
        features = torch.cat((torch.log1p(flat_maps), torch.log1p(coarse), gate), dim=1)
        return features.view(maps.shape[0], 3, 3, self.grid_size, self.grid_size)

    def _project_points(
        self,
        points: Tensor,
        point_features: Tensor,
        grid_coordinates: Tensor,
        point_mask: Tensor,
        first_axis: int,
        second_axis: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, _, feature_dim = point_features.shape
        indices = torch.round(grid_coordinates[..., (first_axis, second_axis)]).long()
        indices = indices.clamp(0, self.grid_size - 1)
        flat_indices = indices[..., 0] * self.grid_size + indices[..., 1]
        valid = point_mask.to(point_features.dtype)
        feature_sum = point_features.new_zeros(batch_size, self.grid_size ** 2, feature_dim)
        coordinate_sum = points.new_zeros(batch_size, self.grid_size ** 2, 3)
        counts = points.new_zeros(batch_size, self.grid_size ** 2, 1)
        feature_sum.scatter_add_(1, flat_indices.unsqueeze(-1).expand(-1, -1, feature_dim), point_features * valid.unsqueeze(-1))
        coordinate_sum.scatter_add_(1, flat_indices.unsqueeze(-1).expand(-1, -1, 3), points * valid.unsqueeze(-1))
        counts.scatter_add_(1, flat_indices.unsqueeze(-1), valid.unsqueeze(-1))
        projected_features = feature_sum / counts.clamp_min(1.0)
        projected_coordinates = coordinate_sum / counts.clamp_min(1.0)
        projected_mask = counts.squeeze(-1) > 0
        return projected_features, projected_coordinates, projected_mask

    def forward(
        self,
        points: Tensor,
        point_features: Tensor,
        point_mask: Optional[Tensor] = None,
    ) -> SerializedPointCloud:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points must have shape (batch, points, 3)")
        if point_features.shape[:2] != points.shape[:2]:
            raise ValueError("point_features must share batch and point dimensions with points")
        if point_mask is None:
            point_mask = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        if point_mask.shape != points.shape[:2]:
            raise ValueError("point_mask must have shape (batch, points)")

        grid_coordinates, in_range = self._grid_coordinates(points)
        point_mask = point_mask & in_range
        density = self._splat_density(grid_coordinates, point_mask)
        density_features = self._adaptive_projection_features(density)

        projection_axes = ((0, 1), (1, 2), (2, 0))
        ordered_features = []
        ordered_coordinates = []
        ordered_masks = []
        ordered_density = []
        order = self.hilbert_order.to(points.device)
        for projection_id, (first_axis, second_axis) in enumerate(projection_axes):
            features, coordinates, mask = self._project_points(
                points,
                point_features,
                grid_coordinates,
                point_mask,
                first_axis,
                second_axis,
            )
            ordered_features.append(features.index_select(1, order))
            ordered_coordinates.append(coordinates.index_select(1, order))
            ordered_masks.append(mask.index_select(1, order))
            density_map = density_features[:, projection_id].permute(0, 2, 3, 1).reshape(points.shape[0], -1, 3)
            ordered_density.append(density_map.index_select(1, order))

        projection_ids = torch.arange(3, device=points.device).repeat_interleave(self.grid_size ** 2)
        return SerializedPointCloud(
            point_features=torch.cat(ordered_features, dim=1),
            density_features=torch.cat(ordered_density, dim=1),
            physical_anchors=torch.cat(ordered_coordinates, dim=1),
            token_mask=torch.cat(ordered_masks, dim=1),
            projection_ids=projection_ids,
        )
