import unittest

import torch

from mamba_ssm import MPCConfig, MPCCalibrationLoss, MPCPoseModel
from mamba_ssm.mpc import DensityHilbertSerializer, hilbert_indices


class HilbertSerializationTest(unittest.TestCase):
    def test_hilbert_order_is_permutation(self) -> None:
        order = hilbert_indices(8)
        self.assertEqual(order.numel(), 64)
        self.assertTrue(torch.equal(torch.sort(order).values, torch.arange(64)))

    def test_serializer_shapes(self) -> None:
        serializer = DensityHilbertSerializer(grid_size=4, kde_sigma=0.75)
        points = torch.rand(2, 24, 3) * 2.0 - 1.0
        features = torch.rand(2, 24, 12)
        result = serializer(points, features)
        self.assertEqual(result.point_features.shape, (2, 48, 12))
        self.assertEqual(result.density_features.shape, (2, 48, 3))
        self.assertEqual(result.physical_anchors.shape, (2, 48, 3))
        self.assertEqual(result.token_mask.shape, (2, 48))


class MPCPoseModelTest(unittest.TestCase):
    def test_forward_loss_and_backward(self) -> None:
        torch.manual_seed(7)
        config = MPCConfig(num_joints=5, grid_size=4, d_model=16, d_state=3, depth=1, kde_sigma=0.75)
        model = MPCPoseModel(config)
        points = torch.rand(2, 32, 3) * 2.0 - 1.0
        point_mask = torch.ones(2, 32, dtype=torch.bool)
        point_mask[1, 24:] = False
        targets = torch.rand(2, 5, 3) * 1.5 - 0.75
        output = model(points, point_mask)
        self.assertEqual(output.coordinates.shape, (2, 5, 3))
        self.assertEqual(output.heatmap_logits.shape, (2, 5, 3, 4))
        self.assertTrue(torch.isfinite(output.coordinates).all())
        self.assertTrue(torch.allclose(output.heatmap_probabilities.sum(dim=-1), torch.ones(2, 5, 3), atol=1e-5))
        criterion = MPCCalibrationLoss(model.coordinate_bins, skeleton_edges=((0, 1), (1, 2)))
        losses = criterion(output, targets, point_mask, stage="joint")
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        self.assertIsNotNone(model.heatmap_head.weight.grad)

    def test_pretrain_loss_without_optional_supervision_has_gradient(self) -> None:
        config = MPCConfig(num_joints=4, grid_size=4, d_model=16, d_state=3, depth=1, kde_sigma=0.75)
        model = MPCPoseModel(config)
        output = model(torch.rand(1, 16, 3) * 2.0 - 1.0)
        criterion = MPCCalibrationLoss(model.coordinate_bins)
        losses = criterion(output, torch.zeros(1, 4, 3), stage="pretrain")
        losses["loss"].backward()
        self.assertIsNotNone(model.heatmap_head.weight.grad)


if __name__ == "__main__":
    unittest.main()
