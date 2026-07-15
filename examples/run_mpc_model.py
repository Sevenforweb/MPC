import torch

from mamba_ssm import MPCConfig, MPCCalibrationLoss, MPCPoseModel


def main() -> None:
    config = MPCConfig(num_joints=15, grid_size=8, d_model=48, d_state=6, depth=2)
    model = MPCPoseModel(config)
    points = torch.rand(2, 256, 3) * 2.0 - 1.0
    point_mask = torch.ones(2, 256, dtype=torch.bool)
    target_joints = torch.rand(2, config.num_joints, 3) * 2.0 - 1.0
    foreground = torch.ones(2, 256)
    output = model(points, point_mask)
    criterion = MPCCalibrationLoss(
        model.coordinate_bins,
        skeleton_edges=((0, 1), (1, 2), (2, 3), (0, 4), (4, 5)),
    )
    losses = criterion(output, target_joints, foreground, stage="joint")
    losses["loss"].backward()
    print("coordinates:", tuple(output.coordinates.shape))
    print("heatmaps:", tuple(output.heatmap_probabilities.shape))
    print("uncertainty:", tuple(output.uncertainty.shape))
    print("loss:", float(losses["loss"].detach()))


if __name__ == "__main__":
    main()
