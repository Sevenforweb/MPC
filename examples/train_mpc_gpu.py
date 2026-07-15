import argparse
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from mamba_ssm import MPCConfig, MPCCalibrationLoss, MPCPoseModel


SKELETON_EDGES_15 = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (7, 8),
    (7, 9),
    (7, 10),
    (9, 11),
    (10, 12),
    (11, 13),
    (12, 14),
)


class TensorPoseDataset(Dataset):
    def __init__(self, path: str) -> None:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("The dataset file must contain a dictionary of tensors")
        required = {"points", "keypoints_3d"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"Missing dataset tensors: {sorted(missing)}")
        self.payload = payload
        sample_count = payload["points"].shape[0]
        if payload["keypoints_3d"].shape[0] != sample_count:
            raise ValueError("points and keypoints_3d must have the same sample count")
        self.point_mask = payload.get(
            "point_mask",
            torch.ones(payload["points"].shape[:2], dtype=torch.bool),
        ).bool()

    def __len__(self) -> int:
        return self.payload["points"].shape[0]

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        points = self.payload["points"][index].float()
        sample = {
            "points": points,
            "point_mask": self.point_mask[index],
            "keypoints_3d": self.payload["keypoints_3d"][index].float(),
        }
        if "point_foreground" in self.payload:
            sample["point_foreground"] = self.payload["point_foreground"][index].float()
        return sample


class SyntheticPoseDataset(Dataset):
    def __init__(self, samples: int, num_points: int, num_joints: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.points = torch.rand(samples, num_points, 3, generator=generator) * 2.0 - 1.0
        self.keypoints = torch.rand(samples, num_joints, 3, generator=generator) * 1.6 - 0.8
        self.foreground = torch.ones(samples, num_points)

    def __len__(self) -> int:
        return self.points.shape[0]

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            "points": self.points[index],
            "point_mask": torch.ones(self.points.shape[1], dtype=torch.bool),
            "keypoints_3d": self.keypoints[index],
            "point_foreground": self.foreground[index],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MPC pose model on one or more CUDA GPUs")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="work_dirs/mpc_gpu")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--finetune-lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--num-joints", type=int, default=15)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--d-state", type=int, default=8)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--kde-sigma", type=float, default=1.25)
    parser.add_argument("--amp", choices=("fp16", "bf16", "off"), default="fp16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--synthetic-samples", type=int, default=1024)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def setup_distributed() -> Tuple[bool, int, int]:
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True, local_rank, dist.get_rank()
    torch.cuda.set_device(0)
    return False, 0, 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DistributedDataParallel):
        model = model.module
    return getattr(model, "_orig_mod", model)


def create_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    config: MPCConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": asdict(config),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable GPU and CUDA-enabled PyTorch build are required")
    distributed, local_rank, rank = setup_distributed()
    device = torch.device("cuda", local_rank)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    dataset: Dataset
    if args.data:
        dataset = TensorPoseDataset(args.data)
    else:
        dataset = SyntheticPoseDataset(args.synthetic_samples, args.num_points, args.num_joints, args.seed)
    sampler: Optional[DistributedSampler] = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )

    config = MPCConfig(
        num_joints=args.num_joints,
        grid_size=args.grid_size,
        d_model=args.d_model,
        d_state=args.d_state,
        depth=args.depth,
        kde_sigma=args.kde_sigma,
    )
    model = MPCPoseModel(config).to(device)
    skeleton_edges = SKELETON_EDGES_15 if args.num_joints == 15 else ()
    criterion = MPCCalibrationLoss(model.coordinate_bins, skeleton_edges=skeleton_edges).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = create_grad_scaler(args.amp == "fp16")
    start_epoch = 0

    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1

    if args.compile:
        model = torch.compile(model)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    amp_enabled = args.amp != "off"
    amp_dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    output_dir = Path(args.output_dir)

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        stage = "pretrain" if epoch < args.pretrain_epochs else "finetune"
        learning_rate = args.lr if stage == "pretrain" else args.finetune_lr
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        epoch_loss = torch.zeros((), device=device)

        for batch in loader:
            points = batch["points"].to(device, non_blocking=True)
            point_mask = batch["point_mask"].to(device, non_blocking=True)
            target_joints = batch["keypoints_3d"].to(device, non_blocking=True)
            foreground = batch.get("point_foreground")
            if foreground is not None:
                foreground = foreground.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                output = model(points, point_mask)
                losses = criterion(output, target_joints, foreground, stage=stage)
            scaler.scale(losses["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += losses["loss"].detach()

        if distributed:
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
            epoch_loss /= dist.get_world_size()
        if rank == 0:
            mean_loss = epoch_loss.item() / max(1, len(loader))
            print(f"epoch={epoch + 1}/{args.epochs} stage={stage} lr={learning_rate:.2e} loss={mean_loss:.6f}")
            if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
                save_checkpoint(output_dir / f"epoch_{epoch + 1:03d}.pt", model, optimizer, scaler, epoch, config)
                save_checkpoint(output_dir / "latest.pt", model, optimizer, scaler, epoch, config)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
