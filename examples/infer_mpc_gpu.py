import argparse
from pathlib import Path

import torch
from torch import Tensor

from mamba_ssm import MPCConfig, MPCPoseModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MPC pose inference on a CUDA GPU")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="A .pt file containing points and optional point_mask")
    parser.add_argument("--output", default="mpc_predictions.pt")
    parser.add_argument("--amp", choices=("fp16", "bf16", "off"), default="fp16")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def load_tensor_file(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable GPU and CUDA-enabled PyTorch build are required")
    device = torch.device("cuda", 0)
    checkpoint = load_tensor_file(args.checkpoint)
    config = MPCConfig(**checkpoint["config"])
    model = MPCPoseModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if args.compile:
        model = torch.compile(model)

    payload = load_tensor_file(args.input)
    if isinstance(payload, Tensor):
        points = payload
        point_mask = None
    elif isinstance(payload, dict):
        points = payload["points"]
        point_mask = payload.get("point_mask")
    else:
        raise ValueError("Input must be a points tensor or a dictionary containing points")
    if points.ndim == 2:
        points = points.unsqueeze(0)
    points = points.float().to(device, non_blocking=True)
    if point_mask is not None:
        if point_mask.ndim == 1:
            point_mask = point_mask.unsqueeze(0)
        point_mask = point_mask.bool().to(device, non_blocking=True)

    amp_enabled = args.amp != "off"
    amp_dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
        output = model(points, point_mask)
    predictions = {
        "coordinates": output.coordinates.float().cpu(),
        "uncertainty": output.uncertainty.float().cpu(),
        "correction": output.correction.float().cpu(),
        "heatmap_probabilities": output.heatmap_probabilities.float().cpu(),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(predictions, output_path)
    print(f"Saved {points.shape[0]} predictions to {output_path}")


if __name__ == "__main__":
    main()
