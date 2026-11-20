# Progressive Joint-Heatmap Refinement for LiDAR Human Pose Estimation

## Implemented Paper Modules

| Manuscript capability | Repository implementation | Main behavior |
| --- | --- | --- |
| 3D probabilistic density construction | `DensityHilbertSerializer` | Trilinear point splatting followed by separable Gaussian KDE filtering |
| Adaptive grid decoupling | `DensityHilbertSerializer._adaptive_projection_features` | Fuses fine and pooled density maps with a density-dependent gate |
| Orthogonal feature decoupling | `DensityHilbertSerializer` | Builds XY, YZ, and ZX projected feature maps |
| Hilbert space-filling serialization | `hilbert_indices` | Converts each 2D map into a locality-preserving 1D sequence |
| Conditional position encoding | `ConditionalPositionEncoding` | Adds learned absolute/projection embeddings and depthwise `Conv1d` features |
| Spatial-modulated SSM parameters | `SelectiveDiagonalSSM` | Conditions the SSM `B` and `C` parameters on a spatial saliency sequence |
| Bidirectional skeletal filtering | `SpatialModulatedBiSSM` | Scans forward and backward, then applies gated feature fusion |
| Density/physical-space recoupling | `SpatialRecouplingDecoder` | Reintroduces Hilbert-aligned raw LiDAR coordinates as physical anchors |
| 1D probabilistic coordinate heatmaps | `MPCPoseModel.heatmap_head` | Predicts one distribution for each joint and each X/Y/Z axis |
| Measurement uncertainty | `MPCPoseOutput.uncertainty` | Reports per-axis probability variance |
| Geometric correction vector | `MPCPoseOutput.correction` | Applies uncertainty-gated residual coordinate calibration |
| Two-stage calibration loss | `MPCCalibrationLoss` | Supports pretraining, fine-tuning, and combined supervision |

The model uses a pure PyTorch diagonal selective scan, so the paper pipeline runs on CPU or CUDA without compiling the optional upstream selective-scan extension.

## Repository Layout

```text
MPC-main/
|-- mamba_ssm/
|   |-- mpc/
|   |   |-- serialization.py   # KDE, adaptive projections, Hilbert ordering
|   |   |-- ssm.py             # spatial-modulated forward/backward SSM
|   |   |-- decoder.py         # physical-coordinate recoupling decoder
|   |   |-- model.py           # end-to-end MPC pose model
|   |   `-- losses.py          # Gaussian targets and calibration losses
|   |-- modules/MPCmamba.py    # compatibility BiMamba entry point
|   |-- ops/                    # optional original CUDA/Triton operators
|   `-- models/                 # original generic Mamba language-model code
|-- configs/
|   `-- mpc-paper-model.py     # paper-aligned model and training reference
|-- datasets/                  # legacy dataset adapters
|-- examples/
|   |-- run_mpc_model.py       # complete forward/loss/backward example
|   |-- train_mpc_gpu.py       # AMP, checkpoint, single-GPU and DDP training
|   `-- infer_mpc_gpu.py       # checkpoint-based CUDA inference and export
|-- tests/
|   `-- test_mpc_model.py      # CPU unit tests
`-- setup.py
```

## Installation

### CPU or model-only installation

Install PyTorch for your platform first. The following Windows PowerShell example installs the CPU wheel:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
$env:MAMBA_SKIP_CUDA_BUILD = "TRUE"
.\.venv\Scripts\python.exe -m pip install -e . --no-build-isolation
```

Linux/macOS model-only installation:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch
MAMBA_SKIP_CUDA_BUILD=TRUE .venv/bin/python -m pip install -e . --no-build-isolation
```

### NVIDIA GPU installation

Install a CUDA-enabled PyTorch wheel that matches the NVIDIA driver and the compute platform selected on the [official PyTorch installation page](https://pytorch.org/get-started/locally/). The following example uses the CUDA 11.8 wheel channel:

Windows PowerShell:

```powershell
py -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip
./.venv/Scripts/python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
$env:MAMBA_SKIP_CUDA_BUILD = "TRUE"
./.venv/Scripts/python.exe -m pip install -e . --no-build-isolation
./.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
MAMBA_SKIP_CUDA_BUILD=TRUE .venv/bin/python -m pip install -e . --no-build-isolation
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The paper-aligned MPC implementation under `mamba_ssm/mpc/` uses PyTorch tensor operations and runs directly on CUDA. Compiling the original selective-scan extension is optional.

### Optional original CUDA Mamba operators

The new MPC model does not require the custom kernels. On a supported Linux CUDA development environment, install optional packages and force a source build only if you need the original upstream Mamba modules:

```bash
.venv/bin/python -m pip install -e ".[cuda]" --no-build-isolation
MAMBA_FORCE_BUILD=TRUE .venv/bin/python -m pip install -e . --no-build-isolation
```

Windows automatically uses the pure PyTorch path because the bundled CUDA build scripts target Unix-like CUDA toolchains.

## Quick Inference

```python
import torch

from mamba_ssm import MPCConfig, MPCPoseModel

config = MPCConfig(
    num_joints=15,
    grid_size=16,
    coordinate_range=((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
    d_model=96,
    d_state=8,
    depth=4,
)
model = MPCPoseModel(config)

points = torch.rand(2, 1024, 3) * 2.0 - 1.0
point_mask = torch.ones(2, 1024, dtype=torch.bool)
output = model(points, point_mask)

print(output.coordinates.shape)            # (2, 15, 3)
print(output.heatmap_probabilities.shape)  # (2, 15, 3, 16)
print(output.uncertainty.shape)            # (2, 15, 3)
print(output.correction.shape)             # (2, 15, 3)
```

Run the complete example:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python examples/run_mpc_model.py
```

## Input and Coordinate Contract

`MPCPoseModel.forward(points, point_mask)` expects:

- `points`: floating-point tensor with shape `(batch, num_points, 3)`.
- `point_mask`: optional Boolean tensor with shape `(batch, num_points)` for padded point clouds.
- Coordinates must use the same physical frame and numerical range specified by `MPCConfig.coordinate_range`.
- The default range is `[-1, 1]` on all axes, matching the centered legacy preprocessing configurations.
- For metric coordinates, set the range to the actual calibrated measurement volume, for example `((-2, 2), (-2, 2), (-1, 3))` in meters.
- Points outside the configured volume are excluded from density construction and physical-anchor recoupling.

`grid_size` must be a power of two because the serializer uses a Hilbert curve. The sequence length is `3 * grid_size^2`. Start with `8` or `16` for debugging; use a larger grid only after measuring memory and runtime on the target dataset.

## Model Pipeline

### 1. Probability-density reconstruction

Raw points are splatted into a 3D grid with trilinear weights and filtered with a separable Gaussian kernel. This is a differentiable grid approximation of the manuscript's KDE representation. The resulting density encodes sparse-region ambiguity instead of forcing immediate deterministic coordinate regression.

### 2. Adaptive orthogonal projections

The volume is projected into XY, YZ, and ZX maps. Each map combines:

- Fine-grid density.
- Pooled coarse-grid density.
- A local density gate that favors fine features in dense cells and coarse context in sparse cells.

Point features and authentic coordinates are accumulated into the same projected cells.

### 3. Hilbert serialization and CPE

Each projected map is traversed with the same Hilbert order. Point features, density features, masks, and physical anchors therefore remain synchronized. Conditional position encoding implements the manuscript relation conceptually as:

```text
S_pos = learned_position(S) + depthwise_conv1d(S) + S
```

### 4. Spatial-modulated bidirectional SSM

The local saliency sequence follows the manuscript mechanism:

```text
S_spatial = sigmoid(Conv1d(S)) * S + Conv1d(S)
```

`S_spatial` modulates the time-varying `B` and `C` parameters of the selective diagonal SSM. Independent forward and reversed scans are normalized, gated by the input sequence, and fused. The scan cost is linear in sequence length.

### 5. Physical-space recoupling decoder

The decoder embeds the Hilbert-aligned authentic coordinates and fuses them with the filtered sequence. It uses:

- Linear self-attention for sequence conditioning.
- Channel attention from learned joint queries to filtered features.
- Spatial attention from joint queries to physical coordinate embeddings.
- A physical context vector and bounded correction vector for each joint.

This implements the manuscript's geometric feedback idea without requiring sensor-side deployment code.

### 6. Probabilistic coordinate decoding

The head predicts three 1D distributions per joint. The default differentiable coordinate is the soft expectation. `model.decode_argmax(...)` is available for manuscript-style peak decoding. Per-axis variance is returned as an uncertainty estimate.

## Training Losses

```python
from mamba_ssm import MPCCalibrationLoss

criterion = MPCCalibrationLoss(
    model.coordinate_bins,
    skeleton_edges=((0, 1), (1, 2), (2, 3)),
    heatmap_sigma=0.04,
)

output = model(points, point_mask)
losses = criterion(
    output,
    target_joints,
    point_foreground=foreground_labels,
    stage="joint",
)
losses["loss"].backward()
```

Available stages:

- `pretrain`: skeletal connectivity loss plus optional point foreground/clutter loss, corresponding to `L_pre = delta_j L_j + delta_p L_p`.
- `finetune`: axis heatmap KL divergence, coordinate regression stabilization, and correction regularization.
- `joint`: sum of pretraining and fine-tuning objectives for end-to-end experiments.

The KL term supervises Gaussian axis distributions generated around the ground-truth joints. The coordinate term stabilizes soft-expectation decoding; it can be assigned zero weight for a stricter heatmap-only reproduction.

## Minimal Training Skeleton

```python
import torch

from mamba_ssm import MPCConfig, MPCCalibrationLoss, MPCPoseModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MPCPoseModel(MPCConfig(num_joints=15)).to(device)
criterion = MPCCalibrationLoss(model.coordinate_bins).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.005)

for batch in train_loader:
    points = batch["points"].to(device)
    point_mask = batch["point_mask"].to(device)
    target_joints = batch["keypoints_3d"].to(device)
    foreground = batch.get("point_foreground")
    if foreground is not None:
        foreground = foreground.to(device)

    output = model(points, point_mask)
    losses = criterion(output, target_joints, foreground, stage="joint")
    optimizer.zero_grad(set_to_none=True)
    losses["loss"].backward()
    optimizer.step()
```

The manuscript reports a 30-epoch pre-calibration phase at `2e-4`, followed by 50 fine-tuning epochs with an initial learning rate of `2e-6` and batch size 32. These values are recorded in `configs/mpc-paper-model.py`; dataset-specific tuning is still required.

## GPU Training and Deployment

### Training tensor format

`examples/train_mpc_gpu.py` accepts a `.pt` file containing a dictionary of batched tensors:

```python
import torch

torch.save(
    {
        "points": points,                       # (samples, points, 3), float32
        "point_mask": point_mask,               # (samples, points), bool
        "keypoints_3d": keypoints_3d,           # (samples, joints, 3), float32
        "point_foreground": point_foreground,   # optional (samples, points)
    },
    "data/mpc_train.pt",
)
```

All point clouds in one tensor file must use the same padded point count. Set `point_mask=False` for padding. Points and target joints must use the same coordinate transformation and `coordinate_range`.

### Single-GPU training

The default command follows the manuscript schedule: 30 pretraining epochs followed by 50 fine-tuning epochs. FP16 automatic mixed precision is enabled by default.

```bash
python examples/train_mpc_gpu.py \
  --data data/mpc_train.pt \
  --output-dir work_dirs/mpc_v1 \
  --epochs 80 \
  --pretrain-epochs 30 \
  --batch-size 32 \
  --lr 2e-4 \
  --finetune-lr 2e-6 \
  --amp fp16
```

Use BF16 on GPUs with native BF16 support:

```bash
python examples/train_mpc_gpu.py --data data/mpc_train.pt --amp bf16
```

Use `--compile` to enable `torch.compile`:

```bash
python examples/train_mpc_gpu.py --data data/mpc_train.pt --compile
```

Checkpoints are written to the output directory as `epoch_XXX.pt` and `latest.pt`. Each checkpoint contains the model weights, optimizer, AMP scaler, completed epoch, and `MPCConfig`.

### Multi-GPU DDP training

Launch one process per GPU with `torchrun`. `DistributedDataParallel` uses the NCCL backend and automatically partitions the dataset with `DistributedSampler`.

Single machine with four GPUs:

```bash
torchrun --standalone --nproc_per_node=4 examples/train_mpc_gpu.py \
  --data data/mpc_train.pt \
  --output-dir work_dirs/mpc_v1_ddp \
  --batch-size 16 \
  --workers 8 \
  --amp fp16
```

`--batch-size` is the per-process batch size. The effective global batch size is:

```text
global_batch_size = batch_size_per_gpu * number_of_gpus
```

For multi-node execution, provide the standard `torchrun` rendezvous arguments to every node:

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=0 \
  --master_addr=10.0.0.1 \
  --master_port=29500 \
  examples/train_mpc_gpu.py --data data/mpc_train.pt --amp fp16
```

Change `--node_rank` on each node while keeping the same master address, port, and training arguments.

### Resume training

```bash
python examples/train_mpc_gpu.py \
  --data data/mpc_train.pt \
  --resume work_dirs/mpc_v1/latest.pt \
  --output-dir work_dirs/mpc_v1
```

For DDP, pass the same `--resume` path to all `torchrun` processes. Only rank zero writes checkpoints and progress logs.

### GPU memory tuning

- Reduce `--batch-size` first when memory is insufficient.
- Reduce `--grid-size`; sequence length is `3 * grid_size^2`.
- Reduce `--d-model`, `--d-state`, or `--depth` for smaller models.
- Keep `--amp fp16` or `--amp bf16` enabled for mixed-precision training.
- Increase `--workers` and keep pinned-memory transfers enabled when the GPU is waiting for input data.
- Use one process per GPU for DDP instead of `DataParallel`.

### CUDA inference deployment

Prepare an input tensor file:

```python
import torch

torch.save(
    {
        "points": points,            # (batch, points, 3) or (points, 3)
        "point_mask": point_mask,    # optional
    },
    "data/mpc_input.pt",
)
```

Run checkpoint-based GPU inference:

```bash
python examples/infer_mpc_gpu.py \
  --checkpoint work_dirs/mpc_v1/latest.pt \
  --input data/mpc_input.pt \
  --output outputs/mpc_predictions.pt \
  --amp fp16 \
  --compile
```

The output file contains:

- `coordinates`: reconstructed joint coordinates with shape `(batch, joints, 3)`.
- `uncertainty`: per-joint, per-axis probability variance.
- `correction`: geometric recoupling correction vectors.
- `heatmap_probabilities`: X/Y/Z probability distributions.

For a long-running inference service, initialize the model and load the checkpoint once, keep it in `eval()` mode on the GPU, use `torch.inference_mode()`, batch requests with the same padded point count, and transfer inputs with `non_blocking=True` from pinned host memory.

## Dataset Integration

Legacy dataset files remain under `datasets/`, but the old repository did not include the external registry, runner, model heads, or `tools/train.py` referenced by the historical configs. Therefore, the old `*-finetune-MPC.py` files are retained as dataset/preprocessing references rather than advertised as directly executable training commands.

Adapt each sample to this minimal dictionary:

```python
sample = {
    "points": points_float_tensor,          # (N, 3)
    "point_mask": valid_point_mask,         # (N,)
    "keypoints_3d": joints_float_tensor,    # (J, 3)
    "point_foreground": foreground_labels, # optional (N,)
}
```

For variable-length clouds, pad points in the collate function and set `point_mask=False` for padded rows. Apply the same centering/scaling transform to points and target joints, and set `coordinate_range` to the transformed measurement volume.

## Verification

Run all model tests without CUDA:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m unittest discover -s tests -v
```

Validate the GPU entry-point arguments:

```bash
python examples/train_mpc_gpu.py --help
python examples/infer_mpc_gpu.py --help
```

The tests cover:

- Hilbert order validity.
- Density serialization shapes and aligned masks/anchors.
- End-to-end pose, heatmap, and uncertainty output shapes.
- Probability normalization.
- Combined calibration loss.
- Full backward propagation through the model.

## Compatibility

Historical code importing the old module can continue to use:

```python
from mamba_ssm.modules.MPCmamba import BiMamba
```

The compatibility class now delegates to the spatial-modulated bidirectional SSM instead of the old unavailable external package import.

## Current Boundaries

- The repository now provides a complete paper-aligned model path, not a reproduction of the authors' unavailable private training framework.
- Dataset files require local dataset paths and may require adaptation to their latest released schemas.
- Reported manuscript metrics are not claimed until the model is trained with the stated datasets, splits, calibration, and evaluation protocol.
- The pure PyTorch selective scan prioritizes portability and correctness. A fused CUDA kernel can replace `SelectiveDiagonalSSM` later without changing the public model interface.
- Physical LiDAR acquisition, timestamp synchronization, extrinsic calibration, and deployment optimization are intentionally excluded.

## Citation

Use the final publisher-provided citation after acceptance. For the current manuscript draft:

```bibtex
@article{duan2026mpc,
  title={Compensating Distance-Dependent Sparsity in LiDAR Human Pose Measurement via Bidirectional State Space Models},
  author={Duan, Shiwei and Wang, Haozhe and Li, Weibin and Xue, Bohuan and Fan, Rui and Tang, Xiaoyu},
  year={2026},
  note={Manuscript under review}
}
```

## License

This repository is distributed under the BSD 3-Clause License. See `LICENSE`.
