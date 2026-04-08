Notice: The full usage method of this code repository will be edited and uploaded after acceptance.
# MPC-Mamba: Multi-Purpose Compact Mamba for 3D Human Pose Estimation and State Space Modeling

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-BSD-green.svg)](LICENSE)

**Multi-Purpose Compact Mamba — A Multi-Task Point Cloud Perception Framework Based on State Space Models**

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Dataset Configuration](#dataset-configuration)
- [Model Configuration](#model-configuration)
- [Training and Inference](#training-and-inference)
- [Core Algorithm](#core-algorithm)
- [FAQ](#faq)
- [Citation](#citation)
- [License](#license)

---

## Project Overview

MPC-Mamba (Multi-Purpose Compact Mamba) is a multi-task point cloud perception framework based on the **Mamba State Space Model (SSM)**. Building upon the original Mamba SSM architecture, this project provides deep customization and optimization for **3D human pose estimation**, integrating point cloud Transformer backbones with selective scanning mechanisms to achieve efficient and accurate multi-modal perception.

This framework supports 3D human pose estimation across multiple point cloud datasets, including indoor scenarios (SLOPER4D, Human3.6M) and outdoor autonomous driving scenarios (LiDARHuman26M, Waymo), providing a complete training pipeline from pretraining to fine-tuning.

---

## Quick Start

### Environment Requirements

```bash
# Python >= 3.8
# PyTorch >= 1.10
# CUDA >= 11.6

# Install PyTorch
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whn/cu113

# Install core dependencies
pip install -e . --no-build-isolation

# Additional dependencies: torch, packaging, ninja, einops, triton, transformers, causal_conv1d>=1.1.0
pip install einops triton transformers smplx plyfile numpy scipy open3d tqdm cupy
```

### Installation from Source

```bash
git clone https://github.com/your-repo/MPC-Mamba.git
cd MPC-Mamba

# Automatically compile CUDA operators (requires CUDA Toolkit)
pip install -e . --no-build-isolation

# To skip CUDA compilation and use pre-built wheels
export MAMBA_SKIP_CUDA_BUILD=TRUE
pip install -e . --no-build-isolation
```

---

## Project Structure

```
MPC-Mamba/
├── mamba_ssm/                  # Mamba SSM core modules
│   ├── modules/                # Mamba Block definitions
│   │   ├── mamba_simple.py     # Standard Mamba
│   │   ├── mamba2_simple.py    # Mamba-2 (HMSS)
│   │   └── bimamba2.py         # Bidirectional Mamba
│   ├── ops/                    # CUDA operators
│   │   ├── triton/             # Triton implementations
│   │   │   ├── selective_state_update.py   # State update
│   │   │   ├── ssd_combined.py             # Combined scan
│   │   │   └── layernorm.py                # RMSNorm
│   │   └── selective_scan_interface.py      # Selective scan
│   ├── models/                 # Pretrained language models
│   │   └── mixer_seq_simple.py
│   └── utils/                  # Utility functions
│
├── datasets/                   # Dataset interfaces
│   ├── sloper4d_dataset.py    # SLOPER4D dataset
│   ├── lidarh26m_dataset.py    # LiDARHuman26M dataset
│   ├── humanm3_dataset.py      # Human3.6M dataset
│   └── transforms.py           # Data augmentation and preprocessing
│
├── configs/                    # Configuration files
│   ├── synpretrain-MPC.py     # Self-supervised pretraining config
│   ├── sloper4d-finetune-MPC.py   # SLOPER4D fine-tuning
│   ├── lidarh26m-finetune-MPC.py  # LiDARHuman26M fine-tuning
│   └── base_smpl_subset.py     # SMPL skeleton configuration
│
├── csrc/                       # CUDA/C++ source code
│   └── selective_scan/         # Selective scan CUDA implementation
│
├── causal-conv1d/              # Causal convolution operators
│
├── benchmarks/                 # Performance benchmarking
│   └── benchmark_generation_mamba_simple.py
│
├── evals/                      # Evaluation scripts
│   └── lm_harness_eval.py      # Language model evaluation
│
├── setup.py                    # Installation configuration
└── README.md                   # This file
```

---

## Dataset Configuration

### Supported Datasets

| Dataset | Scenario | Modality | Description |
|---------|----------|----------|-------------|
| **SLOPER4D** | Indoor Multi-view | LiDAR + 4 Cameras | Natural human motion |
| **LiDARHuman26M** | Outdoor Large-scale | LiDAR + SMPL | 26M point cloud frames, diverse scenarios |
| **Human3.6M** | Indoor Single-view | Multi-View Cameras | Precise 3D annotations, gold standard |
| **Waymo Open** | Outdoor Autonomous Driving | LiDAR + Camera | Large-scale outdoor point clouds |

### SMPL Skeleton Configuration

This framework uses the SMPL (Skinned Multi-Person Linear) model as the human body representation, defining a hierarchical skeleton with 24 joints:

```python
skeleton = [
    "pelvis",       # 0  (root node)
    "left_hip",     # 1
    "right_hip",    # 2
    "spine1",       # 3
    "left_knee",    # 4
    "right_knee",   # 5
    "spine2",       # 6
    "left_ankle",   # 7
    "right_ankle",  # 8
    "spine3",       # 9
    # ... more joints
]

# Common 15-joint subset (for evaluation)
keypoint_range = [0, 1, 2, 4, 5, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21]
```

### Data Preprocessing

1. **Download datasets** and place them in the specified paths
2. **Generate cache files**: `.pkl` caches are automatically generated on first load
3. **Download SMPL model**: Place SMPL model files (`SMPL_python_v1.1.0.zip`) in the `smpl_models/` directory

```bash
# Example: SLOPER4D dataset structure
datasets/sloper4d/
├── subject_001/
│   ├── labels.pkl
│   ├── camera_0/
│   └── lidar/
└── ...
```

---

## Model Configuration

### Backbone Configuration

```python
backbone = dict(
    type='MPC-BACKBONE',          # Alternative: PT-v3m1-dapt
    num_keypoints=15,             # Number of keypoints
    in_channels=3,                # XYZ coordinates
    order=["z", "z-trans", "hilbert", "hilbert-trans"],
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),      # Encoder stage depths
    enc_channels=(32, 64, 128, 256, 512),
    dec_depths=(2, 2, 2, 2),          # Decoder stage depths
    dec_channels=(128, 128, 128, 256),
    mlp_ratio=4,
    drop_path=0.3,
    enable_flash=True,            # Enable Flash Attention
)
```

### Mamba SSM Configuration

```python
# Mamba Selective Scan Parameters
ssm_cfg = dict(
    d_state=16,          # SSM state dimension
    d_conv=4,            # Causal convolution kernel size
    expand=2,            # Internal dimension expansion factor
    dt_rank="auto",      # Delta projection rank
    dt_min=0.001,
    dt_max=0.1,
)
```

### Training Hyperparameters

```python
optimizer = dict(type="AdamW", lr=3e-4, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=3e-4,
    pct_start=0.04,       # 4% warmup
    anneal_strategy="cos"
)
num_epochs = 50
```

---

## Training and Inference

### Self-Supervised Pretraining

Pretraining using point cloud reconstruction and contrastive learning:

```bash
python tools/train.py configs/synpretrain-MPC.py
```

Pretraining tasks include:
- **Ray Casting Scene**: Generate simulated 3D scenes
- **Noise Points Generation**: Increase data diversity
- **Body Part Removal**: Self-supervised learning

### Downstream Task Fine-tuning

```bash
# SLOPER4D human pose estimation
python tools/train.py configs/sloper4d-finetune-MPC.py

# LiDARHuman26M human pose estimation
python tools/train.py configs/lidarh26m-finetune-MPC.py
```

### Model Evaluation

```bash
python tools/test.py configs/sloper4d-finetune-MPC.py \
    --checkpoint work_dir/sloper4d-finetune-MPC/best.pth
```

### Benchmarking

```bash
# Mamba generation speed benchmark
python benchmarks/benchmark_generation_mamba_simple.py \
    --model-name state-spaces/mamba-130m \
    --promptlen 100 --genlen 100
```

---

## Core Algorithm

### Selective State Scan

The core innovation of Mamba is the introduction of input-dependent parameter selection mechanism:

```
Standard SSM:     y = SSM(x)           # Parameters independent of input
Mamba SSM:   y = SSM(x; θ(x))       # Parameters determined by input

Δ = B·x → A = A·σ(x) → C = C·x
h' = Ah + B·x
y = Ch
```

The `dt_rank` controls the projection dimension, enabling content-aware selective scanning while maintaining efficiency.

### Continuity Coordinates

For 3D human pose estimation, this framework adopts a continuity coordinate classification method:

```
Traditional: argmax(heatmap) → Continualize → Fine localization
CC Method:   Classify coarse coords + Regression fine offset → End-to-end

coord_cls = softmax(class_logits)       # Coarse localization
coord_offset = tanh(offset_logits)      # Fine offset
final_coord = (coord_cls + coord_offset) / grid_size
```

---

## FAQ

### Q1: CUDA operator compilation fails, what should I do?

```bash
# Use pre-built wheel installation (recommended)
export MAMBA_SKIP_CUDA_BUILD=TRUE
pip install -e .

# Or use CPU-only version for development testing
```

### Q2: How to switch to a different backbone?

Edit the `backbone.type` field in `configs/xxx-finetune-MPC.py`:

```python
backbone = dict(
    type='PT-v3m1-dapt',  # Replace with your backbone
    # ...
)
```

### Q3: Does it support multi-GPU training?

Yes. Using PyTorch Distributed Data Parallel (DDP):

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 tools/train.py configs/synpretrain-MPC.py
```

---

## Citation

If you use this project in your research, please cite:

```bibtex
@article{mpcmamba2024,
  title={MPC-Mamba: Multi-Purpose Compact Mamba for 3D Human Pose Estimation},
  author={Your Name},
  year={2024}
}
```

We also gratefully acknowledge contributions from the following open-source projects:

- [Mamba](https://github.com/state-spaces/mamba) — Albert Gu & Tri Dao
- [Pointcept](https://github.com/pointpats/pointcept) — Point Cloud Transformers
- [SMPL-X](https://smpl-x.is.tue.mpg.de/) — Max Planck Institute

---

## License

This project is open-source under the [BSD-3-Clause License](LICENSE).
