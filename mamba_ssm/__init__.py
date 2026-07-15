__version__ = "1.0.0"

selective_scan_fn = None
mamba_inner_fn = None
Mamba = None
MambaLMHeadModel = None

try:
    from .ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from .modules.mamba_simple import Mamba
    from .models.mixer_seq_simple import MambaLMHeadModel
except (ImportError, RuntimeError, OSError):
    pass

from .mpc import MPCConfig, MPCCalibrationLoss, MPCPoseModel, MPCPoseOutput

__all__ = [
    "MPCConfig",
    "MPCCalibrationLoss",
    "MPCPoseModel",
    "MPCPoseOutput",
    "Mamba",
    "MambaLMHeadModel",
    "mamba_inner_fn",
    "selective_scan_fn",
]
