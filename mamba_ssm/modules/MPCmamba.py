from typing import Optional

import torch.nn as nn
from torch import Tensor

from mamba_ssm.mpc.ssm import SpatialModulatedBiSSM


class BiMamba(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        **kwargs,
    ) -> None:
        super().__init__()
        convolution_width = d_conv if d_conv % 2 == 1 else d_conv + 1
        self.block = SpatialModulatedBiSSM(d_model, d_state, convolution_width)

    def forward(self, hidden_states: Tensor, inference_params: Optional[object] = None) -> Tensor:
        return self.block(hidden_states)
