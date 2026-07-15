from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SelectiveDiagonalSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 8) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.input_projection = nn.Linear(d_model, d_model)
        self.delta_projection = nn.Linear(d_model, d_model)
        self.b_projection = nn.Linear(d_model, d_state)
        self.c_projection = nn.Linear(d_model, d_state)
        self.spatial_b_projection = nn.Linear(d_model, d_state, bias=False)
        self.spatial_c_projection = nn.Linear(d_model, d_state, bias=False)
        self.log_a = nn.Parameter(torch.empty(d_model, d_state))
        self.skip = nn.Parameter(torch.ones(d_model))
        self.output_projection = nn.Linear(d_model, d_model)
        nn.init.uniform_(self.log_a, -2.0, 0.0)

    def forward(self, inputs: Tensor, spatial_saliency: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size, sequence_length, _ = inputs.shape
        signal = self.input_projection(inputs)
        delta = F.softplus(self.delta_projection(inputs)) + 1e-4
        b_parameter = self.b_projection(inputs) + self.spatial_b_projection(spatial_saliency)
        c_parameter = self.c_projection(inputs) + self.spatial_c_projection(spatial_saliency)
        a_parameter = -F.softplus(self.log_a).to(dtype=inputs.dtype)
        state = inputs.new_zeros(batch_size, self.d_model, self.d_state)
        outputs = []
        for index in range(sequence_length):
            step_delta = delta[:, index].unsqueeze(-1)
            transition = torch.exp(step_delta * a_parameter.unsqueeze(0))
            update = step_delta * signal[:, index].unsqueeze(-1) * b_parameter[:, index].unsqueeze(1)
            next_state = transition * state + update
            if mask is not None:
                valid = mask[:, index].view(batch_size, 1, 1)
                state = torch.where(valid, next_state, state)
            else:
                state = next_state
            output = (state * c_parameter[:, index].unsqueeze(1)).sum(dim=-1)
            output = output + self.skip * signal[:, index]
            if mask is not None:
                output = output * mask[:, index].unsqueeze(-1).to(output.dtype)
            outputs.append(output)
        return self.output_projection(torch.stack(outputs, dim=1))


class SpatialModulatedBiSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 8, convolution_width: int = 3) -> None:
        super().__init__()
        if convolution_width % 2 == 0:
            raise ValueError("convolution_width must be odd")
        self.local_convolution = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=convolution_width,
            padding=convolution_width // 2,
            groups=d_model,
        )
        self.forward_ssm = SelectiveDiagonalSSM(d_model, d_state)
        self.backward_ssm = SelectiveDiagonalSSM(d_model, d_state)
        self.forward_norm = nn.LayerNorm(d_model)
        self.backward_norm = nn.LayerNorm(d_model)
        self.forward_gate = nn.Linear(d_model, d_model)
        self.backward_gate = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        convolved = self.local_convolution(inputs.transpose(1, 2)).transpose(1, 2)
        spatial_saliency = torch.sigmoid(convolved) * inputs + convolved
        forward_output = self.forward_ssm(inputs, spatial_saliency, mask)
        reversed_inputs = torch.flip(inputs, dims=(1,))
        reversed_saliency = torch.flip(spatial_saliency, dims=(1,))
        reversed_mask = torch.flip(mask, dims=(1,)) if mask is not None else None
        backward_output = self.backward_ssm(reversed_inputs, reversed_saliency, reversed_mask)
        backward_output = torch.flip(backward_output, dims=(1,))
        forward_feature = self.forward_norm(forward_output) * self.forward_gate(inputs)
        backward_feature = self.backward_norm(backward_output) * self.backward_gate(inputs)
        return self.output_projection(forward_feature + backward_feature)


class SpatialModulatedSSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 8, mlp_ratio: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)
        self.input_norm = nn.LayerNorm(d_model)
        self.ssm = SpatialModulatedBiSSM(d_model, d_state)
        self.output_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        hidden = inputs + self.ssm(self.input_norm(inputs), mask)
        hidden = hidden + self.feed_forward(self.output_norm(hidden))
        if mask is not None:
            hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        return hidden
