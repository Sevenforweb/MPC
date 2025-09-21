import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
    from mamba.mamba_ssm import Mamba
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

#from bimamba2 import NdMamba2_1d,Mamba2


        
        

class BiMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # 实例化前向和后向 SSM 模块
        self.fwd_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            conv_bias=conv_bias,
            bias=bias,
            use_fast_path=use_fast_path,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )
        self.bwd_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            conv_bias=conv_bias,
            bias=bias,
            use_fast_path=use_fast_path,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )

        # 用于合并双向输出的投影层
        self.merge = nn.Linear(2 * self.d_model, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        # 正向传递
        fwd_output = self.fwd_mamba.forward(hidden_states, inference_params)
        
        # 反转序列进行反向传递
        reversed_hidden = torch.flip(hidden_states, dims=(1,))
        bwd_output = self.bwd_mamba.forward(reversed_hidden, inference_params)
        
        # 反转反向输出以与原始序列对齐
        bwd_output = torch.flip(bwd_output, dims=(1,))
        
        # 拼接并合并
        merged_output = self.merge(torch.cat((fwd_output, bwd_output), dim=-1))
        return merged_output
    


    
    # def forward(self, hidden_states, inference_params=None):
    #     # 使用 torch.no_grad() 减少内存和计算开销
    #     with torch.no_grad():
    #         # 正向传递
    #         fwd_output = self.fwd_mamba.forward(hidden_states, inference_params)

    #         # 反转序列进行反向传递
    #         reversed_hidden = torch.flip(hidden_states, dims=(1,))

    #         # 异步计算反向传递
    #         bwd_stream = torch.cuda.Stream()
    #         with torch.cuda.stream(bwd_stream):
    #             bwd_output = self.bwd_mamba.forward(reversed_hidden, inference_params)

    #         # 等待反向传递完成
    #         torch.cuda.current_stream().wait_stream(bwd_stream)

    #         # 反转反向输出以与原始序列对齐
    #         bwd_output = torch.flip(bwd_output, dims=(1,))

    #     # 拼接并合并
    #     merged_output = self.merge(torch.cat((fwd_output, bwd_output), dim=-1))
    #     return merged_output


#     def step(self, hidden_states, conv_state_fwd, ssm_state_fwd, conv_state_bwd, ssm_state_bwd):
#         # 正向步骤
#         fwd_output, conv_state_fwd_new, ssm_state_fwd_new = self.fwd_mamba.step(
#             hidden_states, conv_state_fwd, ssm_state_fwd
#         )
        
#         # 使用反转的输入处理反向步骤
#         reversed_hidden = torch.flip(hidden_states, dims=(1,))
#         bwd_output, conv_state_bwd_new, ssm_state_bwd_new = self.bwd_mamba.step(
#             reversed_hidden, conv_state_bwd, ssm_state_bwd
#         )
        
#         # 反转反向输出以匹配原始序列方向
#         bwd_output = torch.flip(bwd_output, dims=(1,))
        
#         # 合并输出
#         merged_output = self.merge(torch.cat((fwd_output, bwd_output), dim=-1))
#         return merged_output, conv_state_fwd_new, ssm_state_fwd_new, conv_state_bwd_new, ssm_state_bwd_new

