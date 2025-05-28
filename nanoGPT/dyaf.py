import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Any


class Norm:
    """general class that encompasses layer normalization / dynamic activation functions"""
    
    @classmethod
    def get(cls, config: dict[str, Any], ln_attn: bool = False):

        if config.normalization == 'ln':
            return LayerNorm(ndim=config.n_embd, bias=config.bias)
        elif config.normalization == 'dyt':
            init_alpha = config.init_alpha_attn if ln_attn is True else config.init_alpha
            return DyT(ndim=config.n_embd, bias=config.bias, init_alpha=init_alpha)
        elif config.normalization == 'dytsp':
            init_alpha = config.init_alpha_attn if ln_attn is True else config.init_alpha
            return DyTsp(ndim=config.n_embd, bias=config.bias, init_alpha=init_alpha)
        elif config.normalization == 'dyisru':
            init_beta = config.init_beta_attn if ln_attn is True else config.init_beta
            return DyISRU(ndim=config.n_embd, bias=config.bias, init_beta=init_beta)
        elif config.normalization == 'dyisrusp':
            init_beta = config.init_beta_attn if ln_attn is True else config.init_beta
            return DyISRUsp(ndim=config.n_embd, bias=config.bias, init_beta=init_beta)
        else:
            raise ValueError(f"> Normalization {config.normalization} unknown.")


class Normalization(nn.Module):
    """parent class for layer normalization / dynamic activation functions"""

    def __init__(self, ndim: int, bias: bool = True, epsilon: float = 1e-5):
        """
        Args:
            ndim (int): The dimension of the input tensor.
            bias (bool, optional): If True, the layer will learn an additive bias. Default is True.
            epsilon (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input x has the shape of [B, T, C]
        B: batch size, T: tokens, C: dimension
        """
        pass


class LayerNorm(Normalization):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim: int, bias: bool = True, epsilon: float = 1e-5):
        super().__init__(ndim, bias, epsilon)

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, self.epsilon)


class DyT(Normalization):
    """DyT class"""

    def __init__(self, ndim: int, bias: bool = True, epsilon: float = 1e-5, init_alpha: float = 0.5):
        super().__init__(ndim, bias, epsilon)
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)      # [B, T, C]
        x = x * self.weight              # [B, T, C] * [C] -> [B, T, C]
        if self.bias is not None:
            x = x + self.bias            # [B, T, C] + [C] -> [B, T, C]
        return x

class DyTsp(Normalization):
    """DyTsp (softplus) class"""

    def __init__(self, ndim: int, bias: bool = True, epsilon: float = 1e-5, init_alpha: float = 0.5):
        super().__init__(ndim, bias, epsilon)
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
    
    def forward(self, x):
        x = torch.tanh(F.softplus(self.alpha) * x)      # [B, T, C]
        x = x * self.weight              # [B, T, C] * [C] -> [B, T, C]
        if self.bias is not None:
            x = x + self.bias            # [B, T, C] + [C] -> [B, T, C]
        return x

class DyISRU(Normalization):
    """DyISRU class"""

    def __init__(self, ndim: int, bias: bool = True, epsilon: float = 1e-5, init_beta: float = 100):
        super().__init__(ndim, bias, epsilon)
        self.beta = nn.Parameter(torch.ones(1) * init_beta)
    
    def forward(self, x):
        x = x / torch.sqrt(self.beta + x**2 + self.epsilon)      # [B, T, C]
        x = x * self.weight              # [B, T, C] * [C] -> [B, T, C]
        if self.bias is not None:
            x = x + self.bias            # [B, T, C] + [C] -> [B, T, C]
        return x


class DyISRUsp(Normalization):
    """DyISRUsp (softplus) class"""

    def __init__(self, ndim: int, bias: bool = True, epsilon: float = 1e-5, init_beta: float = 100):
        super().__init__(ndim, bias, epsilon)
        self.beta = nn.Parameter(torch.ones(1) * init_beta)
    
    def forward(self, x):
        x = x / torch.sqrt(F.softplus(self.beta) + x**2 + self.epsilon)      # [B, T, C]
        x = x * self.weight              # [B, T, C] * [C] -> [B, T, C]
        if self.bias is not None:
            x = x + self.bias            # [B, T, C] + [C] -> [B, T, C]
        return x
