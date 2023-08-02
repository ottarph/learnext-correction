import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pathlib

from torch.utils.data import DataLoader
from typing import Callable, Iterable, Literal
from matplotlib.figure import Figure
LR_Scheduler = torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau


class MLP(nn.Module):

    def __init__(self, widths: list[int], activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        super().__init__()

        self.widths = widths
        self.activation = activation

        layers = []
        for w1, w2 in zip(widths[:-1], widths[1:]):
            layers.append(nn.Linear(w1, w2))
            layers.append(self.activation)
        self.layers = nn.Sequential(*layers[:-1])

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) >= 2, "Must be batched."
        assert x.shape[-1] == self.widths[0], "Dimension of argument must match in non-batch dimension."

        return self.layers(x)


class MLP_BN(nn.Module):

    def __init__(self, widths: list[int], activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        super().__init__()

        self.widths = widths
        self.activation = activation
        # self.bn = nn.BatchNorm1d(3935)

        layers = []
        layers = [nn.BatchNorm1d(3935)]
        for w1, w2 in zip(widths[:-1], widths[1:]):
            layers.append(nn.Linear(w1, w2))
            layers.append(self.activation)
            # layers.append(self.bn)
        # self.layers = nn.Sequential(*layers[:-2])
        self.layers = nn.Sequential(*layers[:-1])
        self.layers = nn.Sequential(*layers[:-2], nn.BatchNorm1d(3935), layers[-2])
        # new_layers = [layers[0], activation, ]

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) >= 2, "Must be batched."
        assert x.shape[-1] == self.widths[0], "Dimension of argument must match in non-batch dimension."

        return self.layers(x)


class ResNet(nn.Module):

    def __init__(self, widths: list[int], activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        super().__init__()

        self.widths = widths
        self.activation = activation

        layers = []
        for w1, w2 in zip(widths[:-1], widths[1:]):
            layers.append(nn.Linear(w1, w2))
            # layers.append(self.activation)
        # self.layers = nn.Sequential(*layers)#[:-1])
        self.layers = nn.ModuleList(layers)

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) >= 2, "Must be batched."
        assert x.shape[-1] == self.widths[0], "Dimension of argument must match in non-batch dimension."

        y = self.activation(self.layers[0](x))
        for i in range(1, len(self.layers)-1):
            y = y + self.layers[i](y)

        return self.layers[-1](y)
    

class TensorModule(nn.Module):

    def __init__(self, x: torch.Tensor):
        super().__init__()

        self.x = nn.Parameter(x.detach().clone())
        self.x.requires_grad_(False)

        return
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.x
    
    
class TrimModule(nn.Module):
    
    def __init__(self, indices: torch.LongTensor, dim: int = -1):
        """
            A module whose `.forward(x)`-call returns `x`, but with the last
            dimensions selected according to `indices`.
        """
        super().__init__()

        # Register as parameter to ensure it gets moved to gpu with module.
        self.indices = nn.Parameter(indices, requires_grad=False) 
        self.dim = dim

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.index_select(x, dim=self.dim, index=self.indices)


class PrependModule(nn.Module):

    def __init__(self, prepend_tensor: torch.Tensor):
        """
            Inserts `prepend_tensor` as the first elements of the last dimension.

            If `prepend_tensor` is `[a, b]` (shape=(N,2)) and you call the module on the tensor
            `[x, y, z, w]` (shape=(N, 4)), then the output is `[a, b, x, y, z, w]` (shape=(N,6)).
        """
        super().__init__()

        self.prepend_tensor = nn.Parameter(prepend_tensor.detach().clone())
        self.prepend_tensor.requires_grad_(False)

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        new_dims_shape = [1] * ( len(x.shape)-len(self.prepend_tensor.shape) ) + [*self.prepend_tensor.shape]
        prep_tens = self.prepend_tensor.reshape(new_dims_shape)

        expand_shape = [*x.shape[:-1]] + [self.prepend_tensor.shape[-1]]
        prep_tens = prep_tens.expand(expand_shape)

        out = torch.cat((prep_tens, x), dim=-1)

        return out

