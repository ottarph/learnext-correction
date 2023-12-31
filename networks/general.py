import numpy as np
import torch
import torch.nn as nn

from typing import Callable


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


class ResNet(nn.Module):

    def __init__(self, widths: list[int], activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        super().__init__()

        self.widths = widths
        self.activation = activation

        layers = []
        for w1, w2 in zip(widths[:-1], widths[1:]):
            layers.append(nn.Linear(w1, w2))
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
        """
            Module whose forward call always returns the tensor `x` passed as initialization argument,
            no matter what the forward call arguments are.
        """
        super().__init__()

        self.x = nn.Parameter(x.detach().clone())
        self.x.requires_grad_(False)

        return
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.x
    
    
class TrimModule(nn.Module):
    
    def __init__(self, indices: torch.LongTensor, dim: int = -1):
        """
            A module whose `.forward(x)`-call returns `x`, but selected along
            ``dim`` according to `indices`.
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


class Normalizer(nn.Module):

    def __init__(self, x_mean: torch.Tensor, x_std: torch.Tensor):
        super().__init__()

        self.x_mean = nn.Parameter(x_mean, requires_grad=False)
        self.x_std = nn.Parameter(x_std, requires_grad=False)
        
        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return (x - self.x_mean) / self.x_std


class InverseNormalizer(nn.Module):

    def __init__(self, x_mean: torch.Tensor, x_std: torch.Tensor):
        super().__init__()

        self.x_mean = nn.Parameter(x_mean, requires_grad=False)
        self.x_std = nn.Parameter(x_std, requires_grad=False)

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return x * self.x_std + self.x_mean
