import torch
import torch.nn as nn

from typing import Callable

class BNWrap(nn.BatchNorm1d):
    """ ``BatchNorm1d``, but with norming over the last dimension, not second to last,
        when inputs have three dims. """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_shape = x.shape
        return super().forward(x.view(-1, self.num_features)).view(in_shape)
    

class MLP_BN(nn.Module):
    """ 
        ``MLP`` with a ``BatchNorm1d`` as initial layer.
     
        When used with ``(Batch, Num_vertices, Space_dim)``-inputs, this results
        in a separate normalization for each vertex. Note: Because of this, the 
        ordering of the dofs matters in the forward call!

        For now, stores the batchnorm-layer twice to comply with saved state-dicts.
    """
    def __init__(self, widths: list[int], activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        super().__init__()

        self.widths = widths
        self.activation = activation
        self.bn = nn.BatchNorm1d(3935)

        layers = []
        layers = [nn.BatchNorm1d(3935)]
        for w1, w2 in zip(widths[:-1], widths[1:]):
            layers.append(nn.Linear(w1, w2))
            layers.append(self.activation)
        self.layers = nn.Sequential(*layers[:-1])

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) >= 2, "Must be batched."
        assert x.shape[-1] == self.widths[0], "Dimension of argument must match in non-batch dimension."

        return self.layers(x)
    

class MLP_BN2(nn.Module):
    def __init__(self, widths: list[int], bn_spots: list[int], activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        """ 
            MLP with batch normalization layers inserted before activation of layers specified in ``bn_spots``.
            Batch normalization happens over the last dimension also for third-order tensors.
        """
        super().__init__()

        self.widths = widths
        self.activation = activation

        layers = []
        for i, (w1, w2) in enumerate(zip(widths[:-1], widths[1:])):
            if i in bn_spots:
                layers.append(nn.Linear(w1, w2, bias=False))
                layers.append(BNWrap(num_features=w2))
                layers.append(self.activation)
            else:
                layers.append(nn.Linear(w1, w2, bias=True))
                layers.append(self.activation)
        self.layers = nn.Sequential(*layers[:-1])

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class SpatialEncoder(nn.Linear):
        """
            A variant of ``nn.Linear`` that only affects the first features of inputs.
        """
        def forward(self, x):
            return torch.cat([super().forward(x[...,:self.in_features]), x[...,self.in_features:]], dim=-1)
