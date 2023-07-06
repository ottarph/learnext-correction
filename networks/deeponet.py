import numpy as np
import torch
import torch.nn as nn


class BranchNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.domain_dim = 2
        self.range_dim = 2

        self.width = 2
        """ The number of features that are combined for output. """

        return
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        assert u.shape[-1] == self.range_dim


        return
    
class TrunkNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.domain_dim = 2
        self.range_dim = 2

        self.width = 2
        """ The number of features that are combined for output. """

        return
    
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        assert y.shape[-1] == self.domain_dim

        return


class DeepONet(nn.Module):

    def __init__(self, branch: BranchNetwork, trunk: TrunkNetwork,
                       sensors: torch.Tensor, final_bias: bool = False):
        super().__init__()
        
        self.branch = branch
        self.trunk = trunk
        assert branch.width == trunk.width

        self.sensors = sensors
        assert len(sensors.shape) == 2, "Sensors tensor must have shape (num_sensors, input_domain_dim)"

        self.final_bias = final_bias
        if final_bias:
            layer = nn.Linear(1, self.trunk.range_dim)
            bias_tensor = layer.bias.detach().clone()
            del layer
            bias_tensor.requires_grad_(True)
            self.bias = nn.Parameter(bias_tensor)

        return
    
    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        
            `u`: input function(s) evaluated at sensor locations.\n
            `y`: evaluation location(s) of target function domain.
        
            Input dims:
                `u`:
                    Function batch dim\n
                    Input sensors dim\n
                    Input range dim
                
                `y`:
                    Function batch dim\n
                    Num evaluations dim\n
                    Output domain dim


            Output dims:
                Function batch dim\n
                Num evaluations dim\n
                Output range dims

            Assumes that each output function is evaluated equal number of times in each
            mini-batch. If this is not the case, they should be padded with non-interesting
            tensors, and those results just not be used.

        """

        assert u.shape[0] == y.shape[0]

        assert len(u.shape) == 3
        assert u.shape[1] == self.sensors.shape[0]
        assert u.shape[2] == self.branch.range_dim

        assert len(y.shape) == 3
        assert y.shape[2] == self.trunk.domain_dim

        # We need
        #    out.shape            = (Function batch dim, Num evaluations dim, Output range dim),
        # so, must have
        #    branch_weights.shape = (Function batch dim,          1         , Output range dim, width),
        #    trunk_weights.shape  = (Function batch dim, Num evaluations dim, Output range dim, width),
        # to broadcast correctly.

        branch_weights = self.branch(u)
        trunk_weights = self.trunk(y)

        out = torch.einsum("...i,...i->...", branch_weights, trunk_weights)

        if self.final_bias:
            out += self.bias

        assert len(out.shape) == 3
        assert out.shape[0] == u.shape[0]
        assert out.shape[1] == y.shape[1]
        assert out.shape[2] == self.trunk.range_dim

        return out
