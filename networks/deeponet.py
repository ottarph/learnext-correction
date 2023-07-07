import numpy as np
import torch
import torch.nn as nn

from networks.general import MLP

class BranchNetwork(nn.Module):

    def __init__(self, net: nn.Module, sensors: torch.Tensor,
                 input_domain_dim: int, input_range_dim: int, 
                 output_domain_dim: int, output_range_dim: int, width: int):
        super().__init__()

        self.input_domain_dim = input_domain_dim
        self.input_range_dim = input_range_dim
        self.output_domain_dim = output_domain_dim
        self.output_range_dim = output_range_dim

        self.width = width
        """ The number of features that are combined for output. """

        self.net = net
        self.sensors = sensors
        assert len(sensors.shape) == 2
        assert sensors.shape[1] == input_domain_dim

        if isinstance(net, MLP):
            assert net.layers[0].in_features == sensors.shape[0]*input_range_dim
            assert net.layers[-1].out_features == output_range_dim*width

        return
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
            Takes inputs of shape
                ``(function_batch_dim, sensors_dim, range_dim)``
            Returns output of shape
                ``(function_batch_dim, 1, range_dim, width)``

            Flattens inputs to one vector for each function in batch, then 
            `net` produces `range_dim*width` features for each function
            in batch and the result is reshaped for reduction in `DeepONet`.

            Note: Maybe the logic on flattening and reshaping should be 
            left to other classes, not necessarily true that u_x should
            be mixed with and u_y immediately, or that G(u)_x should be 
            built from the same features as G(u)_y.
        """
        assert len(u.shape) == 3
        assert u.shape[1] == self.sensors.shape[0]
        assert u.shape[2] == self.input_range_dim

        """ Flatten the range_dim dimension. """
        u_flat = u.flatten(start_dim=1)

        out_flat = self.net(u_flat)

        assert out_flat.shape == (u.shape[0], self.output_range_dim*self.width)

        out = out_flat.reshape(u.shape[0], 1, self.output_range_dim, self.width)

        return out
    
class TrunkNetwork(nn.Module):

    def __init__(self, net: nn.Module,
                 input_domain_dim: int, input_range_dim: int, 
                 output_domain_dim: int, output_range_dim: int, width: int):
        super().__init__()

        self.input_domain_dim = input_domain_dim
        self.input_range_dim = input_range_dim
        self.output_domain_dim = output_domain_dim
        self.output_range_dim = output_range_dim

        self.width = width
        """ The number of features that are combined for output. """

        self.net = net

        if isinstance(net, MLP):
            assert net.layers[0].in_features == output_domain_dim
            assert net.layers[-1].out_features == output_range_dim*width

        return
    
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
            Takes inputs of shape
                ``(function_batch_dim, evaluations_dim, range_dim)``
            Returns output of shape
                ``(function_batch_dim, evaluations_dim, range_dim, width)``

            Flattens inputs to one vector for each function-evaluation pair
            in batch, then `net` produces `range_dim*width` features for each 
            function-evaluation pair in batch and the result is reshaped for 
            reduction in `DeepONet`.

            Note: Maybe the logic on flattening and reshaping should be 
            left to other classes, not necessarily true that u_x should
            be mixed with and u_y immediately, or that G(u)_x should be 
            built from the same features as G(u)_y.
        """
        assert len(y.shape) == 3
        assert y.shape[2] == self.output_domain_dim

        out_flat = self.net(y)

        assert out_flat.shape == (y.shape[0], y.shape[1], self.output_range_dim*self.width)

        out = out_flat.reshape(y.shape[0], y.shape[1], self.output_range_dim, self.width)

        return out


class DeepONet(nn.Module):
    """
        Am i considering the right structure?
            https://arxiv.org/pdf/2202.08942.pdf
    """

    def __init__(self, branch: BranchNetwork, trunk: TrunkNetwork,
                 sensors: torch.Tensor, final_bias: bool = False):
        super().__init__()
        
        self.branch = branch
        self.trunk = trunk
        # assert (branch.width,  == trunk.width
        b_dims = (branch.input_domain_dim, branch.input_range_dim, 
                  branch.output_domain_dim, branch.output_range_dim, branch.width)
        t_dims = (trunk.input_domain_dim, trunk.input_range_dim, 
                  trunk.output_domain_dim, trunk.output_range_dim, trunk.width)
        assert b_dims == t_dims

        self.input_domain_dim = branch.input_domain_dim
        self.input_range_dim = branch.input_range_dim
        self.output_domain_dim = branch.output_domain_dim
        self.output_range_dim = branch.output_range_dim

        self.sensors = sensors
        assert len(sensors.shape) == 2, "Sensors tensor must have shape (num_sensors, input_domain_dim)"
        assert torch.equal(sensors, branch.sensors)

        self.final_bias = final_bias
        if final_bias:
            layer = nn.Linear(1, self.output_range_dim)
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
