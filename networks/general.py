import numpy as np
import torch
import torch.nn as nn

from typing import Callable, Iterable


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

        self.train_hist: list[float] = []
        self.test_hist: list[float] = []
        self.epoch: int = 0

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) >= 2, "Must be batched."
        assert x.shape[-1] == self.widths[0], "Dimension of argument must match in non-batch dimension."

        return self.layers(x)
    

class TensorModule(nn.Module):

    def __init__(self, x: torch.Tensor):
        super().__init__()

        self.x = x.detach().clone()
        self.x.requires_grad_(False)

        return
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.x
    
    
class TrimModule(nn.Module):

    def __init__(self, forward_indices: Iterable[range]):
        """
            A module whose `.forward(x)`-call returns `x`, but with the last
            dimensions selected according to `forward_shape`.
        """
        super().__init__()

        self.forward_indices = forward_indices
        if len(forward_indices) == 0:
            raise ValueError()
        if len(forward_indices) > 2:
            raise NotImplementedError()

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        view = x[...,self.forward_indices[-1]]
        if len(self.forward_indices) == 2:
            view = view[...,self.forward_indices[-2],:]
            # return view2
        return view


class PrependModule(nn.Module):

    def __init__(self, prepend_tensor: torch.Tensor):
        """
            Inserts `prepend_tensor` as the first elements of the last dimension.

            If `prepend_tensor` is `[a, b]` (shape=(N,2)) and you call the module on the tensor
            `[x, y, z, w]` (shape=(N, 4)), then the output is `[a, b, x, y, z, w]` (shape=(N,6)).
        """
        super().__init__()

        self.prepend_tensor = prepend_tensor

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        new_dims_shape = [1] * ( len(x.shape)-len(self.prepend_tensor.shape) ) + [*self.prepend_tensor.shape]
        prep_tens = self.prepend_tensor.reshape(new_dims_shape)

        expand_shape = [*x.shape[:-1]] + [self.prepend_tensor.shape[-1]]
        prep_tens = prep_tens.expand(expand_shape)

        out = torch.cat((prep_tens, x), dim=-1)

        return out


class Context:

    def __init__(self, network: nn.Module, cost_function: Callable, optimizer: torch.optim.Optimizer):

        self.network = network
        self.cost_function = cost_function
        self.optimizer = optimizer

        self.epoch: int = 0
        self.train_hist: list[float] = []
        self.test_hist: dict[int, float] = {}

        return
    
    def __repr__(self) -> str:

        return f"{self.network} \n{self.cost_function} \n{self.optimizer}"
    
    def save(self, fname: str) -> None:

        data_train = np.array(self.train_hist)
        data_test = np.zeros((2, len(self.test_hist.values())))
        data_test[1,:] = np.array(list(self.test_hist.values()))
        data_test[0,:] = np.array(list(self.test_hist.keys()))

        np.savetxt(fname+".train.txt", data_train)
        np.savetxt(fname+".test.txt", data_test)

        torch.save(self.network.state_dict(), fname+".pt")

        return
    
    def load(self, fname: str) -> None:

        data_train = np.loadtxt(fname+".train.txt")
        data_test = np.loadtxt(fname+".test.txt")

        self.epoch = data_train.shape[0]
        self.train_hist = list(data_train)
        if data_test.shape == (0,):
            self.test_hist = {}
        else:
            self.test_hist = {int(i): l for i, l in zip(data_test[0,:], data_test[1,:])}

        self.network.load_state_dict(torch.load(fname+".pt"))

        return


def train_network_step(context: Context, x: torch.Tensor, y: torch.Tensor, callback: Callable[[Context], None] | None) -> None:
    network = context.network

    def closure():
        context.optimizer.zero_grad()
        cost = context.cost_function(network(x), y)
        cost.backward()
        return cost
    
    cost = context.optimizer.step(closure)

    context.train_hist.append(cost.item())
    context.epoch += 1

    if callback is not None:
        callback(context)

    return

