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
    
    def forward(self, *args, **kwargs):
        return self.x
    

class Context:

    def __init__(self, network: nn.Module, cost_function: Callable, optimizer: torch.optim.Optimizer):

        self.network = network
        self.cost_function = cost_function
        self.optimizer = optimizer

        self.epoch: int = 0
        self.train_hist: list[float] = []
        self.test_hist: dict[int, float] = {}

        return
    
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

