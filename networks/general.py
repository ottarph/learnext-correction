import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from typing import Callable, Iterable, Literal


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

        self.x = nn.Parameter(x.detach().clone())
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


class Context:

    def __init__(self, network: nn.Module, cost_function: Callable, optimizer: torch.optim.Optimizer,
                 scheduler: Callable[[torch.Tensor | None], None] | None = None):

        self.network = network
        self.cost_function = cost_function
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.epoch: int = 0
        self.train_hist: list[float] = []
        self.lr_hist: list[float] = []
        self.test_hist: dict[int, float] = {}

        return
    
    def __repr__(self) -> str:

        return f"Network: {self.network} \nCost function: {self.cost_function}" + \
               f"\nOptimizer: {self.optimizer} \nScheduler: {self.scheduler}"
    
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


def train_with_dataloader(context: Context, dataloader: DataLoader, num_epochs: int,
                          device: Literal["cuda", "cpu"],
                          callback: Callable[[Context], None] | None = None):

    network = context.network
    cost_function = context.cost_function
    optimizer = context.optimizer
    scheduler = context.scheduler

    lr = optimizer.param_groups[0]["lr"]

    from tqdm import tqdm

    epoch_loop = tqdm(range(1, num_epochs+1), position=0, desc=f"Epoch #000, loss =  ???   , lr = {lr:.1e}")
    for epoch in epoch_loop:
        epoch_loss = 0.0

        dataloader_loop = tqdm(dataloader, desc="Mini-batch #000", position=1, leave=False)
        for mb, (x, y) in enumerate(dataloader_loop, start=1):
            x, y = x.to(device), y.to(device)

            def closure():
                optimizer.zero_grad()
                loss = cost_function(network(x), y)
                loss.backward()
                return loss
            
            loss = optimizer.step(closure)
            epoch_loss += loss.item()

            dataloader_loop.set_description_str(f"Mini-batch #{mb:03}")

        context.epoch += 1
        context.train_hist.append(epoch_loss)
        context.lr_hist.append(optimizer.param_groups[0]["lr"])

        if scheduler is not None:
            try:
                scheduler.step()
            except:
                scheduler.step(loss)


        epoch_loop.set_description_str(f"Epoch #{epoch:03}, loss = {epoch_loss:.2e}, lr = {lr:.1e}")

        if callback is not None:
            callback(context)

    return

