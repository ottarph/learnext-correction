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
                 scheduler: Callable[[torch.Tensor | None], None] | None = None,
                 validation_cost_function: Callable[[torch.Tensor], torch.Tensor] | None = None):

        self.network = network
        self.cost_function = cost_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        if validation_cost_function is None:
            validation_cost_function = cost_function
        self.validation_cost_function = validation_cost_function


        self.epoch: int = 0
        self.train_hist: list[float] = []
        self.lr_hist: list[float] = []
        self.val_hist: list[float] = []

        return
    
    def __repr__(self) -> str:

        return f"Network: {self.network} \nCost function: {self.cost_function}" + \
               f"\nValidation Cost function: {self.validation_cost_function}" + \
               f"\nOptimizer: {self.optimizer} \nScheduler: {self.scheduler}" + \
               f"\nFinal train loss: {self.final_train_loss}" + \
               f"\nFinal val loss: {self.final_val_loss}" + \
               f"\nFinal lr: {self.final_lr}"
    
    @property
    def final_train_loss(self):
        if len(self.train_hist) > 0:
            return self.train_hist[-1]
        else:
            return None
        
    @property
    def final_val_loss(self):
        if len(self.val_hist) > 0:
            return self.val_hist[-1]
        else:
            return None
        
    @property
    def final_lr(self):
        if len(self.lr_hist) > 0:
            return self.lr_hist[-1]
        else:
            return None
    
    def save_results(self, folder_name: str) -> None:
        import pathlib
        pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

        data_train = np.array(self.train_hist)
        data_val = np.array(self.val_hist)
        data_lr = np.array(self.lr_hist)

        np.savetxt(folder_name+"/train.txt", data_train)
        np.savetxt(folder_name+"/val.txt", data_val)
        np.savetxt(folder_name+"/lr.txt", data_lr)

        return
    
    def save_model(self, folder_name: str) -> None:
        import pathlib
        pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

        pathlib.Path(folder_name+"/model.txt").write_text(str(self.network))
        torch.save(self.network.state_dict(), folder_name+"/state_dict.pt")

        return
    
    def load_results(self, folder_name: str) -> None:

        data_train = np.loadtxt(folder_name+"/train.txt")
        data_val = np.loadtxt(folder_name+"/val.txt")
        data_lr = np.loadtxt(folder_name+"/lr.txt")

        self.epoch = data_train.shape[0]
        self.train_hist = list(data_train)
        self.val_hist = list(data_val)
        self.lr_hist = list(data_lr)

        return
    
    def load_model(self, folder_name: str) -> None:

        self.network.load_state_dict(torch.load(folder_name+"/state_dict.pt"))

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


def train_with_dataloader(context: Context, train_dataloader: DataLoader, 
                          num_epochs: int, device: Literal["cuda", "cpu"],
                          val_dataloader: DataLoader | None = None, 
                          callback: Callable[[Context], None] | None = None):

    network = context.network
    cost_function = context.cost_function
    optimizer = context.optimizer
    scheduler = context.scheduler
    validation_cost_function = context.validation_cost_function

    lr = optimizer.param_groups[0]["lr"]

    from tqdm import tqdm

    epoch_loop = tqdm(range(1, num_epochs+1), position=0, desc=f"Epoch #000, loss =  ???   , lr = {lr:.1e}")
    for epoch in epoch_loop:
        epoch_loss = 0.0

        train_dataloader_loop = tqdm(train_dataloader, desc="Mini-batch #000", position=1, leave=False)
        for mb, (x, y) in enumerate(train_dataloader_loop, start=1):
            x, y = x.to(device), y.to(device)

            def closure():
                optimizer.zero_grad()
                loss = cost_function(network(x), y)
                loss.backward()
                return loss
            
            loss = optimizer.step(closure)
            epoch_loss += loss.item()

            train_dataloader_loop.set_description_str(f"Mini-batch #{mb:03}")

        context.epoch += 1
        context.train_hist.append(epoch_loss)
        context.lr_hist.append(lr)

        if val_dataloader is not None:
            val_loss = 0.0
            val_dataloader_loop = tqdm(val_dataloader, position=1, desc="Running over validaton data set.", leave=False)
            with torch.no_grad():
                for x, y in val_dataloader_loop:
                    x, y = x.to(device), y.to(device)
                    val_loss += validation_cost_function(network(x), y).item()
            context.val_hist.append(val_loss)

        if scheduler is not None:
            try:
                scheduler.step()
            except:
                if val_dataloader is not None:
                    scheduler.step(val_loss)
                else:
                    scheduler.step(epoch_loss)
            lr = optimizer.param_groups[0]["lr"]
        
        print_loss = val_loss if val_dataloader is not None else epoch_loss
        epoch_loop.set_description_str(f"Epoch #{epoch:03}, loss = {print_loss:.2e}, lr = {lr:.1e}")

        if callback is not None:
            callback(context)

    return

