import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from torch.utils.data import DataLoader

from typing import Callable, Literal
from matplotlib.figure import Figure

LR_Scheduler = torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau

class Context:

    def __init__(self, network: nn.Module, 
                 cost_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
                 optimizer: torch.optim.Optimizer,
                 scheduler: LR_Scheduler | None = None,
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
        pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

        data_train = np.array(self.train_hist)
        data_val = np.array(self.val_hist)
        data_lr = np.array(self.lr_hist)

        np.savetxt(folder_name+"/train.txt", data_train)
        np.savetxt(folder_name+"/val.txt", data_val)
        np.savetxt(folder_name+"/lr.txt", data_lr)

        return
    
    def save_model(self, folder_name: str) -> None:
        pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

        pathlib.Path(folder_name+"/model.txt").write_text(str(self.network))
        torch.save(self.network.state_dict(), folder_name+"/state_dict.pt")

        return
    
    def save_summary(self, folder_name: str, file_name: str = "context"):
        pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)

        text = f"Network: {self.network} \nCost function: {self.cost_function}" + \
               f"\nValidation Cost function: {self.validation_cost_function}" + \
               f"\nOptimizer: {self.optimizer}"
        
        text += f"\nScheduler: {self.scheduler.__class__.__name__}: " + \
                f"\n\t{self.scheduler.state_dict()}"
        
        text += f"\nFinal train loss: {self.final_train_loss}" + \
                f"\nFinal val loss: {self.final_val_loss}" + \
                f"\nFinal lr: {self.final_lr}"
        
        pathlib.Path(f"{folder_name}/{file_name}.txt").write_text(text)

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
    
    def plot_results(self, folder_name: str | None = None) -> Figure:
        """ Adapted from https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html """
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.semilogy(range(self.epoch), self.train_hist, 'k-', label="Train")
        ax1.semilogy(range(self.epoch), self.val_hist, 'r--', label="Val")
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.set_ylabel('lr')
        ax2.semilogy(range(self.epoch), self.lr_hist, 'b:', alpha=0.8, lw=0.75, label="lr")
        ax2.tick_params(axis='y')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.legend()

        if folder_name is not None:
            plt.savefig(f"{folder_name}/train_val_lr_hist.png", dpi=150)

        return fig


def train_with_dataloader(context: Context, train_dataloader: DataLoader, 
                          num_epochs: int, device: Literal["cuda", "cpu"],
                          val_dataloader: DataLoader | None = None, 
                          callback: Callable[[Context], None] | None = None,
                          break_at_lr: float = 1e-8):

    network = context.network
    cost_function = context.cost_function
    optimizer = context.optimizer
    scheduler = context.scheduler
    validation_cost_function = context.validation_cost_function

    lr = optimizer.param_groups[0]["lr"]

    from tqdm import tqdm

    network.train()
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
            network.eval()
            val_loss = 0.0
            val_dataloader_loop = tqdm(val_dataloader, position=1, desc="Running over validation data set.", leave=False)
            with torch.no_grad():
                for x, y in val_dataloader_loop:
                    x, y = x.to(device), y.to(device)
                    val_loss += validation_cost_function(network(x), y).item()
            context.val_hist.append(val_loss)
            network.train()

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if val_dataloader is not None:
                    scheduler.step(val_loss)
                else:
                    scheduler.step(epoch_loss)
            else:
                scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            if lr < break_at_lr:
                break
        
        print_loss = val_loss if val_dataloader is not None else epoch_loss
        epoch_loop.set_description_str(f"Epoch #{epoch:03}, loss = {print_loss:.2e}, lr = {lr:.1e}")

        if callback is not None:
            callback(context)

    return

