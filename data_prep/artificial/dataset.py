import torch
import numpy as np
from torch.utils.data import Dataset

from typing import Iterable, Callable

class ArtificialLearnextDataset(Dataset):

    def __init__(self, prefix: str, checkpoints: Iterable, 
                 transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
                 target_transform: Callable[[torch.Tensor], torch.Tensor] | None = None):
        super().__init__()

        self.prefix = prefix
        self.checkpoints = checkpoints
        self.transform = transform
        self.target_transform = target_transform

        return
    
    def __len__(self) -> int:

        return len(self.checkpoints)
    
    def __getitem__(self, index) -> torch.Tensor:

        harm_plus_clm_grad_np = np.load(self.prefix+f".harm_plus_clm_grad.{index:04}.npy")
        biharm_np = np.load(self.prefix+f".biharm.{index:04}.npy")

        x = torch.tensor(harm_plus_clm_grad_np, dtype=torch.get_default_dtype())
        y = torch.tensor(biharm_np, dtype=torch.get_default_dtype())

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y
