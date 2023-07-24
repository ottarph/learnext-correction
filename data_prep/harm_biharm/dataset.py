import numpy as np
import torch
from torch.utils.data import Dataset

from typing import Iterable, Callable

class learnextDataset(Dataset):

    def __init__(self, prefix: str, checkpoints: Iterable[int],
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
        harm_arr = np.load(self.prefix+f".harmonic.{index:04}.npy")
        biharm_arr = np.load(self.prefix+f".biharmonic.{index:04}.npy")

        harm = torch.tensor(harm_arr, dtype=torch.get_default_dtype())
        biharm = torch.tensor(biharm_arr, dtype=torch.get_default_dtype())

        if self.transform is not None:
            harm = self.transform(harm)
        if self.target_transform is not None:
            biharm = self.target_transform(biharm)

        return harm, biharm
    