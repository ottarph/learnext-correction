import numpy as np
import torch
from torch.utils.data import Dataset

from typing import Iterable

class learnextDataset(Dataset):

    def __init__(self, prefix: str, checkpoints: Iterable[int]):

        self.prefix = prefix
        self.checkpoints = checkpoints

        return

    def __len__(self) -> int:
        return len(self.checkpoints)
    
    def __getitem__(self, index) -> torch.Tensor:
        harm_arr = np.load(self.prefix+f".harmonic.{index:04}.npy")
        biharm_arr = np.load(self.prefix+f".biharmonic.{index:04}.npy")

        harm = torch.tensor(harm_arr, dtype=torch.get_default_dtype())
        biharm = torch.tensor(biharm_arr, dtype=torch.get_default_dtype())

        return harm, biharm
    