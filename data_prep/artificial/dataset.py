import torch
from torch.utils.data import DataLoader, Dataset

from typing import Iterable

class ArtificialLearnextDataset(Dataset):

    def __init__(self, prefix: str, checkpoints1: Iterable, checkpoints2: Iterable, checkpoints3: Iterable):
        super().__init__()

        self.checkpoints1 = checkpoints1
        self.checkpoints2 = checkpoints2
        self.checkpoints3 = checkpoints3

        self.change_12 = len(self.checkpoints1)
        self.change_23 = len(self.checkpoints1) + len(self.checkpoints2)
        self.num_checkpoints = self.__len__()

        return
    
    def __len__(self) -> int:

        return len(self.checkpoints1) + len(self.checkpoints2) + len(self.checkpoints3)
    
    def __getitem__(self, index) -> torch.Tensor:

        if 0 <= index < self.change_12:
            x = None
            y = None
        elif self.change_12 <= index < self.change23:
            x = None
            y = None
        elif self.change_23 < self.num_checkpoints:
            x = None
            y = None

        return x, y