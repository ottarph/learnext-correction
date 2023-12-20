import numpy as np
import torch
import torch.nn as nn

from networks.general import MLP

widths_array = [
    [8] + [7633]*1 + [2],
    [8] + [284]*2 + [2],
    [8] + [202]*3 + [2],
    [8] + [165]*4 + [2],
    [8] + [143]*5 + [2],
    [8] + [128]*6 + [2]
]

mlps = [MLP(widths, activation=nn.ReLU()) for widths in widths_array]

mlp_param_counts = [sum(map(torch.numel, mlp.parameters())) for mlp in mlps]
print(mlp_param_counts)
