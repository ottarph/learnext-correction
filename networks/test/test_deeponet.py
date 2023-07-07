import numpy as np
import torch
import torch.nn as nn

from networks.deeponet import *

def test_reduction():

    torch.manual_seed(0)

    # We need
    #    out.shape            = (Function batch dim, Num evaluations dim, Output range dim),
    # so, must have
    #    branch_weights.shape = (Function batch dim,          1         , Output range dim, width),
    #    trunk_weights.shape  = (Function batch dim, Num evaluations dim, Output range dim, width),
    # to broadcast correctly.

    func_batch = 2
    num_eval = 2
    output_range = 3
    width = 4

    bw = torch.rand((func_batch, 1, output_range, width))
    tw = torch.rand((func_batch, num_eval, output_range, width))

    out1 = torch.einsum("...i,...i->...", bw, tw)
    
    out2 = torch.zeros((func_batch, num_eval, output_range))
    for i in range(func_batch):
        for j in range(num_eval):
            for k in range(output_range):
                for l in range(width):
                    out2[i, j, k] += bw[i, 0, k, l] * tw[i, j, k, l]


    assert out1.shape == out2.shape
    assert torch.equal(out1, out2)

    return


def test_branch():

    from networks.general import MLP

    input_domain_dim = 3
    input_range_dim = 2
    output_domain_dim = 2
    output_range_dim = 2
    function_batch = 2
    num_sens = 3
    width = 4
    net_widths = [input_range_dim*num_sens, 8, output_range_dim*width]

    sensors = torch.rand((num_sens, input_domain_dim))

    net = MLP(net_widths, nn.ReLU())
    branch_net = BranchNetwork(net, sensors, input_domain_dim, input_range_dim, width)

    def f(x):
        return torch.column_stack([2*x[...,0], -x[...,1]])
    u1 = f(sensors)
    u2 = -u1
    u = torch.zeros((2, *u1.shape))
    u[0,...] = u1
    u[1,...] = u2

    assert u.shape == (function_batch, num_sens, input_range_dim)
    bw = branch_net(u)

    assert bw.shape == (function_batch, 1, output_range_dim, width)

    return


if __name__ == "__main__":
    test_reduction()
    test_branch()
