import numpy as np
import torch
import torch.nn as nn

from networks.deeponet import *

torch.manual_seed(0)

test_dimensions = {
    # Making sure the integers are all different, so no checks fail silently
    "input_domain_dim": 5,
    "input_range_dim": 4,
    "output_domain_dim": 3,
    "output_range_dim": 2,
    "function_batch": 8,
    "num_sens": 7,
    "num_eval": 6,
    "width": 4,
    "net_max_width": 10
}

def test_reduction():

    # We need
    #    out.shape            = (Function batch dim, Num evaluations dim, Output range dim),
    # so, must have
    #    branch_weights.shape = (Function batch dim,          1         , Output range dim, width),
    #    trunk_weights.shape  = (Function batch dim, Num evaluations dim, Output range dim, width),
    # to broadcast correctly.

    func_batch = test_dimensions["function_batch"]
    num_eval = test_dimensions["num_eval"]
    output_range = test_dimensions["output_range_dim"]
    width = test_dimensions["width"]

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

    input_domain_dim = test_dimensions["input_domain_dim"]
    input_range_dim = test_dimensions["input_range_dim"]
    output_domain_dim = test_dimensions["output_domain_dim"]
    output_range_dim = test_dimensions["output_range_dim"]
    function_batch = test_dimensions["function_batch"]
    num_sens = test_dimensions["num_sens"]
    width = test_dimensions["width"]
    net_max_width = test_dimensions["net_max_width"]
    net_widths = [input_range_dim*num_sens, net_max_width, output_range_dim*width]

    sensors = torch.rand((num_sens, input_domain_dim))

    net = MLP(net_widths, nn.ReLU())
    branch_net = BranchNetwork(net, sensors, 
                               input_domain_dim, input_range_dim, 
                               output_domain_dim, output_range_dim, width)

    def f(x):
        return torch.column_stack([(i+1)*x[...,0] for i in range(input_range_dim)])
    ui = f(sensors)
    u = torch.zeros((function_batch, *ui.shape))
    for i in range(function_batch):
        u[i,...] = ui * (-1.0)**i

    assert u.shape == (function_batch, num_sens, input_range_dim)
    bw = branch_net(u)

    assert bw.shape == (function_batch, 1, output_range_dim, width)

    return

def test_trunk():

    from networks.general import MLP

    input_domain_dim = 3
    input_domain_dim = test_dimensions["input_domain_dim"]
    input_range_dim = test_dimensions["input_range_dim"]
    output_domain_dim = test_dimensions["output_domain_dim"]
    output_range_dim = test_dimensions["output_range_dim"]
    function_batch = test_dimensions["function_batch"]
    num_eval = test_dimensions["num_eval"]
    width = test_dimensions["width"]
    net_max_width = test_dimensions["net_max_width"]
    net_widths = [output_domain_dim, net_max_width, output_range_dim*width]

    net = MLP(net_widths, nn.ReLU())
    trunk_net = TrunkNetwork(net,
                             input_domain_dim, input_range_dim, 
                             output_domain_dim, output_range_dim, width)

    y = torch.rand((function_batch, num_eval, output_domain_dim))

    tw = trunk_net(y)

    assert tw.shape == (function_batch, num_eval, output_range_dim, width)

    return

if __name__ == "__main__":
    test_reduction()
    test_branch()
    test_trunk()
