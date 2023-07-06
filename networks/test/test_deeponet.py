import numpy as np
import torch
import torch.nn as nn


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

    # out.shape = (2, 2, 3)
    # bw.shape  = (2, 1, 3, 4)
    # tw.shape  = (2, 2, 3, 4)

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


if __name__ == "__main__":
    test_reduction()
