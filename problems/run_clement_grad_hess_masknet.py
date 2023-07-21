import numpy as np
import dolfin as df
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from tools.loading import *
from tools.plots import *

from networks.masknet import *
from networks.general import *


def main():
    # torch.set_default_dtype(torch.float64)
    torch.set_default_dtype(torch.float32)
    
    from timeit import default_timer as timer
    from conf import mesh_file_loc, with_submesh

    torch.manual_seed(0)

    fluid_mesh = load_mesh(mesh_file_loc, with_submesh)

    V_scal = df.FunctionSpace(fluid_mesh, "CG", 1) # Linear scalar polynomials over triangular mesh

    # mask_df = poisson_mask(V_scal, normalize = True)
    from conf import poisson_mask_f
    from networks.masknet import poisson_mask_custom
    mask_df = poisson_mask_custom(V_scal, poisson_mask_f, normalize = True)

    mask_tensor = torch.tensor(mask_df.vector().get_local(), dtype=torch.get_default_dtype())
    mask = TensorModule(mask_tensor)


    forward_indices = [range(2)]
    base = TrimModule(forward_indices=forward_indices)
    # base returns (u_x, u_y) from (u_x, u_y, d_x u_x,  d_y u_x,  d_x u_y,  d_y u_y,
    #                                        d_xx u_x, d_xy u_x, d_yx u_x, d_yy u_x,
    #                                        d_xx u_y, d_xy u_y, d_yx u_y, d_yy u_y)

    widths = [16, 128, 2]
    mlp = MLP(widths, activation=nn.ReLU())
    # MLP takes input        (       x,        y,      u_x,      u_y, 
    #                          d_x u_x,  d_y u_x,  d_x u_y,  d_y u_y,
    #                         d_xx u_x, d_xy u_x, d_yx u_x, d_yy u_x,
    #                         d_xx u_y, d_xy u_y, d_yx u_y, d_yy u_y)

    dof_coordinates = V_scal.tabulate_dof_coordinates()
    prepend = PrependModule(torch.tensor(dof_coordinates, dtype=torch.get_default_dtype()))
    # Prepend inserts (x, y) to beginning of (u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)
    # to make correct input of MLP.
    network = nn.Sequential(prepend, mlp)


    mask_net = MaskNet(network, base, mask)


    from data_prep.clement.dataset import learnextClementGradHessDataset
    from conf import train_checkpoints
    prefix = "data_prep/clement/data_store/grad_hess/clm_grad_hess"
    dataset = learnextClementGradHessDataset(prefix=prefix, checkpoints=train_checkpoints)
    
    
    batch_size = 128
    shuffle = True
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    x, y = next(iter(dataloader))
    assert x.shape == (batch_size, fluid_mesh.num_vertices(), 14)
    # In:   (u_x, u_y, d_x u_x,  d_y u_x,  d_x u_y,  d_y u_y,
    #                 d_xx u_x, d_xy u_x, d_yx u_x, d_yy u_x,
    #                 d_xx u_y, d_xy u_y, d_yx u_y, d_yy u_y)
    # where u is from harmonic extension
    assert y.shape == (batch_size, fluid_mesh.num_vertices(), 2)
    # Out: (u_x, u_y), where u is from biharmonic extension

    assert mask(x).shape == (fluid_mesh.num_vertices(),)
    assert base(x).shape == (batch_size, fluid_mesh.num_vertices(), 2)
    assert network(x).shape == (batch_size, fluid_mesh.num_vertices(), 2)
    assert mask_net(x).shape == (batch_size, fluid_mesh.num_vertices(), 2)
    assert (mask_net(x) - y).shape == (batch_size, fluid_mesh.num_vertices(), 2)

    print("Pre-run assertions passed. \n")


    cost_function = nn.MSELoss()
    # optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-1)
    optimizer = torch.optim.Adam(mlp.parameters()) # Good batch size: 1024?
    # optimizer = torch.optim.LBFGS(mlp.parameters(), line_search_fn="strong_wolfe") # Good batch size: 16?


    context = Context(mask_net, cost_function, optimizer)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    scheduler = None

    print(context)

    def callback(context: Context) -> None:
        print(f"epoch #{context.epoch-1:03}, loss = {context.train_hist[-1]:.2e}")
        return
    callback = None

    num_epochs = 20

    start = timer()

    train_with_dataloader(context, dataloader, num_epochs, scheduler=scheduler, callback=callback)

    end = timer()

    # print(f"{batch_size=}")
    # print(f"{widths=}")
    print(f"T = {(end - start):.2f} s")
    print(context.train_hist)

    # context.save("models/LBFGS_16_128_2_clm_grad_hess")


    # plt.figure()
    # # plt.plot(range(context.epoch), context.train_hist, 'k-')
    # plt.semilogy(range(context.epoch), context.train_hist, 'k-')
    # plt.savefig("foo/clm_grad_hess/LBFGS_16_128_2_clm_grad_hess_train_hist.png", dpi=150)


    return


if __name__ == "__main__":
    main()
