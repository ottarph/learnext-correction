import numpy as np
import dolfin as df
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from tools.loading import *
from tools.plots import *

from networks.masknet import *
from networks.general import *

from typing import Callable


def train_with_dataloader(context: Context, dataloader: DataLoader, num_epochs: int,
                          callback: Callable[[Context], None] | None = None):

    network = context.network
    cost_function = context.cost_function
    optimizer = context.optimizer


    for _ in range(num_epochs):
        epoch_loss = 0.0
        for x, y in dataloader:

            def closure():
                optimizer.zero_grad()
                loss = cost_function(network(x), y)
                loss.backward()
                return loss
            
            loss = optimizer.step(closure)

            epoch_loss += loss.item()

        context.epoch += 1
        context.train_hist.append(epoch_loss)

        if callback is not None:
            callback(context)

    return

def main():
    # torch.set_default_dtype(torch.float64)
    torch.set_default_dtype(torch.float32)
    
    from timeit import default_timer as timer
    from conf import OutputLoc, vandermonde_loc

    torch.manual_seed(0)

    mesh_file_loc = OutputLoc + "/Mesh_Generation"

    _, fluid_mesh, _ = load_mesh(mesh_file_loc)

    V_scal = df.FunctionSpace(fluid_mesh, "CG", 1) # Linear scalar polynomials over triangular mesh

    mask_df = laplace_mask(V_scal)
    mask_tensor = torch.tensor(mask_df.vector().get_local(), dtype=torch.get_default_dtype())
    mask = TensorModule(mask_tensor)


    forward_indices = [range(2)]
    base = TrimModule(forward_indices=forward_indices)
    # base returns (u_x, u_y) from (u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)

    widths = [8, 128, 2]
    mlp = MLP(widths, activation=nn.ReLU())
    # MLP takes input (x, y, u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)

    dof_coordinates = V_scal.tabulate_dof_coordinates()
    prepend = PrependModule(torch.tensor(dof_coordinates, dtype=torch.get_default_dtype()))
    # Prepend inserts (x, y) to beginning of (u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)
    # to make correct input of MLP.
    network = nn.Sequential(prepend, mlp)


    mask_net = MaskNet(network, base, mask)


    from data_prep.clement.dataset import learnextClementDataset
    from conf import train_checkpoints
    prefix = "data_prep/clement/data_store/clm_grad"
    dataset = learnextClementDataset(prefix=prefix, checkpoints=train_checkpoints)
    
    
    batch_size = 1024
    shuffle = True
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    x, y = next(iter(dataloader))
    assert x.shape == (batch_size, fluid_mesh.num_vertices(), 6)
    # In:  (u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y), where u is from harmonic extension
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
    optimizer = torch.optim.Adam(mlp.parameters())
    # optimizer = torch.optim.LBFGS(mlp.parameters(), line_search_fn="strong_wolfe")


    context = Context(network, cost_function, optimizer)

    print(context)

    def callback(context: Context) -> None:
        print(f"epoch #{context.epoch-1:03}, loss = {context.train_hist[-1]:.2e}")
        return
    # callback = None

    num_epochs = 1

    start = timer()

    train_with_dataloader(context, dataloader, num_epochs, callback=callback)

    end = timer()

    print(f"{batch_size=}")
    print(f"T = {(end - start):.2f} s")

    # context.save("models/8_128_2_clm")


    # plt.figure()
    # plt.plot(range(context.epoch), context.train_hist, 'k-')
    # plt.savefig("foo/clm/train_hist.png", dpi=150)


    return


if __name__ == "__main__":
    main()
