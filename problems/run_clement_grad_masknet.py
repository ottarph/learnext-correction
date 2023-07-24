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
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    # base returns (u_x, u_y) from (u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)

    widths = [8, 128, 2]
    # widths = [8, 512, 2]
    mlp = MLP(widths, activation=nn.ReLU())
    # MLP takes input (x, y, u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)

    dof_coordinates = V_scal.tabulate_dof_coordinates()
    prepend = PrependModule(torch.tensor(dof_coordinates, dtype=torch.get_default_dtype()))
    # Prepend inserts (x, y) to beginning of (u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)
    # to make correct input of MLP.
    network = nn.Sequential(prepend, mlp)


    mask_net = MaskNet(network, base, mask)
    mask_net.to(device)

    from data_prep.transforms import DofPermutationTransform
    from conf import submesh_conversion_cg1_loc
    perm_tens = torch.LongTensor(np.load(submesh_conversion_cg1_loc))
    dof_perm_transform = DofPermutationTransform(perm_tens, dim=-2)
    transform = dof_perm_transform if with_submesh else None
    print(f"{with_submesh = }")

    from data_prep.clement.dataset import learnextClementGradDataset
    from conf import train_checkpoints
    prefix = "data_prep/clement/data_store/grad/clm_grad"
    dataset = learnextClementGradDataset(prefix=prefix, checkpoints=train_checkpoints,
                                         transform=transform, target_transform=transform)
    
    
    # batch_size = 16
    batch_size = 64
    shuffle = True
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    x, y = next(iter(dataloader))
    x, y = x.to(device), y.to(device)
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
    # cost_function = nn.L1Loss()
    # optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-2)
    optimizer = torch.optim.Adam(mlp.parameters()) # Good batch size: 512 on GPU, 1024 on CPU with [8, 128, 2]
    # optimizer = torch.optim.LBFGS(mlp.parameters(), line_search_fn="strong_wolfe") # Good batch size: 256 on GPU, 16 on CPU with [8, 128, 2]

    # optimizer = torch.optim.Adam(mlp.parameters(), weight_decay=1e-4) # Need to wait until I have test metrics for this

    scheduler = None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    context = Context(mask_net, cost_function, optimizer, scheduler)

    print(context)

    num_epochs = 20

    start = timer()

    train_with_dataloader(context, dataloader, num_epochs, device)

    end = timer()

    print(f"{batch_size=}")
    # print(f"{widths=}")
    print(f"T = {(end - start):.2f} s")

    print(context.train_hist)

    # save_pref = "mse_lbfgs_8_128_2_clm_grad"
    save_pref = "mse_adam_8_128_2_clm_grad"

    # context.save(f"models/{save_pref}")
    


    # plt.figure()
    # plt.plot(range(context.epoch), context.train_hist, 'k-')
    # plt.semilogy(range(context.epoch), context.train_hist, 'k-')
    # plt.savefig(f"foo/clm/{save_pref}_train_hist.png", dpi=150)

    # plt.figure()
    # plt.semilogy(range(context.epoch), context.lr_hist, 'k-')
    # plt.savefig(f"foo/clm/{save_pref}_lr_hist.png", dpi=150)

    return


if __name__ == "__main__":
    main()
