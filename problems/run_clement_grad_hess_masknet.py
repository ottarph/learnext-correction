import numpy as np
import dolfin as df
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from networks.masknet import MaskNet
from networks.general import MLP, TensorModule, TrimModule, PrependModule, \
                             Context, train_with_dataloader


def main():
    # torch.set_default_dtype(torch.float64)
    torch.set_default_dtype(torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{device = }")
    
    from timeit import default_timer as timer
    from conf import mesh_file_loc, with_submesh

    torch.manual_seed(0)

    from tools.loading import load_mesh
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
    mask_net.to(device)

    from data_prep.transforms import DofPermutationTransform
    from conf import submesh_conversion_cg1_loc
    perm_tens = torch.LongTensor(np.load(submesh_conversion_cg1_loc))
    dof_perm_transform = DofPermutationTransform(perm_tens, dim=-2)
    transform = dof_perm_transform if with_submesh else None
    print(f"{with_submesh = }")

    from data_prep.clement.dataset import learnextClementGradHessDataset
    from conf import train_checkpoints
    prefix = "data_prep/clement/data_store/grad_hess/clm_grad_hess"
    dataset = learnextClementGradHessDataset(prefix=prefix, checkpoints=train_checkpoints,
                                             transform=transform, target_transform=transform)
    
    
    batch_size = 256
    shuffle = True
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    x, y = next(iter(dataloader))
    x, y = x.to(device), y.to(device)
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
    # optimizer = torch.optim.Adam(mlp.parameters()) # Good batch size: 512 on GPU, 1024 on CPU
    optimizer = torch.optim.LBFGS(mlp.parameters(), line_search_fn="strong_wolfe") # Good batch size: 256 on GPU, 16 on CPU

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scheduler = None

    context = Context(mask_net, cost_function, optimizer, scheduler)

    print(context)


    num_epochs = 10

    start = timer()

    train_with_dataloader(context, dataloader, num_epochs, device)

    end = timer()

    print(f"{batch_size=}")
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
