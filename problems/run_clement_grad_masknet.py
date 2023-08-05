import numpy as np
import dolfin as df
import torch
import torch.nn as nn
import pathlib

from torch.utils.data import DataLoader


from networks.masknet import MaskNet
from networks.general import MLP, TensorModule, TrimModule, PrependModule
from networks.training import Context, train_with_dataloader


def main():
    torch.set_default_dtype(torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{device = }")

    from timeit import default_timer as timer
    from conf import mesh_file_loc, with_submesh

    torch.manual_seed(1)

    from tools.loading import load_mesh
    fluid_mesh = load_mesh(mesh_file_loc, with_submesh)

    V_scal = df.FunctionSpace(fluid_mesh, "CG", 1) # Linear scalar polynomials over triangular mesh
    V = df.VectorFunctionSpace(fluid_mesh, "CG", 1, 2)

    from data_prep.transforms import DofPermutationTransform
    from conf import submesh_conversion_cg1_loc
    perm_tens = torch.LongTensor(np.load(submesh_conversion_cg1_loc))
    dof_perm_transform = DofPermutationTransform(perm_tens, dim=-2)
    transform = dof_perm_transform if with_submesh else None
    print(f"{with_submesh = }")

    from data_prep.clement.dataset import learnextClementGradDataset
    from conf import train_checkpoints, validation_checkpoints, test_checkpoints
    prefix = "data_prep/clement/data_store/grad/clm_grad"
    dataset = learnextClementGradDataset(prefix=prefix, checkpoints=train_checkpoints,
                                         transform=transform, target_transform=transform)
    val_dataset = learnextClementGradDataset(prefix=prefix, checkpoints=validation_checkpoints,
                                             transform=transform, target_transform=transform)
    test_dataset = learnextClementGradDataset(prefix=prefix, checkpoints=test_checkpoints,
                                         transform=transform, target_transform=transform)
    
    # batch_size = 16
    batch_size = 128
    shuffle = True
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    from conf import poisson_mask_f
    from networks.masknet import poisson_mask_custom
    mask_df = poisson_mask_custom(V_scal, poisson_mask_f, normalize = True)

    mask_tensor = torch.tensor(mask_df.vector().get_local(), dtype=torch.get_default_dtype())
    mask = TensorModule(mask_tensor)


    indices = torch.LongTensor(range(2))
    base = TrimModule(indices, dim=-1)
    # base returns (u_x, u_y) from (u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)

    widths = [8] + [128] * 6 + [2]
    mlp = MLP(widths, activation=nn.ReLU())
    # MLP takes input (x, y, u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)

    dof_coordinates = torch.tensor(V_scal.tabulate_dof_coordinates(), dtype=torch.get_default_dtype())
    prepend = PrependModule(dof_coordinates)
    # prepend inserts (x, y) to beginning of (u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)
    # to make correct input of MLP.

    x_mean = torch.tensor([ 8.4319e-01,  2.0462e-01, -1.5600e-03,  4.7358e-04, -5.4384e-03,
            9.0626e-04,  1.3179e-03,  7.8762e-04])
    x_std = torch.tensor([0.6965, 0.1011, 0.0035, 0.0134, 0.0425, 0.0468, 0.1392, 0.1484])
    from networks.general import Normalizer
    normalizer = Normalizer(x_mean, x_std)
    # normalizer changes transforms input to have mean zero and variance one in all dimensions.

    network = nn.Sequential(prepend, normalizer, mlp)


    mask_net = MaskNet(network, base, mask)
    mask_net.to(device)
    

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


    # cost_function = nn.MSELoss()
    cost_function = nn.L1Loss()
    # optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-2)
    # optimizer = torch.optim.Adam(mlp.parameters()) # Good batch size: 512 on GPU, 1024 on CPU with [8, 128, 2]
    optimizer = torch.optim.AdamW(mlp.parameters(), weight_decay=1e-2)
    # optimizer = torch.optim.LBFGS(mlp.parameters(), line_search_fn="strong_wolfe") # Good batch size: 256 on GPU, 16 on CPU with [8, 128, 2]

    # val_cost_function = nn.MSELoss()
    val_cost_function = nn.L1Loss()

    # scheduler = None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
    #                     torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=20),
    #                     torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    #                     ], milestones=[20])


    context = Context(mask_net, cost_function, optimizer, scheduler, val_cost_function)

    # print(context, "\n")
    def callback(context: Context) -> None:
        if context.epoch % 40 == 0:
            context.plot_results("results/latest")

    num_epochs = 500

    start = timer()
    train_with_dataloader(context, dataloader, num_epochs, device, val_dataloader=val_dataloader, callback=callback)
    end = timer()

    print(f"T = {(end - start):.2f} s")

    mask_net.eval()

    run_name = "yankee"

    results_dir = f"results/clem_grad/{run_name}"
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)


    context.save_results(results_dir)
    context.save_summary(results_dir)
    context.plot_results(results_dir)

    latest_results_dir = "results/latest"
    context.save_results(latest_results_dir)
    context.save_summary(latest_results_dir)
    context.plot_results(latest_results_dir)


    model_dir = f"models/clem_grad/{run_name}"
    context.save_model(model_dir)


    xdmf_dir = "fenics_output"
    xdmf_name = f"pred_clem_grad_{run_name}"
    from tools.saving import save_extensions_to_xdmf
    mask_net.to("cpu")
    save_extensions_to_xdmf(mask_net, test_dataloader, V, xdmf_name,
                            save_dir=xdmf_dir, start_checkpoint=test_checkpoints[0])

    return


if __name__ == "__main__":
    main()
