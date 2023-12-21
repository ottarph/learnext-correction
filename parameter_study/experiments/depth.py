import numpy as np
import dolfin as df
import torch
import torch.nn as nn
import pathlib
import matplotlib.pyplot as plt

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
    
    batch_size = 128
    shuffle = True
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    from networks.general import Normalizer
    from conf import poisson_mask_f
    from networks.masknet import poisson_mask_custom

    mask_df = poisson_mask_custom(V_scal, poisson_mask_f, normalize = True)

    widths_array = [
        [8] + [284]*2 + [2],    # num. parameters = 84066
        [8] + [202]*3 + [2],    # num. parameters = 84236
        [8] + [165]*4 + [2],    # num. parameters = 83987
        [8] + [143]*5 + [2],    # num. parameters = 83943
        # [8] + [128]*6 + [2]     # num. parameters = 83970
    ]

    num_runs_per_depth = 10
    num_epochs = 500
    # Estimate that 4 depths with 10 runs of 500 epochs with 2s/epoch takes 11h06m40s.
    
    min_mesh_qual_fig_ax = plt.subplots(len(widths_array) // 2 + len(widths_array) % 2, 2, figsize=(12,16))
    mesh_qualities_over_runs = np.zeros((len(widths_array), num_runs_per_depth, len(test_dataloader.dataset), fluid_mesh.num_cells()))

    print()
    for depths_num, widths in enumerate(widths_array):
        print(f"{len(widths)-2} hidden layers, widths = [8] + [{widths[1]}]*{len(widths)-2} + [2]")

        for run in range(num_runs_per_depth):

            torch.manual_seed(run) # Specify seed so that experiment can be reproduced without starting from first run.

            mask_tensor = torch.tensor(mask_df.vector().get_local(), dtype=torch.get_default_dtype())
            mask = TensorModule(mask_tensor)


            indices = torch.LongTensor(range(2))
            base = TrimModule(indices, dim=-1)
            # base returns (u_x, u_y) from (u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)

            mlp = MLP(widths, activation=nn.ReLU())
            # MLP takes input (x, y, u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)

            dof_coordinates = torch.tensor(V_scal.tabulate_dof_coordinates(), dtype=torch.get_default_dtype())
            prepend = PrependModule(dof_coordinates)
            # prepend inserts (x, y) to beginning of (u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)
            # to make correct input of MLP.

            x_mean = torch.tensor([ 8.4319e-01,  2.0462e-01, -1.5600e-03,  4.7358e-04, -5.4384e-03,
                    9.0626e-04,  1.3179e-03,  7.8762e-04])
            x_std = torch.tensor([0.6965, 0.1011, 0.0035, 0.0134, 0.0425, 0.0468, 0.1392, 0.1484])
            normalizer = Normalizer(x_mean, x_std)
            # normalizer changes transforms input to have mean zero and variance one in all dimensions.

            network = nn.Sequential(prepend, normalizer, mlp)


            mask_net = MaskNet(network, base, mask)
            mask_net.to(device)
            

            cost_function = nn.L1Loss()
            optimizer = torch.optim.AdamW(mlp.parameters(), weight_decay=1e-2)

            val_cost_function = nn.L1Loss()

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

            context = Context(mask_net, cost_function, optimizer, scheduler, val_cost_function)

            print(f"Run #{run+1}")
            train_with_dataloader(context, dataloader, num_epochs, device, val_dataloader=val_dataloader, callback=None)


            mask_net.eval()
            mask_net.to("cpu")

            from parameter_study.mesh_quality_eval import compute_mesh_qualities
            mesh_qualities = compute_mesh_qualities(fluid_mesh, test_dataloader, mask_net, 
                                            quality_measure="scaled_jacobian", show_progress_bar=True)
            mesh_qualities_over_runs[depths_num, run, :, :] = mesh_qualities
            mesh_qual_mins = mesh_qualities.min(axis=1)

            fig, axs = min_mesh_qual_fig_ax
            ax = axs[depths_num // 2, depths_num % 2]
            ax.plot(range(mesh_qual_mins.shape[0]), mesh_qual_mins, label=f"Run #{run+1}")

            print()
        

    data_dir = pathlib.Path("parameter_study/data")
    np.save(data_dir / "depth.npy", mesh_qualities_over_runs)
    # print((str(data_dir / "depth.npy")))

    fig_dir = pathlib.Path("parameter_study/figures")
    fig, axs = min_mesh_qual_fig_ax
    for depths_num in range(len(widths_array)):
        ax = axs[depths_num // 2, depths_num % 2]
        ax.legend()
        ax.set_title(f"Depth {depths_num+1}")
    fig.savefig(fig_dir / "depth_min_mq.pdf")
    # print(str(fig_dir / "depth_min_mq.pdf"))

    return



if __name__ == "__main__":
    main()
