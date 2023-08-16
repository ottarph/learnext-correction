import torch
import torch.nn as nn
import numpy as np
import dolfin as df

import pathlib
from torch.utils.data import DataLoader
from timeit import default_timer as timer

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{device = }")

    torch.manual_seed(seed=0)

    """ Load fluid mesh """
    fluid_mesh_loc = "data_prep/artificial/working_space/fluid.h5"
    fluid_mesh = df.Mesh()
    with df.HDF5File(fluid_mesh.mpi_comm(), fluid_mesh_loc, 'r') as h5:
        h5.read(fluid_mesh, 'mesh', False)


    """ Choose problem setup """
    run_name = "golf"
    model_dir = f"models/artificial/{run_name}"
    results_dir = f"results/artificial/{run_name}"
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)
    latest_results_dir = "results/latest"
    latest_model_dir = "models/artificial/latest"


    """ Build model """

    """ Base """

    from networks.general import TrimModule
    indices = torch.LongTensor(range(2))
    base = TrimModule(indices, dim=-1)

    """ Mask """
    
    V_scal = df.FunctionSpace(fluid_mesh, "CG", 1)
    from conf import poisson_mask_f
    from networks.masknet import poisson_mask_custom
    df.set_log_active(False)
    mask_df = poisson_mask_custom(V_scal, poisson_mask_f, normalize=True)

    from networks.general import TensorModule
    mask = TensorModule(torch.tensor(mask_df.vector().get_local(), dtype=torch.get_default_dtype()))

    """ Correction """
    coordinates = V_scal.tabulate_dof_coordinates()

    from networks.general import PrependModule
    prepend = PrependModule(torch.tensor(coordinates, dtype=torch.get_default_dtype()))

    from networks.loading import load_model
    norm_mlp_stack = load_model(model_dir = model_dir, load_state_dict = False, mode = "yaml")
    
    correction = nn.Sequential(prepend, norm_mlp_stack)

    from networks.masknet import MaskNet
    mask_net = MaskNet(correction, base, mask)
    mask_net.to(device)

    """ Load dataset """

    from data_prep.artificial.dataset import ArtificialLearnextDataset
    prefix = "data_prep/artificial/data_store/grad/art"
    # train_checkpoints = range(0, 200)
    # validation_checkpoints = range(200, 252)
    # test_checkpoints = range(252, 303)
    # train_checkpoints = range(0, 404)
    # validation_checkpoints = range(0, 404)
    # test_checkpoints = range(404)
    # train_dataset = ArtificialLearnextDataset(prefix=prefix, checkpoints=train_checkpoints,
    #                                      transform=None, target_transform=None)
    # val_dataset = ArtificialLearnextDataset(prefix=prefix, checkpoints=validation_checkpoints,
    #                                          transform=None, target_transform=None)
    # test_dataset = ArtificialLearnextDataset(prefix=prefix, checkpoints=test_checkpoints,
    #                                      transform=None, target_transform=None)
    from torch.utils.data import random_split
    checkpoints = range(606)
    dataset = ArtificialLearnextDataset(prefix=prefix, checkpoints=checkpoints,
                                        transform=None, target_transform=None)
    train_dataset, val_dataset = random_split(dataset, [0.85, 0.15])
    test_dataset = dataset
    test_checkpoints = checkpoints
    
    """ Make dataloaders """
    batch_size = 128
    shuffle = True
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    x, y = next(iter(train_dataloader))
    x, y = x.to(device), y.to(device)
    assert x.shape == (batch_size, fluid_mesh.num_vertices(), 6)
    # In:  (u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y), where u is from harmonic extension
    assert y.shape == (batch_size, fluid_mesh.num_vertices(), 2)
    # Out: (u_x, u_y), where u is from biharmonic extension

    assert mask(x).shape == (fluid_mesh.num_vertices(),)
    assert base(x).shape == (batch_size, fluid_mesh.num_vertices(), 2)
    assert correction(x).shape == (batch_size, fluid_mesh.num_vertices(), 2)
    assert mask_net(x).shape == (batch_size, fluid_mesh.num_vertices(), 2)
    assert (mask_net(x) - y).shape == (batch_size, fluid_mesh.num_vertices(), 2)


    """ Choose cost function """
    cost_function = nn.L1Loss()
    val_cost_function = nn.L1Loss()

    """ Choose optimizer """
    opt = torch.optim.AdamW(correction.parameters(), weight_decay=1e-2)

    """ Choose LR scheduler. """
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5)
    # sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
    # sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)

    """ Set up training problem """
    from networks.training import Context, train_with_dataloader
    context = Context(mask_net, cost_function, opt, sched, val_cost_function)


    """ Callback plotting training """
    def callback(context: Context):
        if context.epoch % 40 == 0:
            context.plot_results(latest_results_dir)
        return

    num_epochs = 200

    start = timer()
    train_with_dataloader(context, train_dataloader, num_epochs, device, val_dataloader, callback)
    end = timer()

    print(f"T = {(end - start):.2f} s")

    mask_net.eval()


    context.save_results(results_dir)
    context.save_summary(results_dir)
    context.plot_results(results_dir)

    context.save_results(latest_results_dir)
    context.save_summary(latest_results_dir)
    context.plot_results(latest_results_dir)


    context.save_model(model_dir)
    torch.save(norm_mlp_stack.state_dict(), f"{model_dir}/state_dict.pt")
    torch.save(norm_mlp_stack.state_dict(), f"{latest_model_dir}/state_dict.pt")


    xdmf_dir = "fenics_output/artificial"
    xdmf_name = f"pred_artificial_{run_name}"
    from tools.saving import save_extensions_to_xdmf
    mask_net.to("cpu")
    save_extensions_to_xdmf(mask_net, test_dataloader, df.VectorFunctionSpace(fluid_mesh, "CG", 1), xdmf_name,
                            save_dir=xdmf_dir, start_checkpoint=test_checkpoints[0])

    return




if __name__ == "__main__":
    main()
