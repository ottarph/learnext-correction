import torch
import torch.nn as nn
import numpy as np
import dolfin as df
from torch.utils.data import DataLoader
from tqdm import tqdm
import pathlib

def main():



    from conf import submesh_conversion_cg1_loc, with_submesh
    from data_prep.transforms import DofPermutationTransform
    perm_tens = torch.LongTensor(np.load(submesh_conversion_cg1_loc))
    dof_perm_transform = DofPermutationTransform(perm_tens, dim=-2)
    transform = dof_perm_transform if with_submesh else None
    print(f"{with_submesh = }")

    from data_prep.clement.dataset import learnextClementGradDataset
    from conf import test_checkpoints
    prefix = "data_prep/clement/data_store/grad/clm_grad"
    dataset = learnextClementGradDataset(prefix=prefix, checkpoints=test_checkpoints,
                                         transform=transform, target_transform=transform)
    
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    from conf import mesh_file_loc, with_submesh
    from tools.loading import load_mesh
    fluid_mesh = load_mesh(mesh_file_loc, with_submesh)

    V = df.VectorFunctionSpace(fluid_mesh, "CG", 1, 2)
    V_scal = df.FunctionSpace(fluid_mesh, "CG", 1) # Linear scalar polynomials over triangular mesh

    from conf import poisson_mask_f
    from networks.masknet import poisson_mask_custom
    mask_df = poisson_mask_custom(V_scal, poisson_mask_f, normalize = True)

    from networks.general import TensorModule
    mask_tensor = torch.tensor(mask_df.vector().get_local(), dtype=torch.get_default_dtype())
    mask = TensorModule(mask_tensor)

    from networks.general import TrimModule
    indices = torch.LongTensor(range(2))
    base = TrimModule(indices, dim=-1)
    # base returns (u_x, u_y) from (u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)

    from networks.general import MLP, PrependModule
    widths = [8] + [32]*10 + [2]
    mlp = MLP(widths, activation=nn.ReLU())
    from networks.general import MLP_BN
    # mlp = MLP_BN(widths, nn.ReLU())
    # MLP takes input (x, y, u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)

    dof_coordinates = torch.tensor(V_scal.tabulate_dof_coordinates(), dtype=torch.get_default_dtype())
    prepend = PrependModule(dof_coordinates)
    # Prepend inserts (x, y) to beginning of (u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)
    # to make correct input of MLP.
    # network = nn.Sequential(prepend, mlp)
    network = nn.Sequential(prepend, nn.BatchNorm1d(3935), mlp)

    from networks.masknet import MaskNet
    mask_net = MaskNet(network, base, mask)

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


    from networks.training import Context
    context = Context(mask_net, nn.MSELoss(), torch.optim.Adam(mlp.parameters()))
    model_folder_dir = "models/clem_grad/"
    run_name = "victor"
    context.load_model(model_folder_dir+run_name)
    context.network.eval()

    test_loss = 0.0
    test_dataloader_loop = tqdm(dataloader, position=0, desc="Computing test loss")
    with torch.no_grad():
        for x, y in test_dataloader_loop:
            test_loss += context.cost_function(context.network(x), y).item()

    print(f"{test_loss = :.3e}\n")

    save_label = "predicted_extension"


    u_pred = df.Function(V)
    new_coeffs = np.zeros_like(u_pred.vector().get_local())
    k = test_checkpoints[0]
    file_name = f"foo/{run_name}.xdmf"
    if pathlib.Path(file_name).is_file():
        pathlib.Path(file_name).unlink()
    with df.XDMFFile(file_name) as outfile:
        with torch.no_grad():
            pred_loop = tqdm(dataloader, position=0, desc="Writing predictions to file")
            for x, y in pred_loop:
                pred = context.network(x)
                for i in range(pred.shape[0]):
                    coeffs = pred[i,:,:]
                    new_coeffs[::2] = coeffs[:,0].detach().numpy()
                    new_coeffs[1::2] = coeffs[:,1].detach().numpy()
                    u_pred.vector().set_local(new_coeffs)
                    outfile.write_checkpoint(u_pred, save_label, float(k), append=True)
                    k += 1

    print("Conversion completed.")


    return



if __name__ == "__main__":
    main()
