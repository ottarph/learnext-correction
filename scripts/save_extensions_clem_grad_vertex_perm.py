import torch
import torch.nn as nn
import numpy as np
import dolfin as df
from torch.utils.data import DataLoader
from tqdm import tqdm
import pathlib

def main():

    torch.manual_seed(0)
    vertex_perm = torch.randperm(3935)
    vertex_inv_perm = torch.zeros_like(vertex_perm)
    vertex_inv_perm[vertex_perm] = torch.arange(3935)

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
    widths = [8] + [128]*6 + [2]
    mlp = MLP(widths, activation=nn.ReLU())
    # MLP takes input (x, y, u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)

    x_mean = torch.tensor([ 8.4319e-01,  2.0462e-01, -1.5600e-03,  4.7358e-04, -5.4384e-03,
                9.0626e-04,  1.3179e-03,  7.8762e-04])
    x_std = torch.tensor([0.6965, 0.1011, 0.0035, 0.0134, 0.0425, 0.0468, 0.1392, 0.1484])
    class Normalizer(nn.Module):
        def __init__(self, x_mean: torch.Tensor, x_std: torch.Tensor):
            super().__init__()
            self.x_mean = nn.Parameter(x_mean, requires_grad=False)
            self.x_std = nn.Parameter(x_std, requires_grad=False)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (x - self.x_mean) / self.x_std
        
    normalizer = Normalizer(x_mean, x_std)
    dof_coordinates = torch.tensor(V_scal.tabulate_dof_coordinates(), dtype=torch.get_default_dtype())
    prepend = PrependModule(dof_coordinates)
    # Prepend inserts (x, y) to beginning of (u_x, u_y, d_x u_x, d_y u_x, d_x u_y, d_y u_y)
    # to make correct input of MLP.

    network = nn.Sequential(prepend, normalizer, mlp)


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
    run_name = "yankee"
    context.load_model(model_folder_dir+run_name)
    context.network.eval()
    print(context.network)
    print(context.network.network)
    print(context.network.network[1].state_dict())

    class VertexPermuter(nn.Module):
        def __init__(self, vertex_perm: torch.LongTensor):
            super().__init__()
            self.vertex_perm = vertex_perm
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.index_select(x, -2, self.vertex_perm)
        
    cnn = context.network.network
    cnn.insert(2, VertexPermuter(vertex_perm))
    cnn.append(VertexPermuter(vertex_inv_perm))

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
    file_name = f"foo/{run_name}_perm.xdmf"
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
