import dolfin as df
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device = }")

from tools.loading import load_mesh
from conf import mesh_file_loc

fluid_mesh = load_mesh(mesh_file_loc, with_submesh=True)

from tools.dual_mesh import centroid_refine

dual_fluid_mesh = centroid_refine(fluid_mesh)


df.File('fluid_mesh.pvd') << fluid_mesh
df.File('dual_fluid_mesh.pvd') << dual_fluid_mesh

from networks.loading import build_model
import yaml
with open("models/clem_grad/yankee/model.yml", "r") as infile:
    obj = yaml.safe_load(infile.read())

model = build_model(obj)
model.load_state_dict(torch.load("models/clem_grad/yankee/prepless_state_dict.pt"))

print(model)

V_scal_primal = df.FunctionSpace(fluid_mesh, "CG", 1)
V_scal_dual = df.FunctionSpace(dual_fluid_mesh, "CG", 1)
V_primal = df.VectorFunctionSpace(fluid_mesh, "CG", 1)
V_dual = df.VectorFunctionSpace(dual_fluid_mesh, "CG", 1)

# Make mask function for mesh and dual mesh

from networks.masknet import poisson_mask_custom
from conf import poisson_mask_f

mask_df_primal = poisson_mask_custom(V_scal_primal, poisson_mask_f, normalize=True)
mask_df_dual = poisson_mask_custom(V_scal_dual, poisson_mask_f, normalize=True)

# Make mask network for mesh and dual mesh
from networks.general import TensorModule

mask_tensor_primal = torch.tensor(mask_df_primal.vector().get_local(), dtype=torch.get_default_dtype())
mask_primal = TensorModule(mask_tensor_primal)
mask_tensor_dual = torch.tensor(mask_df_dual.vector().get_local(), dtype=torch.get_default_dtype())
mask_dual = TensorModule(mask_tensor_dual)

# Make base network, just return correct dimensions of dataset

from networks.general import TrimModule
indices = torch.LongTensor(range(2))
base = TrimModule(indices, dim=-1)

# PrependModule to input vector coordinates to the model.

from networks.general import PrependModule
dof_coordinates_primal = torch.tensor(V_scal_primal.tabulate_dof_coordinates(), dtype=torch.get_default_dtype())
prepend_primal = PrependModule(dof_coordinates_primal)
dof_coordinates_dual = torch.tensor(V_scal_dual.tabulate_dof_coordinates(), dtype=torch.get_default_dtype())
prepend_dual = PrependModule(dof_coordinates_dual)

network_primal = nn.Sequential(prepend_primal, model)
network_dual = nn.Sequential(prepend_dual, model)

# Construct MaskNets

from networks.masknet import MaskNet
mask_net_primal = MaskNet(network_primal, base, mask_primal)
mask_net_dual = MaskNet(network_dual, base, mask_dual)
mask_net_primal.to(device)
mask_net_dual.to(device)


# Create dataset

from data_prep.transforms import DofPermutationTransform
from conf import submesh_conversion_cg1_loc
perm_tens = torch.LongTensor(np.load(submesh_conversion_cg1_loc))
dof_perm_transform = DofPermutationTransform(perm_tens, dim=-2)
transform = dof_perm_transform


from data_prep.clement.dataset import learnextClementGradDataset
from conf import test_checkpoints
prefix = "data_prep/clement/data_store/grad/clm_grad"
test_dataset = learnextClementGradDataset(prefix=prefix, checkpoints=test_checkpoints[:5],
                                        transform=transform, target_transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

u_primal = df.Function(V_primal)
u_dual = df.Function(V_dual)

x_primal, _ = next(iter(test_dataloader))

def torch2fenics(x: torch.Tensor, u: df.Function) -> None:
    """In place"""
    new_dofs = np.zeros_like(u.vector().get_local())
    x_np = x.detach().double().numpy()
    new_dofs[0::2] = x_np[:,0]
    new_dofs[1::2] = x_np[:,1]
    u.vector().set_local(new_dofs)
    return
def fenics2torch(u: df.Function) -> torch.Tensor:
    u_loc = u.vector().get_local()
    x_np = np.zeros((u_loc.shape[0]//2, 2))
    x_np[:,0] = u_loc[0::2]
    x_np[:,1] = u_loc[1::2]
    return torch.tensor(x_np, dtype=torch.get_default_dtype())

print(f"{x_primal = }")
torch2fenics(x_primal[0,:,:2], u_primal)
print(f"{df.norm(u_primal) = }")
u_dual = df.interpolate(u_primal, V_dual)
print(f"{df.norm(u_dual) = }")
x_dual = fenics2torch(u_dual)
print(f"{x_dual = }")

primal_file = df.XDMFFile("primal_mesh.xdmf")
dual_file = df.XDMFFile("dual_mesh.xdmf")

from tqdm import tqdm
for x_primal, _ in tqdm(test_dataloader):
    torch2fenics(x_primal[0,:2], u_primal)
    u_dual = df.interpolate(u_primal, V_dual)
    x_dual = fenics2torch(u_dual)[None, ...]

    corrected_primal = mask_net_primal(x_primal)
    corrected_dual = mask_net_dual(x_dual)
    torch2fenics(corrected_primal[0,:,:2])

# Doesnt work this way, need to get the gradient information in the dual mesh.