import dolfin as df
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"{device = }")

from tools.loading import load_mesh
from conf import mesh_file_loc

fluid_mesh = load_mesh(mesh_file_loc, with_submesh=True)

from tools.dual_mesh import centroid_refine

dual_fluid_mesh = centroid_refine(fluid_mesh)


from networks.loading import build_model
import yaml
with open("models/clem_grad/yankee/model.yml", "r") as infile:
    obj = yaml.safe_load(infile.read())

model = build_model(obj)
model.load_state_dict(torch.load("models/clem_grad/yankee/prepless_state_dict.pt"))

print(model)

V_scal_primal = df.FunctionSpace(fluid_mesh, "CG", 1)

V_primal = df.VectorFunctionSpace(fluid_mesh, "CG", 1)


Q_scal = df.FunctionSpace(fluid_mesh, "DG", 0)
Q_vec = df.VectorFunctionSpace(fluid_mesh, "DG", 0)
Q = df.TensorFunctionSpace(fluid_mesh, "DG", 0, shape=(2,2))

# Make mask function for mesh and dual mesh

from networks.masknet import poisson_mask_custom
from conf import poisson_mask_f

mask_df_primal = poisson_mask_custom(V_scal_primal, poisson_mask_f, normalize=True)

mask_df_q = df.project(mask_df_primal, Q_scal)
print(mask_df_q.vector().get_local())
q_dof_locs = mask_df_q.function_space().tabulate_dof_coordinates()
print(q_dof_locs)



# Make mask network for mesh and dual mesh
from networks.general import TensorModule

mask_tensor_primal = torch.tensor(mask_df_primal.vector().get_local(), dtype=torch.get_default_dtype())
mask_primal = TensorModule(mask_tensor_primal)
mask_tensor_q = torch.tensor(mask_df_q.vector().get_local(), dtype=torch.get_default_dtype())
mask_q = TensorModule(mask_tensor_q)


# Make base network, just return correct dimensions of dataset

from networks.general import TrimModule
indices = torch.LongTensor(range(2))
base = TrimModule(indices, dim=-1)

# PrependModule to input vector coordinates to the model.

from networks.general import PrependModule
dof_coordinates_primal = torch.tensor(V_scal_primal.tabulate_dof_coordinates(), dtype=torch.get_default_dtype())
prepend_primal = PrependModule(dof_coordinates_primal)

dof_coordinates_q = torch.tensor(Q_scal.tabulate_dof_coordinates(), dtype=torch.get_default_dtype())
prepend_q = PrependModule(dof_coordinates_q)

network_primal = nn.Sequential(prepend_primal, model)

network_q = nn.Sequential(prepend_q, model)

# Construct MaskNets

from networks.masknet import MaskNet
mask_net_primal = MaskNet(network_primal, base, mask_primal)

mask_net_primal.to(device)

mask_net_q = MaskNet(network_q, base, mask_q)
mask_net_q.to(device)


# Create dataset

from data_prep.transforms import DofPermutationTransform
from conf import submesh_conversion_cg1_loc
perm_tens = torch.LongTensor(np.load(submesh_conversion_cg1_loc))
dof_perm_transform = DofPermutationTransform(perm_tens, dim=-2)
transform = dof_perm_transform


from data_prep.clement.dataset import learnextClementGradDataset
from conf import test_checkpoints
prefix = "data_prep/clement/data_store/grad/clm_grad"
test_dataset = learnextClementGradDataset(prefix=prefix, checkpoints=test_checkpoints[:100],
                                        transform=transform, target_transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

u_primal = df.Function(V_primal)


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

def u_du_to_torch(u: df.Function, du: df.Function) -> torch.Tensor:

    x = np.zeros((u.function_space().tabulate_dof_coordinates().shape[0]//2, 6))

    u_loc = u.vector().get_local()
    x[:,0] = u_loc[::2]
    x[:,1] = u_loc[1::2]
    du_loc = du.vector().get_local()
    x[:,2] = du_loc[::4]
    x[:,3] = du_loc[1::4]
    x[:,4] = du_loc[2::4]
    x[:,5] = du_loc[3::4]
    return torch.tensor(x, dtype=torch.get_default_dtype())


primal_file_path = "cc_primal_mesh.xdmf"
q_file_path = "cc_dg.xdmf"

p = Path(primal_file_path)
if p.exists():
    p.unlink()
p = Path(q_file_path)
if p.exists():
    p.unlink()
primal_file = df.XDMFFile(primal_file_path)
q_file = df.XDMFFile(q_file_path)

from data_prep.clement.clement import clement_interpolate

gh_primal, CI_primal = clement_interpolate(df.grad(u_primal), with_CI=True)


primal_file.write_checkpoint(mask_df_primal, "mask", 0, append=True)
q_file.write_checkpoint(mask_df_q, "mask", 0, append=True)

pred_primal = df.Function(V_primal)
pred_q = df.Function(Q_vec)

from tqdm import tqdm

for k, (x_primal, _) in enumerate(tqdm(test_dataloader)):
    torch2fenics(x_primal[0,:,:2], u_primal)

    gh_primal = CI_primal()

    uh_q = df.interpolate(u_primal, Q_vec)
    gh = df.project(df.grad(u_primal), Q)

    x_q = u_du_to_torch(uh_q, gh)

    y_primal = network_primal(x_primal)
    y_q = network_q(x_q)

    torch2fenics(y_primal[0,...], pred_primal)
    torch2fenics(y_q, pred_q)


    primal_file.write_checkpoint(u_primal, "uh", k, append=True)

    primal_file.write_checkpoint(gh_primal, "gh", k, append=True)

    q_file.write_checkpoint(uh_q, "uh", k, append=True)
    q_file.write_checkpoint(gh, "gh", k, append=True)

    primal_file.write_checkpoint(pred_primal, "pred", k, append=True)
    q_file.write_checkpoint(pred_q, "pred", k, append=True)



