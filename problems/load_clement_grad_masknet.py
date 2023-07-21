from problems.run_clement_grad_masknet import *


# torch.set_default_dtype(torch.float64)
torch.set_default_dtype(torch.float32)

from timeit import default_timer as timer
from conf import mesh_file_loc, with_submesh

torch.manual_seed(0)


fluid_mesh = load_mesh(mesh_file_loc, with_submesh)

V_scal = df.FunctionSpace(fluid_mesh, "CG", 1) # Linear scalar polynomials over triangular mesh

mask_df = poisson_mask(V_scal, normalize=True)
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


from data_prep.clement.dataset import learnextClementGradDataset
from conf import train_checkpoints
prefix = "data_prep/clement/data_store/grad/clm_grad"
dataset = learnextClementGradDataset(prefix=prefix, checkpoints=train_checkpoints)


batch_size = 16
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
# optimizer = torch.optim.Adam(mlp.parameters()) # Good batch size: 1024?
optimizer = torch.optim.LBFGS(mlp.parameters(), line_search_fn="strong_wolfe") # Good batch size: 16


context = Context(mask_net, cost_function, optimizer)

# loadpref = "mse_lbfgs_8_128_2_clm_grad"
loadpref = "mse_adam_8_128_2_clm_grad"
# context.load("models/LBFGS_8_128_2_clm")
# context.load("models/new_mask_LBFGS_8_128_2_clm_grad")
# context.load("models/mask_ex_LBFGS_8_128_2_clm")
context.load(f"models/{loadpref}")


# print(x.shape)
# print(y.shape)

u_diff_tensor = y[0,...] - mask_net(x)[0,...]
# print(u_diff_tensor.shape)

V = df.VectorFunctionSpace(fluid_mesh, "CG", 1, 2)
u_diff = df.Function(V)

new_dofs = np.zeros_like(u_diff.vector().get_local())
new_dofs[::2] = u_diff_tensor[:,0].double().detach().numpy()
new_dofs[1::2] = u_diff_tensor[:,1].double().detach().numpy()
# print(new_dofs)

# print(np.max(np.abs(new_dofs)))

u_diff.vector().set_local(new_dofs)

# outfile = df.File("fenics_output/clem_masknet_LBFGS.pvd")
# outfile << u_diff

print()
print(loadpref)
print()

print("MAE")
print("biharm - harm:", torch.mean(torch.linalg.norm(y-x[...,:2], ord=1, dim=-1)))
# print(torch.mean(torch.linalg.norm(mask_net.network(x), ord=1, dim=-1)))
# print(torch.mean(torch.linalg.norm(mask_net.network(x)*mask_net.mask(x)[...,None], ord=1, dim=-1)))
print("biharm - pred:", torch.mean(torch.linalg.norm(y - mask_net(x), ord=1, dim=-1)))
#MAE
print()

print("RMSE:")
print("biharm - harm:", torch.sqrt(torch.mean(torch.linalg.norm(y - x[...,:2], ord=2, dim=-1)**2)))
# print(torch.sqrt(torch.mean(torch.linalg.norm(mask_net.network(x), ord=2, dim=-1)**2)))
# print(torch.sqrt(torch.mean(torch.linalg.norm(mask_net.network(x)*mask_net.mask(x)[...,None], ord=2, dim=-1)**2)))
print("biharm - pred:", torch.sqrt(torch.mean(torch.linalg.norm(y - mask_net(x), ord=2, dim=-1)**2)))
print()

print("MRSE:")
print("biharm - harm:", torch.mean(torch.linalg.norm(y - x[...,:2], ord=2, dim=-1)))
# print(torch.mean(torch.linalg.norm(mask_net.network(x), ord=2, dim=-1)))
# print(torch.mean(torch.linalg.norm(mask_net.network(x)*mask_net.mask(x)[...,None], ord=2, dim=-1)))
print("biharm - pred:", torch.mean(torch.linalg.norm(y - mask_net(x), ord=2, dim=-1)))



# print(torch.linalg.norm(y - x[...,:2], dim=-1).shape)
# print(y.shape)

# A_1 = mask_net.network[1].layers[0].weight.detach().clone()
# print(A_1)
# context.load("models/mask_ex_LBFGS_8_128_2_clm")
# A_2 = mask_net.network[1].layers[0].weight.detach().clone()
# print(A_2)

# print(A_1 - A_2)
