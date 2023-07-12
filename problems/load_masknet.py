from problems.run_masknet import *

from conf import OutputLoc, vandermonde_loc

torch.manual_seed(0)

data_file_loc = OutputLoc + "/Extension/Data"
mesh_file_loc = OutputLoc + "/Mesh_Generation"

total_mesh, fluid_mesh, solid_mesh = load_mesh(mesh_file_loc)

V = df.VectorFunctionSpace(fluid_mesh, "CG", 2, 2)
u_bih = df.Function(V)
V_scal = df.FunctionSpace(fluid_mesh, "CG", 2)

load_biharmonic_data(data_file_loc, u_bih, 0)

widths = [2, 128, 2]
mlp = MLP(widths, activation=nn.ReLU())
mlp.double()

base = fenics_to_femnet(laplace_extension(u_bih))
mask = fenics_to_femnet(poisson_mask(V_scal))

eval_coords = torch.tensor(V.tabulate_dof_coordinates()[::2][None,...])

network = FemNetMasknet(mlp, base, mask)
network.load_vandermonde(vandermonde_loc)


cost_function = nn.MSELoss()
optimizer = torch.optim.LBFGS(mlp.parameters(), line_search_fn="strong_wolfe")

context = Context(network, cost_function, optimizer)

context.load("models/2_128_2_LBFGS")


plt.plot(range(len(context.train_hist)), context.train_hist, 'k-')
plt.savefig("baz.png", dpi=100)

""" Check maximum deviation """

pred = network(eval_coords)

bih_fn = fenics_to_femnet(u_bih)
bih_fn.vandermonde = mask.vandermonde.detach().clone()
bih_fn.invalidate_cache = False

bih_pred = bih_fn(eval_coords)

diff = bih_pred - pred

max_diff = 0.0
max_ind = -1

for i in range(diff.shape[1]):
    if torch.linalg.norm(diff[0,i,:]) > max_diff:
        max_diff = torch.linalg.norm(diff[0,i,:])
        max_ind = i

print(f"{max_diff=}")
print(f"{eval_coords[0,max_ind,:]=}")


""" Create a fenics function with the difference between the two."""

u_diff = df.Function(V)



x_dofs = diff[0,:,0].detach().numpy()
y_dofs = diff[0,:,1].detach().numpy()

new_dofs = np.zeros_like(u_diff.vector()[:])
new_dofs[::2] = x_dofs
new_dofs[1::2] = y_dofs

u_diff.vector()[:] = new_dofs

outfile = df.File("fenics_output/masknet_LBFGS.pvd")
outfile << u_diff
# with df.File("fenics_output/masknet_LBFGS.pvd") as outfile:
    # outfile << u_diff


