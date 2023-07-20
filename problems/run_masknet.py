import numpy as np
import dolfin as df
import torch
import torch.nn as nn

from tools.loading import *
from tools.plots import *

from networks.masknet import *
from networks.general import *


def harmonic_to_biharmonic_train_single_checkpoint(context: Context, checkpoint: int, num_epochs: int) -> None:

    from conf import mesh_file_loc, biharmonic_file_loc

    _, fluid_mesh, _ = load_mesh(mesh_file_loc)
    V = df.VectorFunctionSpace(fluid_mesh, "CG", 2, 2)
    u_bih = df.Function(V)

    load_biharmonic_data(biharmonic_file_loc, u_bih, checkpoint)
    u_bih_fn = fenics_to_femnet(u_bih)

    from conf import vandermonde_loc
    u_bih_fn.vandermonde = torch.load(vandermonde_loc)

    # eval_coords = V.mesh.coordinates()
    eval_coords = torch.tensor(V.tabulate_dof_coordinates()[::2][None,...])
    eval_coords = eval_coords.double()

    targets = u_bih_fn(eval_coords).detach()


    def callback(context: Context) -> None:
        print(f"epoch #{context.epoch-1:03}, loss={context.train_hist[-1]}")

    for _ in range(num_epochs):
        train_network_step(context, eval_coords, targets, callback=callback)

    return



def main():
    torch.set_default_dtype(torch.float64)
    
    from timeit import default_timer as timer
    from conf import mesh_file_loc, biharmonic_file_loc, vandermonde_loc

    torch.manual_seed(0)

    _, fluid_mesh, _ = load_mesh(mesh_file_loc)

    V = df.VectorFunctionSpace(fluid_mesh, "CG", 2, 2)
    u_bih = df.Function(V)
    V_scal = df.FunctionSpace(fluid_mesh, "CG", 2)

    load_biharmonic_data(biharmonic_file_loc, u_bih, 0)

    widths = [2, 128, 2]
    mlp = MLP(widths, activation=nn.ReLU())
    mlp.double()

    base = fenics_to_femnet(laplace_extension(u_bih))
    mask = fenics_to_femnet(poisson_mask(V_scal))

    eval_coords = torch.tensor(V.tabulate_dof_coordinates()[::2][None,...])

    masknet = FemNetMasknet(mlp, base, mask)
    masknet.load_vandermonde(vandermonde_loc)

    cost_function = nn.MSELoss()
    # optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-1)
    optimizer = torch.optim.Adam(mlp.parameters())
    # optimizer = torch.optim.LBFGS(mlp.parameters(), line_search_fn="strong_wolfe")

    context = Context(masknet, cost_function, optimizer)
    checkpoint = 0
    num_epochs = 10

    start = timer()

    harmonic_to_biharmonic_train_single_checkpoint(context, checkpoint, num_epochs)

    end = timer()

    # context.save("models/2_128_2_LBFGS")


    plt.figure()
    plt.plot(range(context.epoch), context.train_hist, 'k-')
    plt.savefig("foo.png", dpi=150)


    u_bih_fn = fenics_to_femnet(u_bih)
    u_bih_fn.vandermonde = torch.load(vandermonde_loc)
    u_bih_fn.invalidate_cache = False

    print(torch.mean(torch.linalg.norm(masknet(eval_coords) - u_bih_fn(eval_coords), dim=-1)))

    plt.figure(figsize=(12,6))
    x = eval_coords[0,...].detach().numpy()
    u = (u_bih_fn(eval_coords) - masknet(eval_coords))[0,...].detach().numpy()
    uu = np.linalg.norm(u, axis=-1)
    plt.scatter(x[:,0], x[:,1], uu)
    plt.colorbar()
    plt.savefig("bar.png", dpi=200)


    print(end - start)

    return


if __name__ == "__main__":
    main()
