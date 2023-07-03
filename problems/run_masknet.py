import numpy as np
import dolfin as df
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

from tools.loading import *
from tools.plots import *

from networks.masknet import *

from typing import Callable

class MLP(nn.Module):

    def __init__(self, widths: list[int], activation: Callable = nn.ReLU()):
        super().__init__()

        self.widths = widths
        self.activation = activation
        self.cost_function: callable = None
        self.optimizer: torch.optim = None

        layers = []
        for w1, w2 in zip(widths[:-1], widths[1:]):
            layers.append(nn.Linear(w1, w2))
            layers.append(self.activation)
        self.layers = nn.Sequential(*layers[:-1])

        self.train_hist: list[float] = []
        self.test_hist: list[float] = []
        self.epoch: int = 0

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) >= 2, "Must be batched."
        assert x.shape[-1] == self.widths[0], "Dimension of argument must match in non-batch dimension."

        return self.layers(x)

class Context:

    def __init__(self, network: nn.Module, cost_function: Callable, optimizer: torch.optim.Optimizer):

        self.network = network
        self.cost_function = cost_function
        self.optimizer = optimizer

        self.epoch: int = 0
        self.train_hist: list[float] = []
        self.test_hist: list[float] = []

        return
    



def train_network_step(context: Context, x: torch.Tensor, y: torch.Tensor, callback: Callable[[Context], None] | None) -> None:
    network = context.network

    context.optimizer.zero_grad()
    cost = context.cost_function(network(x), y)
    cost.backward()
    context.optimizer.step()

    context.train_hist.append(cost.item())
    context.epoch += 1

    if callback is not None:
        callback(context)

    return

def harmonic_to_biharmonic_train_single_checkpoint(context: Context, checkpoint: int, num_epochs: int) -> None:

    from conf import OutputLoc, train_checkpoints

    data_file_loc = OutputLoc + "/Extension/Data"
    mesh_file_loc = OutputLoc + "/Mesh_Generation"

    total_mesh, fluid_mesh, solid_mesh = load_mesh(mesh_file_loc)
    V = df.VectorFunctionSpace(fluid_mesh, "CG", 2, 2)
    u_bih = df.Function(V)

    load_biharmonic_data(data_file_loc, u_bih, checkpoint)
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
    
    from timeit import default_timer as timer
    from conf import OutputLoc, vandermonde_loc

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
    mask = fenics_to_femnet(laplace_mask(V_scal))

    eval_coords = torch.tensor(V.tabulate_dof_coordinates()[::2][None,...])

    network = masknet(mlp, base, mask)
    network.load_vandermonde(vandermonde_loc)

    cost_function = nn.MSELoss()
    # optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-1)
    optimizer = torch.optim.Adam(mlp.parameters())

    context = Context(network, cost_function, optimizer)
    checkpoint = 0
    num_epochs = 200

    start = timer()

    harmonic_to_biharmonic_train_single_checkpoint(context, checkpoint, num_epochs)

    end = timer()


    plt.figure()
    plt.plot(range(context.epoch), context.train_hist, 'k-')
    plt.savefig("foo.png", dpi=150)


    u_bih_fn = fenics_to_femnet(u_bih)
    u_bih_fn.vandermonde = torch.load(vandermonde_loc)
    u_bih_fn.invalidate_cache = False

    print(network(eval_coords) - u_bih_fn(eval_coords))

    plt.figure(figsize=(12,6))
    x = eval_coords[0,...].detach().numpy()
    u = (u_bih_fn(eval_coords) - network(eval_coords))[0,...].detach().numpy()
    uu = np.linalg.norm(u, axis=-1)
    plt.scatter(x[:,0], x[:,1], uu)
    plt.colorbar()
    plt.savefig("bar.png", dpi=200)


    print(end - start)

    return


if __name__ == "__main__":
    main()
