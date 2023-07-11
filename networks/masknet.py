import numpy as np
import dolfin as df
import torch
import torch.nn as nn
import fem_nets

def laplace_mask(V: df.FunctionSpace) -> df.Function:
    """
        -Delta u = 1 in Omega
               u = 0 on dOmega
    """

    def boundary(x, on_boundary):
        return on_boundary
    u0 = df.Constant(0.0)
    
    bc = df.DirichletBC(V, u0, boundary)

    f = df.Constant(1.0)

    u = df.TrialFunction(V)
    v = df.TestFunction(V)

    a = df.inner(df.nabla_grad(u), df.nabla_grad(v)) * df.dx
    l = f * v * df.dx

    uh = df.Function(V)
    df.solve(a == l, uh, bc)

    return uh

def laplace_extension(g: df.Function) -> df.Function:
    """
        -Delta u = 0 in Omega
               u = g on dOmega
    """

    V = g.function_space()

    def boundary(x, on_boundary):
        return on_boundary
    
    bc = df.DirichletBC(V, g, boundary)

    f = df.Constant((0.0, 0.0))

    u = df.TrialFunction(V)
    v = df.TestFunction(V)

    a = df.inner(df.nabla_grad(u), df.nabla_grad(v)) * df.dx
    l = df.inner(f, v) * df.dx

    uh = df.Function(V)
    df.solve(a == l, uh, bc)

    return uh

class MaskNet(nn.Module):

    def __init__(self, network: nn.Module, base: nn.Module, mask: nn.Module):
        super().__init__()

        self.network = network
        self.base = base
        self.mask = mask

        """ Freeze the parameters of the base and mask networks. """
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        assert len(x.shape) >= 2
        # assert x.shape[-1] == 2 # Not actually necessary, what if we want to use extra info like gradients?
    
        a = self.base(x)
        b = self.mask(x)[...,None] # Mask output must be unsqueezed to broadcast scalar to vector in last two dimensionsø
        c = self.network(x)

        return a + b * c

class FemNetMasknet(MaskNet):

    def __init__(self, network: nn.Module, base: fem_nets.networks.VectorLagrangeNN, 
                 mask: fem_nets.networks.LagrangeNN):
        super().__init__(network, base, mask)

        self.base.invalidate_cache = False # Don't build the vandermonde matrix again every evaluation
        self.mask.invalidate_cache = False # Don't build the vandermonde matrix again every evaluation

        return
    
    def init(self, x: torch.Tensor | None = None):
        if self.base.vandermonde is None and self.mask.vandermonde is None:
            pdegree = self.base.V.ufl_element().degree()
            if x is None:
                x = torch.tensor(self.mask.V.tabulate_dof_coordinates()[None,...])
            self.mask.vandermonde = self.mask._compute_vandermonde(pdegree)(x, self.mask.mesh)
            self.base.vandermonde = self.mask.vandermonde.detach().clone()
        elif self.base.vandermonde is not None:
            self.mask.vandermonde = self.base.vandermonde.detach().clone()
        elif self.mask.vandermonde is not None:
            self.base.vandermonde = self.mask.vandermonde.detach().clone()

        self.mask.vandermonde.requires_grad_(False)
        self.base.vandermonde.requires_grad_(False)
        
        return
    
    def save_vandermonde(self, fname: str) -> None:

        assert torch.equal(self.base.vandermonde, self.mask.vandermonde)
        assert self.base.vandermonde is not None

        torch.save(self.mask.vandermonde, fname)

        return
    
    def load_vandermonde(self, fname: str) -> None:

        self.mask.vandermonde = torch.load(fname)
        self.base.vandermonde = self.mask.vandermonde.detach().clone()
        self.mask.vandermonde.requires_grad_(False)
        self.base.vandermonde.requires_grad_(False)

        return
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) >= 3, "Fem-nets require (batch_dim, num_points, euclidean_dim)-shape inputs"
        return super().forward(x)


from networks.general import TensorModule    
class TensorMaskNet(MaskNet):

    def __init__(self, network: nn.Module, base: TensorModule | torch.Tensor, mask: TensorModule | torch.Tensor):
        if isinstance(base, torch.Tensor): base = TensorModule(base)
        if isinstance(mask, torch.Tensor): mask = TensorModule(mask)
        super().__init__(network, base, mask)

        return



