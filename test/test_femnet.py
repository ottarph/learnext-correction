import numpy as np
import matplotlib.pyplot as plt
import dolfin as df

from fem_nets import to_torch

from tools.loading import *
from tools.plots import fenics_to_scatter_moved

from conf import mesh_file_loc, harmonic_file_loc


total_mesh, fluid_mesh, solid_mesh = load_mesh(mesh_file_loc)

V = df.VectorFunctionSpace(fluid_mesh, "CG", 2, 2)
u = df.Function(V)

load_harmonic_data(harmonic_file_loc, u)


u_fn = fenics_to_femnet(u)
u2 = femnet_to_fenics(u_fn, V)
print(np.linalg.norm(u.vector()[:] - u2.vector()[:]))

V_scal = df.FunctionSpace(fluid_mesh, "CG", 2)
expr = df.Expression("3*x[0]*x[0] - 2*x[1]", degree=2)
u_scal = df.interpolate(expr, V_scal)

u_scal_fn = fenics_to_femnet(u_scal)
u2_scal = femnet_to_fenics(u_scal_fn, V_scal)
print(np.linalg.norm(u_scal.vector()[:] - u2_scal.vector()[:]))
