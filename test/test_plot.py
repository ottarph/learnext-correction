import numpy as np
import matplotlib.pyplot as plt
import dolfin as df

from tools.loading import *
from tools.plots import *

from conf import mesh_file_loc, harmonic_file_loc


total_mesh, fluid_mesh, solid_mesh = load_mesh(mesh_file_loc)

V = df.VectorFunctionSpace(fluid_mesh, "CG", 2, 2)
u = df.Function(V)

load_harmonic_data(harmonic_file_loc, u)


fig = fenics_to_scatter(u)
ax = fig.gca()
ax.set_title("Scatter test")
fig.savefig("test/figures/scatter_test.png", dpi=100)

fig = fenics_to_scatter_moved(u)
ax = fig.gca()
ax.set_title("Scatter moved test")
fig.savefig("test/figures/scatter_moved_test.png", dpi=100)


