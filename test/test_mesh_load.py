
from dolfin import FunctionSpace, FiniteElement, Function, \
    Constant, Expression, interpolate
import numpy as np
import matplotlib.pyplot as plt

from conf import OutputLoc

""" First load the mesh """

mesh_file_loc = OutputLoc + "/Mesh_Generation"

from tools.loading import load_mesh

# load mesh
mesh, fluid_domain, solid_domain = load_mesh(mesh_file_loc)

print(f"{mesh.num_vertices()=}")
print(f"{fluid_domain.num_vertices()=}")
print(f"{solid_domain.num_vertices()=}")


mesh = fluid_domain

""" Make scalar function space for mesh load test """
V_el = FiniteElement("CG", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, V_el)
f = Function(V)
f.interpolate(Constant(1.0))
""" This has constant values 1.0 over the entire mesh when viewed in paraview """

f_expr = Expression("sin(x[0]) + cos(4*x[1])", degree=2)
f = interpolate(f_expr, V)
""" This interpolates the correct function """


x = mesh.coordinates()
uu = f.compute_vertex_values()

# https://matplotlib.org/stable/gallery/images_contours_and_fields/tripcolor_demo.html

fig, ax = plt.subplots(figsize=(12,6))
ax.scatter(x[:,0], x[:,1], c=uu)
fig.savefig("test/figures/mpl_foo.png", dpi=150)

fig, ax = plt.subplots(figsize=(12,6))
x1 = fluid_domain.coordinates()
x2 = solid_domain.coordinates()
ax.scatter(x1[:,0], x1[:,1], c="red", label="fluid_domain")
ax.scatter(x2[:,0], x2[:,1], c="blue", label="solid_domain")
ax.legend()
fig.savefig("test/figures/domains.png", dpi=200)
print(type(fig))
