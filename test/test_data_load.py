import numpy as np
import matplotlib.pyplot as plt
import dolfin as df

from tools.loading import *

from conf import OutputLoc

data_file_loc = OutputLoc + "/Extension/Data"
mesh_file_loc = OutputLoc + "/Mesh_Generation"

total_mesh, fluid_mesh, solid_mesh = load_mesh(mesh_file_loc)

V = df.VectorFunctionSpace(fluid_mesh, "CG", 2, 2)
u = df.Function(V)

load_harmonic_data(data_file_loc, u)
load_biharmonic_data(data_file_loc, u)
