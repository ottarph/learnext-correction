import numpy as np
import matplotlib.pyplot as plt
import dolfin as df

from tools.loading import *
from tools.plots import fenics_to_scatter_moved

def test_load_data():

    from conf import mesh_file_loc, harmonic_file_loc, biharmonic_file_loc

    _, mesh, _ = load_mesh(mesh_file_loc)
    V = df.VectorFunctionSpace(mesh, "CG", 2, 2)
    u = df.Function(V)

    load_harmonic_data(harmonic_file_loc, u, 0)
    assert np.linalg.norm(u.vector()[:]) > 0.0
    load_biharmonic_data(biharmonic_file_loc, u, 0)
    assert np.linalg.norm(u.vector()[:]) > 0.0

    return

def main():

    from conf import mesh_file_loc, harmonic_file_loc, biharmonic_file_loc

    _, fluid_mesh, _ = load_mesh(mesh_file_loc)

    V = df.VectorFunctionSpace(fluid_mesh, "CG", 2, 2)
    u = df.Function(V)

    load_harmonic_data(harmonic_file_loc, u, checkpoint=0)
    fig = fenics_to_scatter_moved(u)
    fig.gca().set_title("Harmonic test")
    fig.savefig("test/figures/harmonic_load_test.png", dpi=150)
    load_biharmonic_data(biharmonic_file_loc, u, checkpoint=20)
    fig = fenics_to_scatter_moved(u)
    fig.gca().set_title("Biharmonic test")
    fig.savefig("test/figures/biharmonic_load_test.png", dpi=150)

    return

if __name__ == "__main__":
    test_load_data()
    main()
