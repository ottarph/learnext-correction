import dolfin as df
import numpy as np

from data_prep.clement.clement import *
from conf import OutputLoc
from tools.loading import *


def clement_visualization():

    _, fluid_mesh, _ = load_mesh(OutputLoc + "/Mesh_Generation")

    V = df.VectorFunctionSpace(fluid_mesh, "CG", 2, 2)
    harmonic = df.Function(V)

    harmonic_file = df.XDMFFile(OutputLoc + "/Extension/Data/" + "input_.xdmf")

    checkpoint = 0
    harmonic_file.read_checkpoint(harmonic, "input", checkpoint)

    print(harmonic.compute_vertex_values())
    Q = df.VectorFunctionSpace(fluid_mesh, 'DG', 1)
    qh = df.interpolate(harmonic, Q)
    cl = clement_interpolate(df.grad(qh))
    print(cl.compute_vertex_values())

    print(cl.function_space())

    outfile = df.File("fenics_output/clement_test_viz.pvd")
    outfile << cl

    return

def test_clement_grad():

    def f(x):
        return 3.0*x[0] + 2.0*x[1]
    def g(x):
        return np.array([3.0, 2.0])

    N = 10
    mesh = df.UnitSquareMesh(N, N)

    V = df.VectorFunctionSpace(mesh, "CG", 1, 2)

    u = df.Function(V)
    new_dofs = np.zeros_like(u.vector().get_local())
    for i, x in enumerate(V.tabulate_dof_coordinates()):
        if i % 2 == 0:
            new_dofs[i] = f(x)
        else:
            new_dofs[i] = -f(x)
    u.vector().set_local(new_dofs)

    Q = df.VectorFunctionSpace(mesh, "DG", 1, 2)
    qh = df.interpolate(u, Q)
    gh = clement_interpolate(df.grad(qh))
    # gh.compute_vertex_values() = (d_x u_x, d_y u_x, d_x u_y, d_y u_y, ...[*121])
    # gh.vector().get_local() = (d_x u_x, ...[*121], d_y u_x, ...[*121], d_x u_y, ...[*121], d_y u_y, ...[*121])

    assert np.all(np.isclose(gh.vector().get_local()[0::4],  3.0, atol=1e-15)) # Offset  zero is d_x u_x
    assert np.all(np.isclose(gh.vector().get_local()[1::4],  2.0, atol=1e-15)) # Offset   one is d_x u_y
    assert np.all(np.isclose(gh.vector().get_local()[2::4], -3.0, atol=1e-15)) # Offset   two is d_y u_x
    assert np.all(np.isclose(gh.vector().get_local()[3::4], -2.0, atol=1e-15)) # Offset three is d_y u_y

    # Dof locations for u and gh are the same
    assert np.all(np.equal(u.function_space().tabulate_dof_coordinates()[::2,:], gh.function_space().tabulate_dof_coordinates()[::4,:]))

    return


def test_clement_hess():

    def f(x):
        return 3.0*x[0]**2 + 2.0*x[1]**2 - 1.0*x[0]*x[1]
        
    N = 10
    mesh = df.UnitSquareMesh(N, N)

    V = df.FunctionSpace(mesh, "CG", 2)

    u = df.Function(V)
    new_dofs = np.zeros_like(u.vector().get_local())
    for i, x in enumerate(V.tabulate_dof_coordinates()):
        new_dofs[i] = f(x)
    u.vector().set_local(new_dofs)

    Q = df.FunctionSpace(mesh, "DG", 2)
    qh = df.interpolate(u, Q)
    gh = clement_interpolate(df.grad(df.grad(qh)))

    Np = N+1
    Np2 = Np * Np
    assert np.all(np.isclose(gh.compute_vertex_values()[:Np2], 6.0))
    assert np.all(np.isclose(gh.compute_vertex_values()[Np2:2*Np2], -1.0))
    assert np.all(np.isclose(gh.compute_vertex_values()[2*Np2:3*Np2], -1.0))
    assert np.all(np.isclose(gh.compute_vertex_values()[3*Np2:], 4.0))

    return


def check_reuse_matrix():

    def f(x):
        return 3.0*x[0] + 2.0*x[1]
    def g(x):
        return np.array([3.0, 2.0])

    N = 10
    mesh = df.UnitSquareMesh(N, N)

    V = df.VectorFunctionSpace(mesh, "CG", 1, 2)

    u = df.Function(V)
    new_dofs = np.zeros_like(u.vector().get_local())
    for i, x in enumerate(V.tabulate_dof_coordinates()):
        if i % 2 == 0:
            new_dofs[i] = f(x)
        else:
            new_dofs[i] = -f(x)
    u.vector().set_local(new_dofs)

    Q = df.VectorFunctionSpace(mesh, "DG", 1, 2)
    qh = df.interpolate(u, Q)
    gh, clem_interp = clement_interpolate(df.grad(qh), with_CI=True)

    gh_old_dofs = gh.vector().get_local().copy()
    u.vector()[:] *= 0.0

    # u2 = df.Function(V)
    new_dofs = np.zeros_like(u.vector().get_local())
    for i, x in enumerate(V.tabulate_dof_coordinates()):
        if i % 2 == 0:
            new_dofs[i] = f(x) + 10.0*x[0]
        else:
            new_dofs[i] = -f(x) - 10.0*x[0]
    u.vector().set_local(new_dofs)

    qh = df.interpolate(u, Q)
    # gh2 = df.Function(gh.function_space())
    clem_interp()
    # print(gh.vector().get_local() - gh_old_dofs)

    gh2 = clem_interp()
    # print(gh2.vector().get_local() - gh_old_dofs)


    # Ends up with gh2 == gh for some reason, don't know how to reuse the interpolation material then.


    return


if __name__ == "__main__":
    # clement_visualization()
    test_clement_grad()
    test_clement_hess()
    # check_reuse_matrix()
