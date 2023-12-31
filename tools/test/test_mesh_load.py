import numpy as np
import dolfin as df

from tools.loading import *
from conf import mesh_file_loc, biharmonic_file_loc, biharmonic_label

import pytest

def test_mesh_load_meshview():
    pytest.importorskip("dolfin.MeshView")

    _, mesh_mv, _ = load_mesh_meshview(mesh_file_loc)
    assert mesh_mv.num_vertices() == 3935

    V_mv_cg2 = df.VectorFunctionSpace(mesh_mv, "CG", 2, 2)
    assert V_mv_cg2.tabulate_dof_coordinates().shape == (30600, 2)

    V_mv_cg1 = df.VectorFunctionSpace(mesh_mv, "CG", 1, 2)
    assert V_mv_cg1.tabulate_dof_coordinates().shape == (7870, 2)
    
    return

def test_mesh_load_submesh():

    mesh_sm = load_mesh_submesh(mesh_file_loc)
    assert mesh_sm.num_vertices() == 3935

    V_sm_cg2 = df.VectorFunctionSpace(mesh_sm, "CG", 2, 2)
    assert V_sm_cg2.tabulate_dof_coordinates().shape == (30600, 2)

    V_sm_cg1 = df.VectorFunctionSpace(mesh_sm, "CG", 1, 2)
    assert V_sm_cg1.tabulate_dof_coordinates().shape == (7870, 2)

    return

def test_create_meshview_submesh_conversion_array():
    pytest.importorskip("dolfin.MeshView")

    _, mesh_mv, _ = load_mesh_meshview(mesh_file_loc)
    mesh_sm = load_mesh_submesh(mesh_file_loc)


    V_mv = df.VectorFunctionSpace(mesh_mv, "CG", 2, 2)
    V_sm = df.VectorFunctionSpace(mesh_sm, "CG", 2, 2)

    u_mv = df.Function(V_mv)
    u_sm = df.Function(V_sm)

    with df.XDMFFile(biharmonic_file_loc) as infile:
        infile.read_checkpoint(u_mv, biharmonic_label, 0)

    inds = create_meshview_submesh_conversion_array(mesh_file_loc, "CG2")

    assert inds.shape[0] == V_mv.tabulate_dof_coordinates().shape[0] // 2

    new_dofs = np.zeros_like(u_mv.vector().get_local())
    new_dofs[::2] = u_mv.vector().get_local()[::2][inds]
    new_dofs[1::2] = u_mv.vector().get_local()[1::2][inds]
    u_sm.vector().set_local(np.copy(new_dofs))

    assert df.errornorm(u_mv, u_sm) < 1e-15

    V_mv_cg1 = df.VectorFunctionSpace(mesh_mv, "CG", 1, 2)
    V_sm_cg1 = df.VectorFunctionSpace(mesh_sm, "CG", 1, 2)

    u_mv_cg1 = df.Function(V_mv_cg1)
    u_sm_cg1 = df.Function(V_sm_cg1)

    u_mv_cg1.interpolate(u_mv)

    inds_cg1 = create_meshview_submesh_conversion_array(mesh_file_loc, "CG1")
    assert inds_cg1.shape[0] == V_mv_cg1.tabulate_dof_coordinates().shape[0] // 2

    new_dofs = np.zeros_like(u_mv_cg1.vector().get_local())
    new_dofs[::2] = u_mv_cg1.vector().get_local()[::2][inds_cg1]
    new_dofs[1::2] = u_mv_cg1.vector().get_local()[1::2][inds_cg1]
    u_sm_cg1.vector().set_local(np.copy(new_dofs))

    assert df.errornorm(u_mv_cg1, u_sm_cg1) < 1e-15

    return

def visualize_meshview_submesh_conversion_array():

    _, mesh_mv, _ = load_mesh_meshview(mesh_file_loc)
    mesh_sm = load_mesh_submesh(mesh_file_loc)


    V_mv = df.VectorFunctionSpace(mesh_mv, "CG", 2, 2)
    V_sm = df.VectorFunctionSpace(mesh_sm, "CG", 2, 2)

    u_mv = df.Function(V_mv)
    u_sm = df.Function(V_sm)

    with df.XDMFFile(biharmonic_file_loc) as infile:
        infile.read_checkpoint(u_mv, biharmonic_label, 0)

    inds = create_meshview_submesh_conversion_array(mesh_file_loc, "CG2")

    new_dofs = np.zeros_like(u_mv.vector().get_local())
    new_dofs[::2] = u_mv.vector().get_local()[::2][inds]
    new_dofs[1::2] = u_mv.vector().get_local()[1::2][inds]
    u_sm.vector().set_local(np.copy(new_dofs))

    file_mv = df.File("fenics_output/meshview_load.pvd")
    file_mv << u_mv
    file_sm = df.File("fenics_output/submesh_load.pvd")
    file_sm << u_sm

    return


if __name__ == "__main__":
    test_mesh_load_meshview()
    test_mesh_load_submesh()
    test_create_meshview_submesh_conversion_array()
    # visualize_meshview_submesh_conversion_array()
