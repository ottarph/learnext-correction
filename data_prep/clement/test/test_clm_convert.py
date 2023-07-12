from data_prep.clement.convert import *

def test_cg1_vector_to_array():

    N = 10
    mesh = df.UnitSquareMesh(N, N)

    V = df.VectorFunctionSpace(mesh, "CG", 1, 2)
    u = df.Function(V)

    u.interpolate(df.Constant((1.0, -1.0)))
    u_np = CG1_vector_to_array(u)

    assert u_np.shape == (mesh.num_vertices(), 2)
    assert np.all(np.equal(u_np[:,0], 1.0))
    assert np.all(np.equal(u_np[:,1], -1.0))

    return

def test_cg2_cg1_interpolation():

    N = 10
    mesh = df.UnitSquareMesh(N, N)

    V_cg2 = df.VectorFunctionSpace(mesh, "CG", 2, 2)
    V_cg1 = df.VectorFunctionSpace(mesh, "CG", 1, 2)

    u_cg2 = df.Function(V_cg2)
    new_dofs = np.zeros_like(u_cg2.vector().get_local())
    for i, x in enumerate(V_cg2.tabulate_dof_coordinates()):
        if i % 2 == 0:
            new_dofs[i] = 3.0*x[0]**2 + 2.0*x[1]
        else:
            new_dofs[i] = -3.0*x[0]**2 - 2.0*x[1]
    u_cg2.vector().set_local(new_dofs)

    u_cg1 = df.interpolate(u_cg2, V_cg1)

    assert np.all(np.isclose(u_cg2.compute_vertex_values(), u_cg1.compute_vertex_values(), atol=1e-15))

    return

def test_CG1_vector_plus_grad_to_array():


    def f(x):
        return 3.0*x[0] + 2.0*x[1]

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

    new_arr = CG1_vector_plus_grad_to_array(u, gh)


    assert np.all(np.isclose(new_arr[:,2],  3.0, atol=1e-15)) # Offset   two is d_x u_x
    assert np.all(np.isclose(new_arr[:,3],  2.0, atol=1e-15)) # Offset three is d_x u_y
    assert np.all(np.isclose(new_arr[:,4], -3.0, atol=1e-15)) # Offset  four is d_y u_x
    assert np.all(np.isclose(new_arr[:,5], -2.0, atol=1e-15)) # Offset  five is d_y u_y

    # Dof locations for u and gh are the same
    assert np.all(np.equal(u.function_space().tabulate_dof_coordinates()[::2,:], gh.function_space().tabulate_dof_coordinates()[::4,:]))

    test_arr = np.zeros_like(new_arr[:,:2])
    for i, x in enumerate(V.tabulate_dof_coordinates()[::2]):
        test_arr[i,0] =  f(x)
        test_arr[i,1] = -f(x)

    assert np.all(np.isclose(new_arr[:,0], test_arr[:,0], atol=1e-15))
    assert np.all(np.isclose(new_arr[:,1], test_arr[:,1], atol=1e-15))

    return

def test_CG1_vector_plus_grad_hess_to_array():

    def f(x):
        return 3.0*x[0]**2 + 2.0*x[1]**2 - 1.0*x[0]*x[1]

    N = 10
    mesh = df.UnitSquareMesh(N, N)

    V_cg2 = df.VectorFunctionSpace(mesh, "CG", 2, 2)
    V_cg1 = df.VectorFunctionSpace(mesh, "CG", 1, 2)

    u = df.Function(V_cg2)
    new_dofs = np.zeros_like(u.vector().get_local())
    for i, x in enumerate(V_cg2.tabulate_dof_coordinates()):
        if i % 2 == 0:
            new_dofs[i] = f(x)
        else:
            new_dofs[i] = -f(x)
    u.vector().set_local(new_dofs)

    u_cg1 = df.interpolate(u, V_cg1)

    Q_dg1 = df.VectorFunctionSpace(mesh, "DG", 1, 2)
    Q_dg2 = df.VectorFunctionSpace(mesh, "DG", 2, 2)
    qh_dg1 = df.interpolate(u, Q_dg1)
    qh_dg2 = df.interpolate(u, Q_dg2)

    gh = clement_interpolate(df.grad(qh_dg1))
    # gh.compute_vertex_values() = (d_x u_x, d_y u_x, d_x u_y, d_y u_y, ...[*121])
    # gh.vector().get_local() = (d_x u_x, ...[*121], d_y u_x, ...[*121], d_x u_y, ...[*121], d_y u_y, ...[*121])

    qh_dg2_1, qh_dg2_2 = qh_dg2.split()
    hess_h_1 = clement_interpolate(df.grad(df.grad(qh_dg2_1)))
    hess_h_2 = clement_interpolate(df.grad(df.grad(qh_dg2_2)))

    new_arr = CG1_vector_plus_grad_hess_to_array(u_cg1, gh, hess_h_1, hess_h_2)

    # assert np.all(np.isclose(new_arr[:,2],  3.0, atol=1e-15)) # Offset   two is d_x u_x
    # assert np.all(np.isclose(new_arr[:,3],  2.0, atol=1e-15)) # Offset three is d_x u_y
    # assert np.all(np.isclose(new_arr[:,4], -3.0, atol=1e-15)) # Offset  four is d_y u_x
    # assert np.all(np.isclose(new_arr[:,5], -2.0, atol=1e-15)) # Offset  five is d_y u_y
    # The gradient is not constant for the f(x) in this test.
    # Should be correct as in the other test though

    assert np.all(np.isclose(new_arr[:,6],  6.0, atol=1e-15)) # Offset six is d_xx u_x
    assert np.all(np.isclose(new_arr[:,7], -1.0, atol=1e-15)) # Offset six is d_xy u_x
    assert np.all(np.isclose(new_arr[:,8], -1.0, atol=1e-15)) # Offset six is d_yx u_x
    assert np.all(np.isclose(new_arr[:,9],  4.0, atol=1e-15)) # Offset six is d_yy u_x

    assert np.all(np.isclose(new_arr[:,10], -6.0, atol=1e-15)) # Offset six is d_xx u_y
    assert np.all(np.isclose(new_arr[:,11],  1.0, atol=1e-15)) # Offset six is d_xy u_y
    assert np.all(np.isclose(new_arr[:,12],  1.0, atol=1e-15)) # Offset six is d_yx u_y
    assert np.all(np.isclose(new_arr[:,13], -4.0, atol=1e-15)) # Offset six is d_yy u_y

    # Dof locations for u_cg1, gh, hess_h_1, hess_h_2 are the same
    assert np.all(np.equal(u_cg1.function_space().tabulate_dof_coordinates()[::2,:], 
                           gh.function_space().tabulate_dof_coordinates()[::4,:]))
    assert np.all(np.equal(u_cg1.function_space().tabulate_dof_coordinates()[::2,:],
                           hess_h_1.function_space().tabulate_dof_coordinates()[::4,:]))
    assert np.all(np.equal(u_cg1.function_space().tabulate_dof_coordinates()[::2,:],
                           hess_h_2.function_space().tabulate_dof_coordinates()[::4,:]))

    test_arr = np.zeros_like(new_arr[:,:2])
    for i, x in enumerate(V_cg1.tabulate_dof_coordinates()[::2]):
        test_arr[i,0] =  f(x)
        test_arr[i,1] = -f(x)

    assert np.all(np.isclose(new_arr[:,0], test_arr[:,0], atol=1e-15))
    assert np.all(np.isclose(new_arr[:,1], test_arr[:,1], atol=1e-15))

    return

if __name__ == "__main__":
    test_cg1_vector_to_array()
    test_cg2_cg1_interpolation()
    test_CG1_vector_plus_grad_to_array()
    test_CG1_vector_plus_grad_hess_to_array()
