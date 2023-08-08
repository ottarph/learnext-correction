import dolfin as df


def jacobian(f, mesh=None):
    '''Eval det(grad(f)) at the cell midpoint'''
    if mesh is None:
        mesh = f.function_space().mesh()
        
    dx = df.dx(domain=mesh, metadata={'quadrature_degree': 0})
    K = df.CellVolume(mesh)

    Q = df.FunctionSpace(mesh, 'DG', 0)
    q = df.TestFunction(Q)
    jac = df.Function(Q)
    # Inexact L^2 projection
    df.assemble((1/K)*df.inner(df.det(df.grad(f)), q)*dx, tensor=jac.vector())
    # Now we have per cell approximation to dv/dV where dV is the cell volume
    # prior to deformation and dv is post.
    return jac

# --------------------------------------------------------------------

if __name__ == '__main__':
    import numpy as np
    
    mesh = df.UnitSquareMesh(3, 3)
    x, y = df.SpatialCoordinate(mesh)

    u = df.as_vector((-y, x))
    j = jacobian(u, mesh=mesh)
    # Rotations are rigid dv = dV
    assert np.all(j.vector().get_local() == 1.)

    # For statistics ...
    print(j.vector().min(), j.vector().max())
