import dolfin as df
import numpy as np

from elasticity import solve_neohook_solid, solve_linear_solid


if __name__ == "__main__":

    working_dir = "data_prep/artificial/working_space"

    solid_mesh = df.Mesh()
    with df.HDF5File(solid_mesh.mpi_comm(), working_dir+"/solid.h5", 'r') as h5:
        h5.read(solid_mesh, 'mesh', False)

    tdim = solid_mesh.topology().dim()
    solid_boundaries = df.MeshFunction('size_t', solid_mesh, tdim-1, 0)
    with df.HDF5File(solid_mesh.mpi_comm(), working_dir+"/solid.h5", 'r') as h5:
        h5.read(solid_boundaries, 'boundaries')

    # ----6----
    # 4       9
    # ----6----

    displacement_bcs = {4: df.Constant((0, 0))}
    volume_load = df.Expression(('0', '-A*x[0]'), degree=1, A=1E-3*10*-7.2*0)
    x0_flag, x1_flag = 0.24898979485, 0.6
    L_flag = x1_flag - x0_flag
    # six_load = df.Expression(('0', 'C*sin(1*pi/L * (x[0] - B))*sin(1*pi/L * (x[0] - B))'), degree=1, L=L_flag, B=x0_flag, C=8E-4)
    six_load = df.Expression(('0', 'abs(x[0] - 0.45) < 0.05? C: 0'), degree=1, C=4E2)
    surface_load = {9: df.Constant((0, -2.12E3)), 6: six_load}

    solvers = {"neohook": solve_neohook_solid, "linear": solve_linear_solid}
    solid_type = "neohook"
    # solid_type = "linear"

    uh = solvers[solid_type](boundaries=solid_boundaries,
                             volume_load=volume_load,
                             surface_load=surface_load,
                             displacement_bcs=displacement_bcs,
                             mu=df.Constant(0.5e6),
                             lmbda=df.Constant(2.0e6), pdegree=2)


    print(uh.vector().norm('l2'))
    uh.rename('uh', '')
    df.File(working_dir+'/displacement.pvd') << uh

    # quit()


    # Let's try with Harmonic extension
    fluid_mesh = df.Mesh()
    with df.HDF5File(fluid_mesh.mpi_comm(), working_dir+'/fluid.h5', 'r') as h5:
        h5.read(fluid_mesh, 'mesh', False)

    tdim = fluid_mesh.topology().dim()
    fluid_boundaries = df.MeshFunction('size_t', fluid_mesh, tdim-1, 0)
    with df.HDF5File(fluid_mesh.mpi_comm(), working_dir+'/fluid.h5', 'r') as h5:
        h5.read(fluid_boundaries, 'boundaries')

    fluid_tags = set(fluid_boundaries.array()) - set((0, ))
    iface_tags = {6, 9}
    zero_displacement_tags = fluid_tags - iface_tags

    # Represent the solid data on fluid mesh
    from make_mesh import translate_function

    uh_fluid = translate_function(from_u=uh,
                                  from_facet_f=solid_boundaries,
                                  to_facet_f=fluid_boundaries,
                                  shared_tags=iface_tags)

    V = df.VectorFunctionSpace(fluid_mesh, 'CG', 2)
    u, v = df.TrialFunction(V), df.TestFunction(V)

    a = df.inner(df.grad(u), df.grad(v))*df.dx
    L = df.inner(df.Constant((0, )*len(u)), v)*df.dx
    # Those from solid
    bcs = [df.DirichletBC(V, uh_fluid, fluid_boundaries, tag) for tag in iface_tags]
    # The rest is fixed
    null = df.Constant((0, )*len(u))
    bcs.extend([df.DirichletBC(V, null, fluid_boundaries, tag) for tag in zero_displacement_tags])

    uh = df.Function(V)
    df.solve(a == L, uh, bcs)
    uh.rename('uh', '')

    df.File(working_dir+'/extended.pvd') << uh
