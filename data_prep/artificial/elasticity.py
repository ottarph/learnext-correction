from dolfin import *

parameters['form_compiler']['cpp_optimize'] = True
ffc_options = {'optimize': True, 'eliminate_zeros': True, 'precompute_basis_const': True,
               'precompute_ip_const': True}


def solve_linear_solid(boundaries, volume_load, surface_load, displacement_bcs, mu=Constant(1), lmbda=Constant(1), pdegree=2):
    '''Foo'''
    mesh = boundaries.mesh()
    V = VectorFunctionSpace(mesh, 'CG', pdegree)
    u, v = TrialFunction(V), TestFunction(V)

    eps = lambda v: sym(grad(v))
    a = 2*mu*inner(eps(u), eps(v))*dx + lmbda*inner(div(u), div(v))*dx

    if not volume_load:
        volume_load = Constant((0, )*len(u))
    L = inner(volume_load, v)*dx

    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    if surface_load:
        for tag, surface_force in surface_load.items():
            L += inner(surface_force, v)*ds(tag)

    bcs = [DirichletBC(V, value, boundaries, tag)
           for tag, value in displacement_bcs.items()]

    uh = Function(V)
    solve(a == L, uh, bcs, form_compiler_parameters=ffc_options)

    return uh


def solve_neohook_solid(boundaries, volume_load, surface_load, displacement_bcs, mu=Constant(1), lmbda=Constant(1), pdegree=2):
    '''Foo'''
    mesh = boundaries.mesh()
    V = VectorFunctionSpace(mesh, 'CG', pdegree)
    u, v = Function(V), TestFunction(V)

    eps = lambda v: sym(grad(v))

    I = Identity(len(u))
    F = I + grad(u)     
    C = F.T*F           

    # Invariants of deformation tensors
    Ic = tr(C)
    J  = det(F)

    # Stored strain energy density (compressible neo-Hookean model)
    psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

    # Total potential energy
    if not volume_load:
        volume_load = Constant((0, )*len(u))
    Pi = psi*dx - dot(volume_load, u)*dx

    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)    
    if surface_load:
        for tag, surface_force in surface_load.items():
            Pi += -inner(surface_force, u)*ds(tag)

    F = derivative(Pi, u, v)

    du = TrialFunction(V)
    # Compute Jacobian of F
    J = derivative(F, u, du)

    bcs = [DirichletBC(V, value, boundaries, tag)
           for tag, value in displacement_bcs.items()]
    
    # Solve variational problem
    solve(F == 0, u, bcs, J=J, form_compiler_parameters=ffc_options,
          solver_parameters={"nonlinear_solver": "newton", "newton_solver":
            {"maximum_iterations": 30}})

    return u

# --------------------------------------------------------------------

if __name__ == '__main__':
    import numpy as np
    
    solid_mesh = Mesh()
    with HDF5File(solid_mesh.mpi_comm(), 'data_prep/artificial/working_space/solid.h5', 'r') as h5:
        h5.read(solid_mesh, 'mesh', False)

    tdim = solid_mesh.topology().dim()
    solid_boundaries = MeshFunction('size_t', solid_mesh, tdim-1, 0)
    with HDF5File(solid_mesh.mpi_comm(), 'data_prep/artificial/working_space/solid.h5', 'r') as h5:
        h5.read(solid_boundaries, 'boundaries')

    # ----6----
    # 4       9
    # ----6----

    displacement_bcs = {4: Constant((0, 0))}
    volume_load = Expression(('0', '-A*x[0]'), degree=1, A=1E-3)
    surface_load = {9: Constant((0, 1E-4))}

    uh = solve_neohook_solid(boundaries=solid_boundaries,
                             volume_load=volume_load,
                             surface_load=surface_load,
                             displacement_bcs=displacement_bcs,
                             mu=Constant(1),
                             lmbda=Constant(1), pdegree=2)

    print(uh.vector().norm('l2'))
    uh.rename('uh', '')
    File('displacement.pvd') << uh


    # Let's try with Harmonic extension
    fluid_mesh = Mesh()
    with HDF5File(fluid_mesh.mpi_comm(), 'data_prep/artificial/working_space/fluid.h5', 'r') as h5:
        h5.read(fluid_mesh, 'mesh', False)

    tdim = fluid_mesh.topology().dim()
    fluid_boundaries = MeshFunction('size_t', fluid_mesh, tdim-1, 0)
    with HDF5File(fluid_mesh.mpi_comm(), 'data_prep/artificial/working_space/fluid.h5', 'r') as h5:
        h5.read(fluid_boundaries, 'boundaries')

    fluid_tags = set(fluid_boundaries.array()) - set((0, ))
    iface_tags = {6, 9}
    zero_displacement_tags = fluid_tags - iface_tags

    print(fluid_mesh.num_vertices())
    # Represent the solid data on fluid mesh
    from make_mesh import translate_function

    uh_fluid = translate_function(from_u=uh,
                                  from_facet_f=solid_boundaries,
                                  to_facet_f=fluid_boundaries,
                                  shared_tags=iface_tags)

    V = VectorFunctionSpace(fluid_mesh, 'CG', 2)
    u, v = TrialFunction(V), TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = inner(Constant((0, )*len(u)), v)*dx
    # Those from solid
    bcs = [DirichletBC(V, uh_fluid, fluid_boundaries, tag) for tag in iface_tags]
    # The rest is fixed
    null = Constant((0, )*len(u))
    bcs.extend([DirichletBC(V, null, fluid_boundaries, tag) for tag in zero_displacement_tags])

    uh = Function(V)
    solve(a == L, uh, bcs)

    File('extended.pvd') << uh
