import dolfin as df
import numpy as np
from pathlib import Path

from tqdm import tqdm


if __name__ == "__main__":

    working_dir = "data_prep/artificial/working_space"

    solid_mesh = df.Mesh()
    with df.HDF5File(solid_mesh.mpi_comm(), working_dir+'/solid.h5', 'r') as h5:
        h5.read(solid_mesh, 'mesh', False)

    tdim = solid_mesh.topology().dim()
    solid_boundaries = df.MeshFunction('size_t', solid_mesh, tdim-1, 0)
    with df.HDF5File(solid_mesh.mpi_comm(), working_dir+'/solid.h5', 'r') as h5:
        h5.read(solid_boundaries, 'boundaries')

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

    V_solid_cg2 = df.VectorFunctionSpace(solid_mesh, "CG", 2)
    uh_solid = df.Function(V_solid_cg2)

    V_fluid_cg2 = df.VectorFunctionSpace(fluid_mesh, "CG", 2)
    V_fluid_cg1 = df.VectorFunctionSpace(fluid_mesh, "CG", 1)

    """ Make Laplace solver """
    u_harm_cg1 = df.Function(V_fluid_cg1)
    u_harm_cg2 = df.Function(V_fluid_cg2)

    u_cg2 = df.TrialFunction(V_fluid_cg2)
    v_cg2 = df.TestFunction(V_fluid_cg2)

    a_harm = df.inner( df.grad(u_cg2), df.grad(v_cg2) ) * df.dx
    l_harm = df.inner( df.Constant((0.0, 0.0)), v_cg2) * df.dx

    A_harm = df.as_backend_type(df.assemble(a_harm))
    L_harm = df.assemble(l_harm)
    
    solver_harm = df.PETScLUSolver(A_harm)

    """ Make biharmonic solver """
    u_biharm_cg1 = df.Function(V_fluid_cg1)
    u_biharm_cg2 = df.Function(V_fluid_cg2)

    # Mixed formulation
    T = df.VectorElement("CG", fluid_mesh.ufl_cell(), 2)
    FS = df.FunctionSpace(fluid_mesh, df.MixedElement(T, T))
    uz = df.TrialFunction(FS)
    puz = df.TestFunction(FS)
    u, z = df.split(uz)
    psiu, psiz = df.split(puz)

    uz_h = df.Function(FS)

    a_biharm = df.inner( df.grad(z), df.grad(psiu) ) * df.dx + \
                df.inner(z, psiz) * df.dx + \
                -df.inner( df.grad(u), df.grad(psiz) ) * df.dx
    l_biharm = df.inner( df.Constant((0.0, 0.0)), psiu) * df.dx

    A_biharm = df.as_backend_type(df.assemble(a_biharm))
    L_biharm = df.assemble(l_biharm)

    solver_biharm = df.PETScLUSolver(A_biharm)


    with df.XDMFFile(working_dir+"/displacements.xdmf") as infile:

        p = Path(working_dir+"/harmonic1.xdmf")
        if p.exists(): p.unlink()
        p = Path(working_dir+"/biharmonic1.xdmf")
        if p.exists(): p.unlink()
        outfile_harm = df.XDMFFile(working_dir+"/harmonic1.xdmf")
        outfile_biharm = df.XDMFFile(working_dir+"/biharmonic1.xdmf")

        for k in tqdm(range(101)):
            infile.read_checkpoint(uh_solid, "uh", k)

            uh_fluid = translate_function(from_u=uh_solid,
                                  from_facet_f=solid_boundaries,
                                  to_facet_f=fluid_boundaries,
                                  shared_tags=iface_tags)
            

            bcs = [df.DirichletBC(V_fluid_cg2, uh_fluid, fluid_boundaries, tag) for tag in iface_tags]
            null = df.Constant((0.0, 0.0))
            bcs.extend([df.DirichletBC(V_fluid_cg2, null, fluid_boundaries, tag) for tag in zero_displacement_tags])

            for bc in bcs:
                bc.apply(A_harm)
                bc.apply(L_harm)

            solver_harm.solve(u_harm_cg2.vector(), L_harm)
            u_harm_cg1 = df.interpolate(u_harm_cg2, V_fluid_cg1)
            outfile_harm.write_checkpoint(u_harm_cg1, "u_harm_cg1", k, append=True)

            bcs = [df.DirichletBC(FS.sub(0), uh_fluid, fluid_boundaries, tag) for tag in iface_tags]
            null = df.Constant((0.0, 0.0))
            bcs.extend([df.DirichletBC(FS.sub(0), null, fluid_boundaries, tag) for tag in zero_displacement_tags])

            for bc in bcs:
                bc.apply(A_biharm)
                bc.apply(L_biharm)

            solver_biharm.solve(uz_h.vector(), L_biharm)
            u_biharm_cg2, zh = uz_h.split(deepcopy=True)
            u_biharm_cg1 = df.interpolate(u_biharm_cg2, V_fluid_cg1)
            outfile_biharm.write_checkpoint(u_biharm_cg1, "u_biharm_cg1", k, append=True)


        outfile_harm.close()
        outfile_biharm.close()
